"""来源状态的单一事实源：权威在验证记录，`corpus_meta` 只是副本（计划 D4）。

这一组测试守的是**判定规矩**，不是返回值长相：

* 权威只能读验证记录。把副本改成 `verified_zip` 而验证记录不存在时，
  结论必须是 `source_unverified` —— 副本不许把自己升格成证据。
* 两份记录不一致时不许静默取一边：必须留下冲突记录，且结论取更保守的那侧。
  这条是关键：如果实现偷懒取"权威"，那么"副本过期"这种真事故就没人看得见。
* 拼错的状态值（`verified_zip ` 带空格、`VERIFIED_ZIP` 大写、空串）一律按最保守处理。
  状态是字符串比较出来的，一个尾空格就能把 `strength` 抬到"已验证"。
* `verification_path()` 必须跟随 `config.DATA_DIR` 的重定向，否则整条链路在测试里
  读的是真实仓库，测什么都是真值的回声。
"""
from __future__ import annotations

import json

import pytest

from arkrag import config as config_mod
from arkrag import source


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """把 DATA_DIR 指到临时目录：验证记录与副本都写在这里。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(config_mod, "DATA_DIR", data_dir)
    return data_dir


def _write(path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------- 权威与副本


def test_verification_record_is_authority_when_copy_absent(sandbox):
    """冻结 corpus_meta（无 source_status 字段）时必须从验证记录取到已验证。

    这就是本任务要修的 bug：副本缺字段被读成 source_unverified，
    于是一份 ZIP 字节与解压树两边对得上的语料被拒发布。
    """
    _write(
        sandbox / "source_verification.json",
        {"status": "verified_zip", "checks": ["a"], "problems": []},
    )
    resolved = source.resolve_for_build({"corpus_fp": "abc123"})

    assert resolved["status"] == "verified_zip"
    assert resolved["authority_status"] == "verified_zip"
    assert resolved["authority_present"] is True
    assert resolved["copy_present"] is False
    assert resolved["conflicts"] == []
    assert resolved["verified"] is True


def test_copy_alone_cannot_self_verify(sandbox):
    """没有验证记录时，副本说自己 verified_zip 也不作数。"""
    resolved = source.resolve_for_build({"source_status": "verified_zip"})

    assert resolved["authority_present"] is False
    assert resolved["status"] == "source_unverified"
    assert resolved["verified"] is False
    # 副本声称与权威缺失之间的落差必须被记录，不能悄悄降级
    assert len(resolved["conflicts"]) == 1
    assert "verified_zip" in resolved["conflicts"][0]
    assert "source_unverified" in resolved["conflicts"][0]


def test_stronger_authority_with_stale_weaker_copy_is_reported_and_downgraded(sandbox):
    """权威已验证、副本停留在旧结论：必须报冲突，并采用更保守的那个。"""
    _write(sandbox / "source_verification.json", {"status": "verified_zip"})

    resolved = source.resolve_for_build({"source_status": "source_unverified"})

    assert resolved["authority_status"] == "verified_zip"
    assert resolved["copy_status"] == "source_unverified"
    assert resolved["status"] == "source_unverified"  # 更保守 = 不发布
    assert len(resolved["conflicts"]) == 1


def test_weaker_authority_with_stale_stronger_copy_is_downgraded(sandbox):
    """权威退化成未验证、副本还留着旧的 verified_zip：同样取保守侧。"""
    _write(sandbox / "source_verification.json", {"status": "source_unverified"})

    resolved = source.resolve_for_build({"source_status": "verified_zip"})

    assert resolved["status"] == "source_unverified"
    assert resolved["verified"] is False
    assert len(resolved["conflicts"]) == 1


def test_agreeing_records_produce_no_conflict(sandbox):
    """两边一致时不许无中生有地报冲突。"""
    _write(sandbox / "source_verification.json", {"status": "verified_tree"})

    resolved = source.resolve_for_build({"source_status": "verified_tree"})

    assert resolved["status"] == "verified_tree"
    assert resolved["conflicts"] == []
    assert resolved["verified"] is True


def test_present_but_empty_copy_is_reported(sandbox):
    """字段在但值为空，是"有人写了空"，不是"没有字段"，要报出来。"""
    _write(sandbox / "source_verification.json", {"status": "verified_zip"})

    resolved = source.resolve_for_build({"source_status": ""})

    assert resolved["status"] == "verified_zip"
    assert resolved["copy_present"] is True
    assert len(resolved["conflicts"]) == 1
    assert "值为空" in resolved["conflicts"][0]


# --------------------------------------------------------------- 状态强度


@pytest.mark.parametrize(
    "weaker_status,stronger_status",
    [
        ("source_unverified", "verified_tree"),
        ("verified_tree", "verified_zip"),
        ("source_unverified", "verified_zip"),
    ],
)
def test_weaker_picks_the_more_conservative(weaker_status, stronger_status):
    assert source.weaker(weaker_status, stronger_status) == weaker_status
    assert source.weaker(stronger_status, weaker_status) == weaker_status


@pytest.mark.parametrize("bad", ["", "VERIFIED_ZIP", "verified_zip ", "trust_me", "verified"])
def test_unknown_or_misspelled_status_is_never_treated_as_verified(bad):
    """未知拼写按最保守算：一个尾空格不许把状态抬成已验证。"""
    assert source.strength(bad) == 0
    # 与已验证并列时，未知拼写必须是拖低的那一侧
    assert source.weaker(bad, "verified_zip") == bad


def test_misspelled_status_in_record_downgrades_build(sandbox):
    """验证记录里写成 `verified_zip `（尾空格）也不许通过。"""
    _write(sandbox / "source_verification.json", {"status": "verified_zip "})

    resolved = source.resolve_for_build({})

    assert resolved["status"] == "verified_zip "
    assert resolved["verified"] is False


# --------------------------------------------------------------- 路径重定向


def test_verification_path_follows_data_dir_redirect(sandbox):
    """路径必须是算出来的，不能是 import 时烧死的常量。"""
    assert source.verification_path() == sandbox / "source_verification.json"


def test_status_for_build_reads_redirected_location(sandbox):
    """写进临时 DATA_DIR 的记录必须被读到——若真去读仓库，这里会拿到真实值。"""
    assert source.load() == {}

    _write(sandbox / "source_verification.json", {"status": "verified_tree"})
    status, report = source.status_for_build()

    assert status == "verified_tree"
    assert report["status"] == "verified_tree"


def test_corrupt_verification_record_degrades_to_unverified(sandbox):
    """记录坏了就是未知，不许当已验证用。"""
    (sandbox / "source_verification.json").write_text("{not json", encoding="utf-8")

    status, report = source.status_for_build()

    assert status == "source_unverified"
    assert report == {}
