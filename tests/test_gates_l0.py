"""L0 门禁的形状与「能失败」验证。

## 为什么单独有这一组

计划原则 5：**所有门禁必须能失败，禁止"只打印问题但退出码为 0"冒充验收。**
一条恒 PASS 的门比没有门更糟——它让人以为这件事被检查过了。

这里不测"真实语料上的读数应该是多少"（那由 `run_l0.py` 在真机上负责），
只测**判据本身**：给不同的输入，门必须给出不同的结论。一个对所有输入
都返回 PASS 的门会在这里立刻露馅。

## L0.8 为什么值得单独反复测

来源状态有两个写入方（`source_verification.json` 权威 + `corpus_meta.json` 副本），
一个读取方（`gates.l0_gates`），一个发布门（`release.verify_generation`）。
这四个地方曾经各自手抄字面量，于是**同一个拼写错误在有的地方算已验证、
在有的地方算未验证**。现在它们共用 `source` 的常量，这两组参数化用例
就是那条纪律的守门人：任何人把 `source.VERIFIED_ZIP` 改回手抄字符串，
或把判据改回 `== "source_unverified"`，都会在这里失败。
"""

from __future__ import annotations

import pytest

from arkrag import gates, source


def _manifest(**overrides):
    """造一份最小但结构完整的 L0 manifest。

    字段取自 `l0_gates` 实际读取的键——少一个就会 KeyError，
    这样"门依赖哪些输入"这件事也顺便被固化下来了。
    """
    base = {
        "baseline": {
            "story_files": 3,
            "story_name_missing": 0,
            "dialogue_turns": 10,
            "narration_turns": 5,
        },
        "encoding": {"bom_files": [], "decode_failures": []},
        "dropped_files": [],
        "source_verification": {"status": source.VERIFIED_ZIP},
    }
    base.update(overrides)
    return base


def _gate(manifest, prefix: str):
    run = gates.l0_gates(manifest)
    matches = [r for r in run.rows() if r["name"].startswith(prefix)]
    assert matches, f"没有找到 {prefix} 开头的门；实际有 {[r['name'] for r in run.rows()]}"
    return matches[0]


class TestL08SourceVerification:
    @pytest.mark.parametrize("status", [source.VERIFIED_ZIP, source.VERIFIED_TREE])
    def test_verified_statuses_pass(self, status):
        assert _gate(_manifest(), "L0.8")["verdict"] == "PASS"
        assert (
            _gate(_manifest(source_verification={"status": status}), "L0.8")["verdict"] == "PASS"
        )

    @pytest.mark.parametrize(
        "status",
        [
            source.UNVERIFIED,  # 显式未验证
            "",  # 空串：没写，不等于验证过
            "VERIFIED_ZIP",  # 大小写不同
            "verified_zip ",  # 尾空格（笔误）
            "verified_zip\n",  # 换行
            "verified",  # 缩写
            "bogus",  # 完全不认识
            "unverified",
        ],
    )
    def test_any_unrecognized_status_fails(self, status):
        """门必须是白名单：只有明确的已验证取值才放行。

        只拦 `source_unverified` 一个拼写的写法等于没有门。
        """
        verdict = _gate(_manifest(source_verification={"status": status}), "L0.8")["verdict"]
        assert verdict == "FAIL", f"source_status={status!r} 竟然通过了来源门"

    def test_missing_verification_record_fails(self):
        """没有验证记录 = 身份未知，必须 FAIL 而不是默认通过。"""
        assert _gate(_manifest(source_verification={}), "L0.8")["verdict"] == "FAIL"

    def test_falls_back_to_manifest_source_status(self):
        """没有 source_verification 时读 manifest 顶层 source_status（旧产物兼容）。"""
        m = _manifest(source_verification={}, source_status=source.VERIFIED_TREE)
        assert _gate(m, "L0.8")["verdict"] == "PASS"
        m2 = _manifest(source_verification={}, source_status="bogus")
        assert _gate(m2, "L0.8")["verdict"] == "FAIL"

    def test_problems_are_reported_in_note(self):
        row = _gate(
            _manifest(
                source_verification={
                    "status": source.UNVERIFIED,
                    "problems": ["解压树与 ZIP 不一致：仅 ZIP 有 3"],
                }
            ),
            "L0.8",
        )
        assert "仅 ZIP 有 3" in row["note"]


class TestEveryL0GateCanFail:
    """每一条门都要有能触发 FAIL 的输入。

    这条测试的意义不是"再测一遍业务逻辑"，而是**防止有人新增一条恒 PASS 的门**：
    新增门时若没写进这里的清单，`test_no_uncovered_gates` 会失败并提醒补上。
    """

    # 门名 -> 能让它 FAIL 的 manifest 覆盖
    FAILING_INPUTS = {
        "L0.1": {"dropped_files": [{"doc_id": "x", "reason": ""}]},
        "L0.2": {"baseline": {**_manifest()["baseline"], "story_name_missing": 2}},
        "L0.3": {"encoding": {"bom_files": ["a.json"], "decode_failures": []}},
        "L0.4": {"encoding": {"bom_files": [], "decode_failures": ["b.json"]}},
        "L0.5": {
            "baseline": {
                **_manifest()["baseline"],
                "dialogue_turns": 0,
                "narration_turns": 0,
            }
        },
        # 轮次口径与字数口径剧烈分叉 = 旁白被成片降级成对话
        "L0.6": {
            "baseline": {
                **_manifest()["baseline"],
                "dialogue_turns": 10,
                "narration_turns": 1,
                "dialogue_chars": 100,
                "narration_chars": 1000,
            }
        },
        # 全部轮次都无说话人检测 = 无说话人检测失效
        "L0.6a": {"baseline": {**_manifest()["baseline"], "narration_turns": 0}},
        # 旁白占比过高 = 被全量降级为旁白
        "L0.6b": {
            "baseline": {
                **_manifest()["baseline"],
                "dialogue_turns": 5,
                "narration_turns": 5,
            }
        },
        "L0.7": {"reconciliation": {"some_index": {"unexplained": ["a.json"]}}},
        "L0.8": {"source_verification": {"status": source.UNVERIFIED}},
    }

    # 明确「只申报、不判定」的门。列在这里是**声明**，不是豁免通道：
    # 下面的 test_declared_informational_gates_are_actually_always_pass
    # 会验证它们真的恒 PASS，所以没人能用这个清单藏一条本该能红的门。
    INFORMATIONAL = {
        "L0.6c",  # 含无说话人轮次的文档数：纯对话剧本天然有，只申报（门名已标注）
    }

    @pytest.mark.parametrize("prefix", sorted(FAILING_INPUTS))
    def test_gate_fails_on_bad_input(self, prefix):
        assert _gate(_manifest(**self.FAILING_INPUTS[prefix]), prefix)["verdict"] == "FAIL"

    def test_declared_informational_gates_are_actually_always_pass(self):
        """被声明为"只申报"的门，必须真的恒 PASS。

        反过来验一遍：如果某条申报门其实会 FAIL，说明它是个判定门，
        就不该挂在申报清单里——那会让人以为它被检查过而实际没过。
        """
        for prefix in sorted(self.INFORMATIONAL):
            for m in (
                _manifest(),
                _manifest(source_verification={"status": "bogus"}),
                _manifest(dropped_files=[{"doc_id": "x", "reason": ""}]),
            ):
                assert _gate(m, prefix)["verdict"] == "PASS", (
                    f"{prefix} 被声明为只申报，却给出了 FAIL —— 它其实是一条判定门"
                )
            # 门名里必须能看出它只申报
            name = _gate(_manifest(), prefix)["name"]
            assert "申报" in name, f"{prefix} 是申报门，门名里应标明「仅申报」，实际是 {name!r}"

    def test_no_uncovered_gates(self):
        """新增的 L0 门必须写进 FAILING_INPUTS 或 INFORMATIONAL，否则这条会失败。

        这是"门禁只增不减、且每条都真的能红"的机械化保证。
        """
        names = [r["name"] for r in gates.l0_gates(_manifest()).rows()]
        prefixes = {n.split(" ")[0] for n in names}
        covered = set(self.FAILING_INPUTS) | set(self.INFORMATIONAL)
        missing = sorted(prefixes - covered)
        assert not missing, (
            f"这些 L0 门既没有'能触发 FAIL'的测试输入，也没被声明为只申报：{missing}。"
            "请在 FAILING_INPUTS 里补一条会让它判 FAIL 的 manifest 覆盖；"
            "如果它本来就只申报不判定，请加进 INFORMATIONAL 并在门名里标明「仅申报」。"
        )
