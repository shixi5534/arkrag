"""`scripts/ask.py::_latest_identity_generation` —— 默认世代选择。

为什么单独立一个文件守它：这个函数是 `--evaluate` **不带** `--generation` 时的唯一
入口，写错不会报错，只会静默评测另一个世代，然后照常打印七项指标。

原实现：

```python
for info in release.list_generations():
    manifest = release.Generation(info["generation_id"]).load()
    if ... == GRAPH_SCHEMA_IDENTITY:
        return str(info["generation_id"])   # 遇到第一个就返回
```

`list_generations()` 按目录名升序遍历，目录名形如 `g-<UTC 时间戳>-<hash>`，
升序即**最旧在前**。于是"取最新"实际返回最旧那个。

当时看不出来，是因为磁盘上三个 ark-rag-3 世代内容完全一致（`index_content_digest`
/ `probe_digest` / `graph_build_fp` / `chunk_manifest_hash` 全相同），跑谁的数字都一样。
**这是最坏的一种缺陷形态：结论正确，机制错误。** 只要出现第四个内容不同的世代，
默认评测就悄悄退化到旧数据，而报告里没有一个字段会变红。

本文件的用例全部**构造假世代**（不读盘、不碰真实 `data/`），并且都写成
「喂坏输入 → 必须变红」的形状，而不是只验证"好输入能过"。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _ask_module():
    """按路径加载 `scripts/ask.py`（它不是包的一部分）。"""
    path = ROOT / "scripts" / "ask.py"
    spec = importlib.util.spec_from_file_location("ask_gen_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ask():
    return _ask_module()


class _FakeGeneration:
    """只提供 `.load()` 的最小替身。"""

    def __init__(self, manifest: dict) -> None:
        self._manifest = manifest

    def load(self) -> dict:
        return self._manifest


def _install(monkeypatch, ask, rows: list[dict]) -> None:
    """把 `list_generations()` 与 `Generation` 换成给定的一组假世代。

    `rows` 每项是 `{generation_id, status, created_at, graph_schema}`。
    """
    listed = [
        {
            "generation_id": row["generation_id"],
            "status": row.get("status", "READY"),
            "created_at": row.get("created_at", ""),
        }
        for row in rows
    ]

    def fake_list():
        # 刻意按目录名升序返回——与真实实现一致，让"取到最旧"的缺陷能被复现
        return sorted(listed, key=lambda r: r["generation_id"])

    manifests = {
        row["generation_id"]: {
            "graph_schema": row.get("graph_schema", "ark-rag-3"),
            "created_at": row.get("created_at", ""),
        }
        for row in rows
    }

    monkeypatch.setattr(ask.release, "list_generations", fake_list)
    monkeypatch.setattr(
        ask.release, "Generation", lambda gid, _m=manifests: _FakeGeneration(_m[gid])
    )


NEWEST = "g-20261031T035231Z-4eafae"
MIDDLE = "g-20261031T034009Z-348206"
OLDEST = "g-20261031T033200Z-b22ca8"


class TestPicksNewest:
    def test_returns_newest_not_oldest(self, ask, monkeypatch):
        """核心回归：三个内容不同、时间不同的 ark-rag-3 世代，必须取最新的那个。

        原实现在这里返回 `OLDEST`（列表升序的第一个匹配项）。
        """
        _install(
            monkeypatch,
            ask,
            [
                {"generation_id": OLDEST, "created_at": "2026-10-31T03:32:00Z"},
                {"generation_id": MIDDLE, "created_at": "2026-10-31T03:40:09Z"},
                {"generation_id": NEWEST, "created_at": "2026-10-31T03:52:31Z"},
            ],
        )
        assert ask._latest_identity_generation() == NEWEST

    def test_directory_order_does_not_decide(self, ask, monkeypatch):
        """即使 `list_generations()` 返回**逆序**，结果也必须是最新的。

        这一条防的是"改成 list[-1] 就行"——那样只是把对目录名的依赖反过来，
        依然没有看 `created_at`。
        """
        rows = [
            {"generation_id": NEWEST, "created_at": "2026-10-31T03:52:31Z"},
            {"generation_id": MIDDLE, "created_at": "2026-10-31T03:40:09Z"},
            {"generation_id": OLDEST, "created_at": "2026-10-31T03:32:00Z"},
        ]
        _install(monkeypatch, ask, rows)
        # 直接把它翻成降序喂进去
        real_list = ask.release.list_generations
        monkeypatch.setattr(ask.release, "list_generations", lambda: list(reversed(real_list())))
        assert ask._latest_identity_generation() == NEWEST

    def test_created_at_beats_generation_id_order(self, ask, monkeypatch):
        """`created_at` 与目录名时间戳不一致时，以 `created_at` 为准。

        真实场景：世代目录可以被复制/改名，`created_at` 才是 Manifest 里记录的
        创建时刻。若两者冲突，按 Manifest 的记录走，不按目录名猜。
        """
        _install(
            monkeypatch,
            ask,
            [
                # 目录名时间戳更晚，但 Manifest 说它创建得更早
                {"generation_id": "g-20261231T235959Z-aaaaaaaa", "created_at": "2026-01-01T00:00:00Z"},
                {"generation_id": "g-20260101T000000Z-bbbbbbbb", "created_at": "2026-12-31T00:00:00Z"},
            ],
        )
        assert ask._latest_identity_generation() == "g-20260101T000000Z-bbbbbbbb"


class TestSchemaFilter:
    def test_skips_legacy_schema_even_if_newer(self, ask, monkeypatch):
        """`ark-rag-1` 世代更新也不能被选中——它没有断言，做不了可定位引用。"""
        _install(
            monkeypatch,
            ask,
            [
                {"generation_id": OLDEST, "created_at": "2026-10-31T03:32:00Z"},
                {
                    "generation_id": NEWEST,
                    "created_at": "2026-10-31T03:52:31Z",
                    "graph_schema": "ark-rag-1",
                },
            ],
        )
        assert ask._latest_identity_generation() == OLDEST

    def test_raises_when_only_legacy_exists(self, ask, monkeypatch):
        """全是 ark-rag-1 时必须明确失败，不能悄悄返回一个用不了的世代。"""
        _install(
            monkeypatch,
            ask,
            [{"generation_id": OLDEST, "graph_schema": "ark-rag-1"}],
        )
        with pytest.raises(SystemExit) as exc:
            ask._latest_identity_generation()
        assert "[2]" in str(exc.value)

    def test_raises_when_nothing_listed(self, ask, monkeypatch):
        _install(monkeypatch, ask, [])
        with pytest.raises(SystemExit) as exc:
            ask._latest_identity_generation()
        assert "[2]" in str(exc.value)


class TestBrokenDirectories:
    def test_directory_without_manifest_is_skipped(self, ask, monkeypatch):
        """缺 Manifest 的残留目录必须跳过，不能让整条默认路径崩掉。

        `list_generations()` 会把这类目录以 `status="(无 Manifest)"` 列出来，
        对它 `.load()` 会抛 `FileNotFoundError`。原实现不区分 status，直接 `.load()`。
        """
        listed = [
            {"generation_id": "g-20990101T000000Z-dead0000", "status": "(无 Manifest)", "created_at": ""},
            {"generation_id": NEWEST, "status": "READY", "created_at": "2026-10-31T03:52:31Z"},
        ]
        monkeypatch.setattr(ask.release, "list_generations", lambda: listed)

        def fake_generation(gid):
            if gid != NEWEST:
                raise FileNotFoundError(gid)
            return _FakeGeneration({"graph_schema": "ark-rag-3", "created_at": "2026-10-31T03:52:31Z"})

        monkeypatch.setattr(ask.release, "Generation", fake_generation)
        assert ask._latest_identity_generation() == NEWEST

    def test_corrupt_manifest_is_skipped(self, ask, monkeypatch):
        """Manifest 损坏的目录同样跳过——它的 status 已被 `list_generations()` 标出。"""
        listed = [
            {"generation_id": "g-20990101T000000Z-bad00000", "status": "(Manifest 损坏)", "created_at": ""},
            {"generation_id": NEWEST, "status": "READY", "created_at": "2026-10-31T03:52:31Z"},
        ]
        monkeypatch.setattr(ask.release, "list_generations", lambda: listed)

        def fake_generation(gid):
            if gid != NEWEST:
                raise AssertionError("损坏的目录不该被 load")
            return _FakeGeneration({"graph_schema": "ark-rag-3", "created_at": "2026-10-31T03:52:31Z"})

        monkeypatch.setattr(ask.release, "Generation", fake_generation)
        assert ask._latest_identity_generation() == NEWEST


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
