"""基线校验对**缓存目录追加**的处理：追加不算篡改，改/删才算。

## 这条门原本是坏的

`scripts/snapshot_baseline.py --verify` 把 `data/arkrag/cache` 当成一个整体摘要
比对（`legacy.dir_digest()` 折叠 count + bytes + digest）。而
`legacy.guard_frozen_artifact()` 的 docstring 明确写了：**刻意不拦这个目录**，
因为「缓存目录里新记录以新键追加写入是正常运行，拦了会卡死抽取流水线」。

两条规矩互相打架，结果是：每跑一次合法抽取，`--verify` 就红一次。实测触发点
是 K2 —— 34 个新块抽取入库后，`count 13801->13835`，`digest a244f036->78353ab8`，
校验 FAIL。于是这条门只有在「永远不再抽取」时才是绿的——**那不是门，那是禁止干活**。

更糟的是它没法区分两件完全不同的事：
  · 有人往缓存里**追加**了 34 条新抽取记录（正常）；
  · 有人**改**了冻结的某条抽取记录（篡改，必须红）。

两种情况的整体摘要都会变，报出来的都是同一句话。

## 修法

给基线补一份逐文件清单 `data/releases/legacy-baseline/cache-inventory.json`
（相对名 / 字节 / SHA-256），校验时分两类判：

  · 清单里列过的文件**少了**或**字节变了** → 红；
  · 盘上多出清单之外的文件 → 只报数量，不判红。

清单本身不能用来掩盖篡改：`cache_inventory_digest(rows)` 的重算值必须逐字等于
Manifest 里记的那个 `digest`（`legacy.dir_digest()` 用的正是同一算法），所以改清单
一样会红。

## 补清单不等于重建基线

Manifest 的 `created_at`、全部 artifacts 记录、gates 冻结副本、metrics **一字未动**。
清单的正确性不是"猜的"：Manifest 里那个 `digest` 早就把清单摘要冻结住了，补出来的
清单必须能重算回那个值才算数（实测 13,801 条 → `a244f0362aa423dc`，逐字相同）。

## 这个文件守什么

守的是「门必须能失败」这条本身：`TestFrozenCacheStillFailsTheGate` 里每一条都构造了
一个**本该拦住它**的场景，并断言 `verify()` 真的返回 1。一个永远返回 0 的校验函数
比没有更糟——它会在台账里多出一行 PASS。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from arkrag import legacy

ROOT = Path(__file__).resolve().parents[1]


def _load_sb():
    spec = importlib.util.spec_from_file_location(
        "snapshot_baseline_under_test", ROOT / "scripts" / "snapshot_baseline.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["snapshot_baseline_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """把基线目录、缓存目录、ROOT 全指到 tmp，造一份小基线。"""
    root = tmp_path / "repo"
    cache = root / "data" / "arkrag" / "cache"
    baseline_dir = root / "data" / "releases" / "legacy-baseline"

    _write(cache / "ab" / "aaa.json", '{"content_key":"aaa"}')
    _write(cache / "cd" / "bbb.json", '{"content_key":"bbb"}')

    monkeypatch.setattr(legacy, "BASELINE_DIR", baseline_dir)
    monkeypatch.setattr(legacy, "BASELINE_MANIFEST", baseline_dir / "manifest.json")
    monkeypatch.setattr(legacy.config, "ROOT", root)
    monkeypatch.setattr(legacy.config, "CACHE_DIR", cache)
    monkeypatch.setattr(legacy.config, "DATA_DIR", root / "data" / "arkrag")
    monkeypatch.setattr(legacy.config, "INDEX_DIR", root / "data" / "arkrag" / "index")

    sb = _load_sb()

    rows = sb.collect_cache_inventory()
    _write(
        baseline_dir / sb.CACHE_INVENTORY_NAME,
        json.dumps(rows, ensure_ascii=False, indent=0) + "\n",
    )
    d = legacy.dir_digest(cache)
    manifest = {
        "baseline_id": "legacy-baseline",
        "artifacts": [
            {
                "path": "data/arkrag/cache",
                "kind": "extraction_cache_dir",
                "count": d["count"],
                "bytes": d["bytes"],
                "digest": d["digest"],
            }
        ],
    }
    _write(baseline_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    _write(baseline_dir / "gates.snapshot.json", "[]\n")
    return type("S", (), {"sb": sb, "root": root, "cache": cache, "baseline_dir": baseline_dir, "manifest": manifest})


class TestInventoryIsFrozenByTheManifestDigest:
    """清单的摘要必须与 Manifest 记录逐字相同——否则「补清单」就是「改基线」。"""

    def test_inventory_digest_equals_manifest_digest(self, sandbox):
        rows = sandbox.sb.collect_cache_inventory()
        rec = sandbox.manifest["artifacts"][0]
        assert sandbox.sb.cache_inventory_digest(rows) == rec["digest"]
        assert len(rows) == rec["count"]

    def test_digest_algorithm_matches_legacy_dir_digest(self, sandbox):
        """同一批文件，两种记账方式必须给出同一个摘要。

        否则迁移前记的 `digest` 与迁移后重算的对不上，整份基线立刻作废。
        """
        rows = sandbox.sb.collect_cache_inventory()
        assert sandbox.sb.cache_inventory_digest(rows) == legacy.dir_digest(sandbox.cache)["digest"]

    def test_inventory_lists_relative_posix_paths_only(self, sandbox):
        rows = sandbox.sb.collect_cache_inventory()
        for r in rows:
            assert not r["path"].startswith("/")
            assert "\\" not in r["path"]
            assert set(r) == {"path", "bytes", "sha256"}


class TestAppendDoesNotFailTheGate:
    """合法追加不许判红——这是这条门原本最大的错。"""

    def test_clean_baseline_passes(self, sandbox):
        assert sandbox.sb.verify() == 0

    def test_appended_cache_record_still_passes(self, sandbox, capsys):
        _write(sandbox.cache / "ef" / "ccc.json", '{"content_key":"ccc"}')
        assert sandbox.sb.verify() == 0
        err = capsys.readouterr().err
        assert "新增 1 条" in err, err

    def test_many_appends_only_report_a_count(self, sandbox, capsys):
        for i in range(5):
            _write(sandbox.cache / "xy" / f"n{i}.json", json.dumps({"content_key": f"n{i}"}))
        assert sandbox.sb.verify() == 0
        assert "新增 5 条" in capsys.readouterr().err


class TestFrozenCacheStillFailsTheGate:
    """真正的篡改一条都不能漏。"""

    def test_modified_frozen_record_fails(self, sandbox):
        _write(sandbox.cache / "ab" / "aaa.json", '{"content_key":"TAMPERED"}')
        assert sandbox.sb.verify() == 1

    def test_same_size_byte_flip_fails(self, sandbox):
        """只改一个字节、长度不变——整体摘要那种比法也会红，但清单必须也红。"""
        p = sandbox.cache / "ab" / "aaa.json"
        raw = bytearray(p.read_bytes())
        raw[-2] = ord("Z") if raw[-2] != ord("Z") else ord("Y")
        p.write_bytes(bytes(raw))
        assert p.stat().st_size == len(raw)
        assert sandbox.sb.verify() == 1

    def test_deleted_frozen_record_fails(self, sandbox):
        (sandbox.cache / "cd" / "bbb.json").unlink()
        assert sandbox.sb.verify() == 1

    def test_tampered_inventory_itself_fails(self, sandbox):
        """改清单里那条被改文件的 sha256，想掩盖篡改——摘要先红。"""
        p = sandbox.cache / "ab" / "aaa.json"
        _write(p, '{"content_key":"TAMPERED"}')
        inv = sandbox.baseline_dir / sandbox.sb.CACHE_INVENTORY_NAME
        rows = json.loads(inv.read_text("utf-8"))
        for r in rows:
            if r["path"].endswith("aaa.json"):
                r["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
                r["bytes"] = p.stat().st_size
        _write(inv, json.dumps(rows, ensure_ascii=False, indent=0) + "\n")
        assert sandbox.sb.verify() == 1

    def test_dropping_a_row_from_inventory_fails(self, sandbox):
        inv = sandbox.baseline_dir / sandbox.sb.CACHE_INVENTORY_NAME
        rows = json.loads(inv.read_text("utf-8"))
        _write(inv, json.dumps(rows[:-1], ensure_ascii=False, indent=0) + "\n")
        assert sandbox.sb.verify() == 1


class TestMissingInventoryDegradesLoudly:
    """没有清单时退回旧比法，但必须说清「这条门只是变弱，不是通过」。"""

    def test_missing_inventory_with_append_fails_and_says_why(self, sandbox, capsys):
        (sandbox.baseline_dir / sandbox.sb.CACHE_INVENTORY_NAME).unlink()
        _write(sandbox.cache / "ef" / "ccc.json", '{"content_key":"ccc"}')
        assert sandbox.sb.verify() == 1
        err = capsys.readouterr().err
        assert "无法区分追加与篡改" in err, err

    def test_missing_inventory_without_change_still_passes(self, sandbox):
        (sandbox.baseline_dir / sandbox.sb.CACHE_INVENTORY_NAME).unlink()
        assert sandbox.sb.verify() == 0


class TestRealBaselineIsConsistent:
    """真实仓库：清单必须存在，且它的摘要必须等于真实 Manifest 记的值。"""

    def test_real_inventory_present_and_digest_matches(self):
        manifest_path = ROOT / "data" / "releases" / "legacy-baseline" / "manifest.json"
        inv_path = ROOT / "data" / "releases" / "legacy-baseline" / "cache-inventory.json"
        if not manifest_path.exists() or not inv_path.exists():
            pytest.skip("真实基线尚未补清单")
        manifest = json.loads(manifest_path.read_text("utf-8"))
        rec = next(a for a in manifest["artifacts"] if a["kind"] == "extraction_cache_dir")
        rows = json.loads(inv_path.read_text("utf-8"))
        body = "\n".join(
            f"{r['path']}\x00{r['bytes']}\x00{r['sha256']}"
            for r in sorted(rows, key=lambda r: r["path"])
        )
        got = hashlib.sha256(body.encode("utf-8")).hexdigest()
        assert got == rec["digest"], "cache-inventory.json 与 Manifest 记录的 digest 不符"
        assert len(rows) == rec["count"]
