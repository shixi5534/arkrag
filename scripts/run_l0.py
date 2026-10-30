"""L0 语料准入：体检 + 门判定 + 新鲜度自检。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import config, fingerprint, gates, l0_audit  # noqa: E402

SOURCE_CONTRACT = {
    "release_tag": "akdp-26-10-14-13-45-10-99469376-v1",
    "asset": "zh_CN.zip",
    "zip_bytes": 35904420,
    "zip_sha256": "46b106207bac11af75fe70169e49b00f25e360da0f32d41824e29b52a01e0aeb",
    "compatibility_contract": "prts-story-v1",
    "repo": "3aKHP/ArknightsStoryJson",
}


def main() -> int:
    zip_path = Path("zh_CN.zip")
    if zip_path.exists():
        actual = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        if actual != SOURCE_CONTRACT["zip_sha256"]:
            print(f"FAIL: zh_CN.zip 字节与清单不符 {actual}", file=sys.stderr)
            return 1
        print(f"语料字节校验 PASS  sha256={actual[:16]}")
    else:
        print("提示: zh_CN.zip 不在场，跳过字节校验（corpus/ 已在盘上）")

    code = l0_audit.main()
    if code:
        return code
    manifest = json.loads(config.L0_MANIFEST_PATH.read_text("utf-8"))
    manifest["source_contract"] = SOURCE_CONTRACT
    run = gates.l0_gates(manifest)
    gates.write_report(run, config.DATA_DIR / "gates.json", "L0")
    print("\n=== L0 门判定 ===")
    for g in run.gates:
        print(f"  {g.verdict:16s} {g.name:28s} 门线[{g.threshold}] 实测[{g.measured[:60]}] {g.note[:60]}")
    fails = [g for g in run.gates if g.verdict == "FAIL"]
    print(f"FAIL {len(fails)} / {len(run.gates)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
