"""L0 语料准入：来源合同验证 + 体检 + 门判定 + 新鲜度自检。

与旧版的区别（计划 D4）：ZIP 不在场时不再"打印提示然后继续"。
来源验不过就是 `source_unverified`，写进 `data/arkrag/source_verification.json`，
下游 `release.verify_generation` 会拒绝把这样的世代发布成 Release。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import config, gates, l0_audit, source  # noqa: E402


def main() -> int:
    recorded = source.load()
    expected_tree = recorded.get("tree_digest", "") if recorded else ""
    report = source.verify(expected_tree_digest=expected_tree)
    path = source.save(report)

    print("=== 来源验证（D4）===")
    print(f"  状态      : {report['status']}")
    print(f"  合同      : {report['contract']['repo']}@{report['contract']['release_tag']}")
    print(f"  ZIP       : {report['zip_path']} 在场={report['zip_present']}")
    print(f"  解压树    : {report['tree_files']} 文件 {report['tree_bytes']:,} 字节 "
          f"摘要 {report['tree_digest'][:16]}")
    print(f"  已通过检查: {report['checks'] or '(无)'}")
    for problem in report["problems"]:
        print(f"  ! {problem}", file=sys.stderr)
    print(f"  证据写入  : {path}")

    # 判据用白名单。写 `== source.UNVERIFIED` 是**反过来的**：它只拦一个特定取值，
    # 于是任何"既不是已验证、也不是这个拼写"的状态（上游将来新增的状态、人为写错的值）
    # 都会被当成通过。这与 release.py 发布门曾经犯的错是同一个 —— 门只拒绝一个
    # 特定值，等于没有门。
    verified = report["status"] in (source.VERIFIED_ZIP, source.VERIFIED_TREE)
    if not verified:
        # 来源身份未知绝不能冒充 Release（计划 D4）。这里返回非零，让调用者知道
        # 这一轮产出不是可发布产物；但体检与门仍然照跑，好让人看到语料本身的状态。
        print(
            f"\n来源未通过验证（status={report['status']!r}）：本轮产物只能作为分析用途，"
            "不得发布为 Release（计划 D4）。",
            file=sys.stderr,
        )

    code = l0_audit.main()
    if code:
        return code
    manifest = json.loads(config.l0_manifest_path().read_text("utf-8"))
    manifest["source_contract"] = report["contract"]
    manifest["source_status"] = report["status"]
    manifest["source_verification"] = {
        "status": report["status"],
        "contract_fp": report["contract_fp"],
        "tree_digest": report["tree_digest"],
        "tree_files": report["tree_files"],
        "zip_sha256": report.get("zip_sha256", ""),
        "checks": report["checks"],
        "problems": report["problems"],
    }
    run = gates.l0_gates(manifest)
    gates.write_report(run, config.gates_path(), "L0")
    print("\n=== L0 门判定 ===")
    for g in run.gates:
        print(f"  {g.verdict:16s} {g.name:28s} 门线[{g.threshold}] 实测[{g.measured[:60]}] {g.note[:60]}")
    fails = [g for g in run.gates if g.verdict == "FAIL"]
    print(f"FAIL {len(fails)} / {len(run.gates)}")

    if fails:
        return 1
    # 门全过但来源未验证：仍然不能当 Release，退出码非零（原则 5：门必须能失败）
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
