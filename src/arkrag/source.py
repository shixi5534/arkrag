"""来源合同与语料身份（计划 D4）。

## 为什么需要一个独立的来源层

旧 `run_l0.py` 的逻辑是：

    if zh_CN.zip 存在:  比对 sha256
    else:              打印一句"跳过校验"，然后继续

也就是说，**ZIP 不在场时语料来源处于"没人检查过"的状态，而这个状态和"已验证"
在下游完全无法区分**。产物照样打上 schema_version、照样发布、照样被当成 Release
引用。计划 D4 要求把这两种情况分开：

    verified_zip      ZIP 在场，字节与合同一致，**且**解压树与 ZIP 内容逐文件一致
    verified_tree     没有 ZIP，但解压树与已记录的确定性文件清单一致
    source_unverified 两者都无法核对——身份未知

`source_unverified` 不是错误，是**诚实的未知**。把它静默当成已验证才是错误。
下游（`release.verify_generation`）会拒绝把 `source_unverified` 的世代发布成 Release。

## 确定性文件清单

「确定性」是关键：不能拿"文件数"当清单。文件数一样而内容全换，计数比对毫无察觉。
这里用 (相对路径, 字节数, sha256) 三元组按路径排序后累进，得到一个树摘要；
ZIP 侧用同样的算法算同一批成员，两个摘要可直接比较——这就同时满足了
"ZIP 存在则验证 ZIP 与释放内容"和"缺失则验证解压树确定性文件清单"。
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

from . import atomic, config

# 上游发布合同。改这里必须同时改 docs/决策台账.md 并说明理由——
# 它是"这份语料是哪一份"的唯一声明。
SOURCE_CONTRACT = {
    "repo": "3aKHP/ArknightsStoryJson",
    "release_tag": "akdp-26-10-14-13-45-10-99469376-v1",
    "asset": "zh_CN.zip",
    "zip_bytes": 35904420,
    "zip_sha256": "46b106207bac11af75fe70169e49b00f25e360da0f32d41824e29b52a01e0aeb",
    "compatibility_contract": "prts-story-v1",
    # art 资源不进剧情流水线，只用于核对上游清单规模没有意外变化
    "factory_manifest_events": 485,
    "factory_manifest_chapters": 2032,
    "factory_manifest_json_files": 2360,
}

ZIP_STORY_PREFIX = "zh_CN/gamedata/story/"

VERIFIED_ZIP = "verified_zip"
VERIFIED_TREE = "verified_tree"
UNVERIFIED = "source_unverified"


def verification_path() -> Path:
    """验证记录的位置：**产出验证的地方**，即 `DATA_DIR / source_verification.json`。

    这里是函数不是模块级常量，理由与 `config.artifact_root()` 完全相同
    （见 config.py 那段注释）：测试要把整条链路指到临时目录，只能靠
    `monkeypatch.setattr(config, "DATA_DIR", …)`；一旦在 import 时把路径
    烧进模块常量，重定向对它无效 —— 读到的永远是**真实仓库**里那份记录，
    于是「来源状态」在测试里成了真值的回声，测不出任何东西。
    """
    return Path(config.DATA_DIR) / "source_verification.json"

# 路径规范化方式。本语料上游 ZIP 用 `obt/`，解压树用 `Obt/`（823 个文件，内容逐字节相同，
# 见决策台账 D-03）。身份里显式记下这个口径，而不是悄悄折叠或不折叠。
PATH_NORM = "casefold"


def _relkey(rel: str) -> str:
    return rel.casefold() if PATH_NORM == "casefold" else rel


def _sha256_file(path: Path, *, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def tree_entries(root: Path) -> list[tuple[str, int, str]]:
    """解压树的确定性文件清单：(相对路径, 字节数, sha256)，按路径排序。"""
    root = Path(root)
    if not root.is_dir():
        return []
    out: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        out.append((rel, path.stat().st_size, _sha256_file(path)))
    return out


def zip_entries(zip_path: Path, prefix: str = ZIP_STORY_PREFIX) -> list[tuple[str, int, str]]:
    """ZIP 内剧情成员的同一份清单，相对路径口径与 tree_entries 一致。"""
    out: list[tuple[str, int, str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir() or not name.startswith(prefix):
                continue
            rel = name[len(prefix):]
            h = hashlib.sha256()
            with zf.open(info) as fh:
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
            out.append((rel, info.file_size, h.hexdigest()))
    out.sort()
    return out


def digest_entries(entries: list[tuple[str, int, str]], *, path_norm: str = "") -> str:
    """把清单摘要成一个值。路径、字节数、内容三者都进，缺一项就能被"同数换内容"绕过。

    `path_norm="casefold"` 时路径先折叠大小写再摘要。这不是为了绕过差异，而是因为
    本语料的上游 ZIP 与解压树在 823 个路径上大小写不同（`obt/` vs `Obt/`，
    实测内容逐字节相同，见决策台账 D-03）。把规范化方式**显式写进身份**，
    比"两边各自摘要后报不一致"或"悄悄折叠不说"都更可查：前者会永久红，
    后者会让身份失去可复现性。
    """
    h = hashlib.sha256()
    h.update(f"path_norm={path_norm}\x1e".encode("utf-8"))
    # 规范化后必须重排：`Obt/` 排在 `activities/` 之前（大写 O < 小写 a），
    # 而 `obt/` 排在它之后。只折叠不重排，同一批文件会因目录名大小写差异摘要不同。
    rows = sorted(
        ((rel.casefold() if path_norm == "casefold" else rel), size, sha)
        for rel, size, sha in entries
    )
    for key, size, sha in rows:
        h.update(f"{key}\x00{size}\x00{sha}\n".encode("utf-8"))
    return h.hexdigest()


def contract_fingerprint(contract: dict | None = None) -> str:
    contract = SOURCE_CONTRACT if contract is None else contract
    parts = [
        str(contract.get("repo", "")),
        str(contract.get("release_tag", "")),
        str(contract.get("asset", "")),
        str(contract.get("zip_bytes", "")),
        str(contract.get("zip_sha256", "")),
        str(contract.get("compatibility_contract", "")),
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def zip_path_from_env() -> Path:
    """ZIP 位置。默认仓库根下 zh_CN.zip；旧代码用相对 cwd 的 Path("zh_CN.zip")，
    A3 已把这处 cwd 依赖视为缺陷——从别的目录跑就会"ZIP 不在场"，进而跳过校验。"""
    return Path(os.environ.get("ARKRAG_SOURCE_ZIP", str(config.ROOT / "zh_CN.zip")))


def verify(
    *,
    story_root: Path | None = None,
    zip_path: Path | None = None,
    contract: dict | None = None,
    expected_tree_digest: str = "",
) -> dict:
    """核对来源，返回可落盘的证据字典。

    顺序很重要：先看 ZIP（字节可核），再看解压树（清单可核），
    都在不了就明确报 unknown，**绝不静默当成通过**。
    """
    contract = SOURCE_CONTRACT if contract is None else contract
    # 内部一律用**真实绝对路径**读盘（下面的 exists()/stat()/tree_entries() 都吃它）。
    # 只有写进报告的字符串走 rel_to_root 变相对——报告要进 git，绝对机台路径
    # 换台机器就是错的；相对路径在任何机器上都指向同一个仓库内对象。
    story_root = Path(story_root or config.STORY_ROOT)
    zip_path = Path(zip_path) if zip_path else zip_path_from_env()

    report: dict = {
        "status": UNVERIFIED,
        "contract": contract,
        "contract_fp": contract_fingerprint(contract),
        "story_root": config.rel_to_root(story_root),
        "zip_path": config.rel_to_root(zip_path),
        "zip_present": zip_path.exists(),
        "checks": [],
        "problems": [],
    }

    tree = tree_entries(story_root)
    tree_digest = digest_entries(tree, path_norm=PATH_NORM)
    report["tree_files"] = len(tree)
    report["tree_bytes"] = sum(size for _rel, size, _sha in tree)
    report["tree_digest"] = tree_digest
    report["path_norm"] = PATH_NORM

    if zip_path.exists():
        size = zip_path.stat().st_size
        sha = _sha256_file(zip_path)
        report["zip_bytes"] = size
        report["zip_sha256"] = sha
        if size != contract["zip_bytes"]:
            report["problems"].append(f"ZIP 字节数不符：{size} != {contract['zip_bytes']}")
        if sha != contract["zip_sha256"]:
            report["problems"].append(f"ZIP sha256 不符：{sha}")
        if not report["problems"]:
            report["checks"].append("zip_bytes_and_sha256")

        # ZIP 与释放内容必须一致：这是"ZIP 在场时"真正该查的东西。
        # 只查 ZIP 本身说明不了解压树是不是从它来的（曾经手动改过盘上的文件就无从发现）。
        try:
            zipped = zip_entries(zip_path)
        except (zipfile.BadZipFile, OSError) as exc:
            report["problems"].append(f"ZIP 不可读：{exc}")
            zipped = []
        if zipped:
            report["zip_story_files"] = len(zipped)
            report["zip_tree_digest"] = digest_entries(zipped, path_norm=PATH_NORM)
            tree_map = {_relkey(rel): (size, sha) for rel, size, sha in tree}
            zip_map = {_relkey(rel): (size, sha) for rel, size, sha in zipped}
            only_zip = sorted(set(zip_map) - set(tree_map))
            only_tree = sorted(set(tree_map) - set(zip_map))
            differ = sorted(
                rel for rel in set(tree_map) & set(zip_map) if tree_map[rel] != zip_map[rel]
            )
            if only_zip or only_tree or differ:
                report["problems"].append(
                    f"解压树与 ZIP 不一致：仅 ZIP 有 {len(only_zip)}、仅盘上有 {len(only_tree)}、"
                    f"内容不同 {len(differ)}（示例 {(only_zip + only_tree + differ)[:3]}）"
                )
            else:
                report["checks"].append("zip_matches_tree")
        if not report["problems"]:
            report["status"] = VERIFIED_ZIP
    else:
        # 没有 ZIP：只能核对解压树。期望摘要来自上一次验证的记录（确定性清单），
        # 没有记录就没有可核对的基准——那仍然是 unknown，不是通过。
        if expected_tree_digest:
            if tree_digest == expected_tree_digest:
                report["checks"].append("tree_matches_recorded_manifest")
                report["status"] = VERIFIED_TREE
            else:
                report["problems"].append(
                    f"解压树摘要与记录不符：现在 {tree_digest[:16]} 记录 {expected_tree_digest[:16]}"
                )
        elif tree:
            report["problems"].append(
                "ZIP 不在场且没有已记录的解压树清单：无法验证来源，标 source_unverified"
            )

    if report["status"] == UNVERIFIED and not report["problems"]:
        report["problems"].append("无法验证来源：既无 ZIP 也无解压树")
    return report


def load(path: Path | None = None) -> dict:
    path = Path(path) if path else verification_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save(report: dict, path: Path | None = None) -> Path:
    path = Path(path) if path else verification_path()
    atomic.atomic_write_json(path, report)
    return path


def status_for_build(path: Path | None = None) -> tuple[str, dict]:
    """下游用：读出可用的来源状态。没有记录即 unknown，绝不默认成已验证。"""
    report = load(path)
    return (report.get("status") or UNVERIFIED), report


# ---------------------------------------------------------------- 状态强弱


#: 状态强度：数值越小越保守。**只有**下面三个值有定义，任何其它字符串
#: （拼错、上游旧版本写的、被人手改的）一律按 0 处理 —— 未知拼写绝不能
#: 变成"比 unverified 更强"，否则把 `verified_zip ` 写成带尾空格就能自称已验证。
_STRENGTH = {UNVERIFIED: 0, VERIFIED_TREE: 1, VERIFIED_ZIP: 2}


def strength(status: str) -> int:
    """状态的强度，未知取值按最保守（0）算。

    这里是**精确匹配**，不 strip、不 casefold：`"verified_zip "`（尾空格）、
    `"VERIFIED_ZIP"`（大写）、`"verified"`（缩写）全都算 0。
    理由是这个状态是被人写进 JSON 的字符串，而"看起来像验证过"和"验证过"
    的差别正是这套检查存在的意义；任何形式的宽容归一化都在把
    "笔误"翻译成"已验证"。要归一化的话，该在写入端做，并且归一化本身也得
    有记录——不是在读取端猜。
    """
    return _STRENGTH.get(status, 0) if isinstance(status, str) else 0


def weaker(a: str, b: str) -> str:
    """两个状态里更保守的那个；相等时返回 a。"""
    return a if strength(a) <= strength(b) else b


def resolve_for_build(
    corpus_meta: dict | None = None,
    path: Path | None = None,
) -> dict:
    """构建用：定下这一代该采用的 `source_status`，并留下对账记录。

    ## 为什么权威是验证记录，不是 `corpus_meta.json`

    `source_verification.json` 是**验证发生地**的产物（`run_l0.py` 跑完核对后写的），
    它是"这份语料是什么"这个判断的证据本身。`corpus_meta.json` 里的
    `source_status` 只是 `run_l1.py` 顺手抄进下游语料元数据的一份**副本**。

    副本不配当权威，有三个现实理由，第三个就是这次要修的 bug：

    1. 副本会过期：L0 重跑后验证记录更新了，旧副本还留着上一轮的结论；
    2. 副本会串味：产物从一个目录搬到另一个目录（staging / 冻结基线）时，
       副本跟着走而它描述的验证记录没跟着走；
    3. **副本会缺**：`data/arkrag/corpus_meta.json` 是 D4 之前的旧产物，
       全文件 153 字节，压根没有 `source_status` 字段。旧代码
       `corpus_meta.get("source_status", "source_unverified")` 于是把一份
       实际已验证（`verified_zip`，ZIP 字节与解压树两边对得上）的语料判成
       未知，构建被拒 —— 缺字段被当成了"没验证"，而真相只是"这个文件里没写"。

    所以：**权威取验证记录**，副本只作兜底与交叉核对。

    ## 不一致怎么办

    `corpus_meta` 里有副本且与权威不符时，不静默取一边：`conflicts` 里如实登录
    两边的值，返回值取**更保守**的那个（见 `weaker`），调用方必须把它显式报出来。
    取保守是因为"两边不一致"这件事本身就说明有一个产物在说谎，而在能证明之前，
    按较弱的那个申报不会高估来源可信度。
    """
    authority_status, authority_report = status_for_build(path)
    authority_path = Path(path) if path else verification_path()

    meta = corpus_meta or {}
    copy_present = "source_status" in meta
    # 同样不 strip：副本写成 `"verified_zip "` 时它就是一个**不认识的**状态，
    # 该被 `weaker` 拖成未验证并报冲突，而不是被归一化悄悄抹平。
    raw_copy = meta.get("source_status")
    copy_status = raw_copy if isinstance(raw_copy, str) else ("" if raw_copy is None else str(raw_copy))

    conflicts: list[str] = []
    if copy_present and copy_status and copy_status != authority_status:
        conflicts.append(
            f"来源状态两份记录不一致：验证记录 {authority_status}"
            f"（{authority_path}）vs corpus_meta 副本 {copy_status}；"
            f"取更保守的 {weaker(authority_status, copy_status)}"
        )
    if copy_present and not copy_status:
        conflicts.append(
            f"corpus_meta 有 source_status 字段但值为空；以验证记录 {authority_status} 为准"
        )

    chosen = (
        weaker(authority_status, copy_status) if (copy_present and copy_status) else authority_status
    )
    return {
        "status": chosen,
        # 权威侧原值，保留下来方便事后区分"记录说未验证"与"副本拖低了它"
        "authority_status": authority_status,
        "authority_path": str(authority_path),
        "authority_present": authority_path.exists(),
        "copy_status": copy_status,
        "copy_present": copy_present,
        "conflicts": conflicts,
        "verified": chosen in (VERIFIED_ZIP, VERIFIED_TREE),
        "report": authority_report,
    }
