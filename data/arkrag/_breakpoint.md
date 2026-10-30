# L3 抽取断点（2026-10-29 08:35 人工停机）

## 进度
- 已入库 **10,652 / 13,786** 块（T1 口径），剩余 **3,134** 块。
- 大包方案 `B01`：**47 / 101** 包入库（含 part48 收尾）。

## 停机时各包状态
- 已完成归档：46 包（`agent_outbox/done/B01.*`）
- 已写全待回收：`B01.part16`(77/77)、`B01.part48`(58/58，已入库)
- **半成品（需重跑）**：`B01.part39`(69/80)、`B01.part41`(60/80)、`B01.part50`(5/67)、`B01.part52`(10/80)
- 未开工：**49 包**，清单见 `data/arkrag/_pending_packs.txt`

## 后台进程
- 已 `job_kill`：`workflow-c28c6592a806425c`（抽取池）、`pwsh-0f35b2b09ff74b38`（回收循环）
- 8 个 teammate（ext-s11a、ext-g1..g7）均为 inactive，未删除（上限 8 且无释放工具）

## 恢复步骤
1. 清半成品包缓存后重下发：
   `py -3 scripts/make_repair.py --list <坏块清单> --label R02`
   或直接删 `agent_outbox` 里 4 个半成品文件 + 用 `pack_big.py` 按 `_pending_packs.txt` 重新出包
2. 派 6 个子代理跑 `_pending_packs.txt` 里的包（每包 60–80 块，按片分组保序）
3. 后台起回收循环 `py -3 scripts/ingest_all.py`
4. 全部收完后：修 `B01.part23` 的 6 条 confidence 0.4 → 全库 `qa_evidence.py` → `run_l5.py` → 更新 `docs/抽取台账.md`、`docs/L3质检报告.md`

## 已知质量问题（待收尾统一处理）
- `B01.part23.jsonl`：6 条 confidence=0.4 低于 0.5 下限
- 首轮全库 91 条（0.34%）evidences/词表/端点问题已由 R01 修复
