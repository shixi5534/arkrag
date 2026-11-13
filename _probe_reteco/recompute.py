import io,sys,math
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
print("RECOR Table 17 (paired t-tests):")
print("  H+R vs Baseline  MeanD=+0.243  SE=0.0022  t=119.7  n=23,768")
print("  H+R vs History   MeanD=+0.039  SE=0.0013  t=30.2   n=23,768")
print()
print("n 的来历: 2,971 turns x 8 retrievers =", 2971*8)
print("  -> n=23,768 恰好是 8 个检索器在【同一批 turn】上的观测被当成独立")
print()
SD=0.3758  # my measured per-turn SD of paired difference on dev
print("我的 dev 实测: per-turn paired SD = %.4f"%SD)
print()
print("=== 反推 RECOR 的 SE 隐含了什么 ===")
for label,se in [("H+R vs Baseline",0.0022),("H+R vs History",0.0013)]:
    n_implied=(SD/se)**2
    print(f"  {label:18s} SE={se}: 隐含 n_eff = (SD/SE)^2 = {n_implied:,.0f}  (报告 n=23,768)")
print()
print("=== 按对话聚类（deff=4.48 实测）应有的 SE ===")
for n_turns,label in [(2971,"全量 2,971 turn")]:
    n_eff=n_turns/4.48
    for D in (0.243,0.039):
        se=SD/math.sqrt(n_eff)
        print(f"  {label}: n_eff={n_eff:.0f}  D={D:.3f} -> SE={se:.4f}  t={D/se:.1f}  CI95=±{1.96*se:.4f}")
print()
print("=== 结论 ===")
print("  报告 CI ±.004  ->  聚类后 ±.029   宽 %.1f 倍"%((0.029)/0.004))
print("  t 从 119.7 降到 ~16.6，仍 p<0.001 —— 定性结论存活，精度声明不存活")
