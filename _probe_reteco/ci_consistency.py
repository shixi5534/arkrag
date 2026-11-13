import io,sys,math
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
print("=== RECOR 报告的三个 CI 之间的一致性检查 ===")
print()
print("(a) Average 行: +-0.004  (五列全都是 .004，同值)")
print("(b) DIVER:  .584  CI [.520, .649]  -> 半宽 %.4f, 全宽 %.3f"%((0.649-0.520)/2, 0.649-0.520))
print("(c) ReasonIR: .552 CI [.501, .603] -> 半宽 %.4f, 全宽 %.3f"%((0.603-0.501)/2, 0.603-0.501))
print()
print("--- 若 (a) 是 turn 观测口径 ---")
for n,label in [(23768,"n=23,768 (=2,971 turn x 8 retrievers)"),(2971,"n=2,971 turn")]:
    sd = 0.004/1.96*math.sqrt(n)
    print(f"  {label:42s} 隐含 per-turn SD = {sd:.4f}")
print("  我 dev 实测 per-turn nDCG SD: Base 0.30~0.39, +Hist 0.24~0.41")
print("  -> n=23,768 隐含 SD=0.31，与实测一致；n=2,971 隐含 SD=0.11，明显过小")
print()
print("--- 若 (b)(c) 是 domain 口径 ---")
t10=2.228
for name,val,lo,hi in [("DIVER",0.584,0.520,0.649),("ReasonIR",0.552,0.501,0.603)]:
    half=(hi-lo)/2
    se=half/t10
    sd_dom=se*math.sqrt(11)
    print(f"  {name:10s} 半宽={half:.4f}  t(10)=2.228 -> SE={se:.4f} -> 11 域均值的 SD={sd_dom:.4f}")
print("  我 dev 实测 11 域均值的 SD = 0.1485（BM25 +Hist）——同一量级，合理")
print()
print("=== 结论 ===")
print("  同一张表里用了两套抽样单位：")
print("   Average 行  -> per-observation（23,768 个观测，把 8 个检索器当独立）")
print("   DIVER/ReasonIR -> 按域（11 域，t(10)）")
print("  两者宽度相差 %.1f 倍"%((0.649-0.520)/(2*0.004)))
