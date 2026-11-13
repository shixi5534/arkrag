import io,sys,math
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
# RECOR Table 3 H+R column, sorted
vals={"DIVER":.584,"ReasonIR":.552,"BM25":.489,"SFR":.464,"Qwen":.461,"BGE":.445,"E5":.429,"Contriever":.409}
s=sorted(vals.items(), key=lambda x:-x[1])
print("RECOR Table 3, History+Reasoning column, sorted:")
prev=None
gaps=[]
for name,v in s:
    if prev is not None:
        g=v-prev; gaps.append((prev_name,name,g))
        print(f"  {prev_name:11s} {prev:.3f} -> {name:11s} {v:.3f}   gap={g:.3f}")
    else:
        print(f"  {name:11s} {v:.3f}")
    prev=v; prev_name=name
print()
print("adjacent gaps: n=%d  mean=%.3f  median=%.3f  min=%.3f  max=%.3f"%(
    len(gaps), sum(g for *_,g in gaps)/len(gaps),
    sorted(g for *_,g in gaps)[len(gaps)//2],
    min(g for *_,g in gaps), max(g for *_,g in gaps)))
print()
MDE_clu=0.1576; MDE_iid=0.0744
below=sum(1 for *_,g in gaps if g < MDE_clu)
below_iid=sum(1 for *_,g in gaps if g < MDE_iid)
print("vs MDE(clustered, 200 turns) = %.3f : %d/%d gaps below MDE"%(MDE_clu,below,len(gaps)))
print("vs MDE(iid,       200 turns) = %.3f : %d/%d gaps below MDE"%(MDE_iid,below_iid,len(gaps)))
print()
print("=> 用 200-turn 隐藏集给这 8 个系统排名，相邻名次的差距全部小于可检测阈值。")
print("   榜单能把它们排成 1..8，但那些名次差不是统计证据。")
