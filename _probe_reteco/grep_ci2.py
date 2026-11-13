import re, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
b=open(r"E:\图rag\_probe_reteco\recor.html",encoding="utf-8").read()
txt=re.sub(r"<[^>]+>"," ",b); txt=re.sub(r"\s+"," ",txt)
for kw in ["bootstrap","standard error","significan","paired comparison","t-test","confidence interval"]:
    print("####", kw)
    for i,m in enumerate(re.finditer(re.escape(kw),txt,re.I)):
        if i>=4: break
        print("  ...",txt[max(0,m.start()-240):m.start()+240],"...")
        print("  ---")
