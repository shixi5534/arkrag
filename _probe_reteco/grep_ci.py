import re
b=open(r"E:\图rag\_probe_reteco\recor.html",encoding="utf-8").read()
txt=re.sub(r"<[^>]+>"," ",b); txt=re.sub(r"\s+"," ",txt)
for kw in ["bootstrap","standard error","significance","paired comparison","t-test","Welch","confidence"]:
    print("#### ",kw)
    for m in re.finditer(re.escape(kw),txt,re.I):
        s=txt[max(0,m.start()-260):m.start()+260]
        print("  ...",s,"...")
        print("  ---")
