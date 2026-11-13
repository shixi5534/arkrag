import urllib.request, ssl, re, io, sys
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
req=urllib.request.Request("https://arxiv.org/html/2411.00640v1", headers={"User-Agent":"Mozilla/5.0"})
try:
    b=urllib.request.urlopen(req,timeout=60,context=ctx).read().decode("utf-8","replace")
    open(r"E:\图rag\_probe_reteco\miller.html","w",encoding="utf-8").write(b)
    txt=re.sub(r"<[^>]+>"," ",b); txt=re.sub(r"\s+"," ",txt)
    print("len",len(b))
    for kw in ["3.05","1.88","DROP","MGSM","clustered standard error","cluster"]:
        for i,m in enumerate(re.finditer(re.escape(kw),txt)):
            if i>=3: break
            print(f"\n[{kw}] ...{txt[max(0,m.start()-320):m.start()+320]}...")
except Exception as e:
    print("ERR",type(e).__name__,str(e)[:150])
