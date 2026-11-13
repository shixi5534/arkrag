import urllib.request, ssl, re
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
req = urllib.request.Request("https://arxiv.org/html/2601.05461v1", headers={"User-Agent":"Mozilla/5.0"})
try:
    b = urllib.request.urlopen(req, timeout=45, context=ctx).read().decode("utf-8","replace")
    open(r"E:\图rag\_probe_reteco\recor.html","w",encoding="utf-8").write(b)
    print("len", len(b))
    txt = re.sub(r"<[^>]+>"," ",b)
    txt = re.sub(r"\s+"," ",txt)
    for kw in ["confidence interval","95% CI","paired t","p < 0.001","p<0.001","23,768","23768","±","effect size","Cohen"]:
        for m in re.finditer(re.escape(kw), txt, re.I):
            print(f"[{kw}] ...{txt[max(0,m.start()-200):m.start()+200]}...")
            print("-"*40)
            break
except Exception as e:
    print("ERR", type(e).__name__, str(e)[:200])
