import re, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
b=open(r"E:\图rag\_probe_reteco\recor.html",encoding="utf-8").read()
txt=re.sub(r"<[^>]+>","|",b); txt=re.sub(r"\|+","|",txt); txt=re.sub(r"[ \t]+"," ",txt)
i=txt.find("Retriever | Base")
if i<0: i=txt.find("DIVER")
print(txt[i-200:i+2600])
