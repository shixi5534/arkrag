import json, os
root=r"E:\图rag\_probe_reteco\track2_recor\politics"
target="politics_ex_88512_doc_3"
found=False
with open(os.path.join(root,"documents.jsonl"),encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        d=json.loads(ln)
        if d["doc_id"]==target:
            found=True
            print("FOUND in corpus:", d["doc_id"], "len(content)=", len(d["content"]))
            print("  first 300:", d["content"][:300].replace("\n"," "))
            break
if not found: print("NOT in corpus:", target)
# does ex_88512 exist; show its turns' gold
data=json.load(open(os.path.join(root,"benchmark_dev.json"),encoding="utf-8"))
for c in data:
    if c["id"]=="ex_88512":
        print("\nconversation", c["id"], "turns", len(c["turns"]))
        for t in c["turns"]:
            print(" turn", t["turn_id"], "gold:", t.get("gold_doc_ids"))
