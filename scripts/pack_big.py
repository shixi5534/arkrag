# -*- coding: utf-8 -*-
"""把尚未抽取的 pending part 合并成"大包"任务文件，供子代理一次处理更多块。

用法：
  py -3 scripts/pack_big.py --per-pack 120 --label B01
产出：
  data/arkrag/agent_inbox/B01.part00.jsonl ... （每行仍是 chunk_id/content_key/user_prompt）
  data/arkrag/_packs.txt  （写入每包的 chunk 数与来源 part）
原则：同一个 pack 尽量只放同一片（章节相邻），并按 part 顺序拼接，保证上下文连贯。
"""
import argparse
import json
import os
import glob

INBOX = "data/arkrag/agent_inbox"
OUTBOX = "data/arkrag/agent_outbox"
DONE = os.path.join(OUTBOX, "done")


def pending_parts():
    have = set(os.listdir(OUTBOX)) | set(os.listdir(DONE))
    pend = []
    for q in sorted(glob.glob(os.path.join(INBOX, "_queue_g*.txt"))):
        for line in open(q, encoding="utf-8"):
            job = line.strip()
            if not job:
                continue
            jf = os.path.join(INBOX, "_job_%s.txt" % job)
            if not os.path.exists(jf):
                continue
            for p in open(jf, encoding="utf-8"):
                p = p.strip()
                if p and p not in have and p not in pend:
                    pend.append(p)
    return pend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pack", type=int, default=120)
    ap.add_argument("--label", default="B01")
    ap.add_argument("--parts-file", default="data/arkrag/_pending_parts.txt")
    a = ap.parse_args()

    parts = pending_parts()
    open(a.parts_file, "w", encoding="utf-8").write("\n".join(parts))

    # 按片号分组，保持片内顺序
    def shard_of(p):
        return p.split(".")[0]

    buckets = {}
    order = []
    for p in parts:
        s = shard_of(p)
        if s not in buckets:
            buckets[s] = []
            order.append(s)
        buckets[s].append(p)

    packs = []  # (label, [part...])
    idx = 0
    for s in order:
        cur, n = [], 0
        for p in buckets[s]:
            cnt = sum(1 for l in open(os.path.join(INBOX, p), encoding="utf-8") if l.strip())
            if cur and n + cnt > a.per_pack:
                packs.append(("%s.part%02d.jsonl" % (a.label, idx), cur))
                idx += 1
                cur, n = [], 0
            cur.append(p)
            n += cnt
        if cur:
            packs.append(("%s.part%02d.jsonl" % (a.label, idx), cur))
            idx += 1

    lines_out = []
    for name, plist in packs:
        rows = []
        for p in plist:
            for l in open(os.path.join(INBOX, p), encoding="utf-8"):
                if l.strip():
                    rows.append(json.loads(l))
        with open(os.path.join(INBOX, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        lines_out.append("%s\t%d blocks\t%s" % (name, len(rows), ",".join(plist)))

    open("data/arkrag/_packs.txt", "w", encoding="utf-8").write("\n".join(lines_out))
    print("pending parts:", len(parts))
    print("packs:", len(packs))
    for l in lines_out:
        print(l)


if __name__ == "__main__":
    main()
