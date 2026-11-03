#!/usr/bin/env python3
"""s109_observable_shortcut.py — 用**可观测**的图规则验证捷径污染。

不看 comp_depth_by_index（那是元数据，会构成循环论证），
而是只从题面里的 `facts` 构图，对每个候选「走到尽头」数下游链长，
看「选最长」或「选最短」能否 100% 命中答案。

这条规则是模型在只看到 facts 时也能学到的，因此是真实的捷径。
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def downstream_depth(facts, start, limit=64):
    """从 start 出发沿有向边能走的最长步数（DAG 假设）。"""
    adj = {}
    for f in facts:
        adj.setdefault(f["subj"], []).append(f["obj"])

    best = 0
    # 迭代 DFS，防环
    stack = [(start, 0, frozenset())]
    while stack:
        node, d, seen = stack.pop()
        if d > best:
            best = d
        if d > limit:
            continue
        for nxt in adj.get(node, []):
            if nxt in seen:
                continue
            stack.append((nxt, d + 1, seen | {nxt}))
    return best


def audit(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows or "facts" not in rows[0] or "candidates" not in rows[0]:
        return None

    res = {}
    for rule in ("max", "min"):
        hit = n = 0
        for r in rows:
            facts = r["facts"]
            cands = r["candidates"]
            depths = [downstream_depth(facts, c["obj"]) for c in cands]
            target = max(depths) if rule == "max" else min(depths)
            preds = [i for i, d in enumerate(depths) if d == target]
            if len(preds) != 1:
                continue  # 并列时规则无法唯一决策，计为未命中（保守）
            n += 1
            if preds[0] == r["answer_index"]:
                hit += 1
        total = len(rows)
        res[rule] = f"{hit}/{total} = {hit/total:.4f} (可判定 {n})"
    return res


def main():
    out = {}
    for d in sorted(os.listdir(os.path.join(ROOT, "data"))):
        dd = os.path.join(ROOT, "data", d)
        if not os.path.isdir(dd):
            continue
        files = sorted(glob.glob(os.path.join(dd, "*.jsonl")))
        res = {}
        for f in files:
            r = audit(f)
            if r:
                res[os.path.basename(f)] = r
        if res:
            out[d] = res
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
