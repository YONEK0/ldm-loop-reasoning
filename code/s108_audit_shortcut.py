#!/usr/bin/env python3
"""s108_audit_shortcut.py — 实测每个数据集的"最大捷径"命中率。

捷径定义（来自 docs/errors/err-task-shortcut.md）：
  "选下游链更长的候选"——即用 comp_depth_by_index 取最大值对应的候选，
  完全不看输入图结构。

输出：每个数据集 train/dev 上的命中率。1.0 = 100% 污染，不可用于能力结论。
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def shortcut_acc(rows):
    """实测"下游链长"捷径。

    数据编码（2026-10-30 实测）：
      `comp_depth_by_index` 只列出**干扰项**的下游链长；
      **不在键里的那个候选**就是正确答案。
      因此规则="选 comp_depth 键之外的候选"，完全不看输入图结构。

    cd 为空（如 MVT-D v2）时该捷径不可用，返回 None。
    """
    hit = 0
    n = 0
    for r in rows:
        cd = r.get("comp_depth_by_index")
        if not cd:
            continue
        keys = set(int(k) for k in cd)
        cand_idx = list(range(r.get("n_candidates") or len(r.get("candidates") or [])))
        preds = [i for i in cand_idx if i not in keys]
        if len(preds) != 1:
            continue
        n += 1
        if preds[0] == r.get("answer_index"):
            hit += 1
    return hit, n


def main():
    out = {}
    for d in sorted(os.listdir(os.path.join(ROOT, "data"))):
        dd = os.path.join(ROOT, "data", d)
        if not os.path.isdir(dd):
            continue
        files = sorted(glob.glob(os.path.join(dd, "*.jsonl")))
        if not files:
            continue
        res = {}
        for f in files:
            rows = load(f)
            if not rows or "comp_depth_by_index" not in rows[0]:
                res[os.path.basename(f)] = "no comp_depth field"
                continue
            hit, n = shortcut_acc(rows)
            res[os.path.basename(f)] = f"{hit}/{n} = {hit/n:.4f}" if n else "n/a"
        out[d] = res
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
