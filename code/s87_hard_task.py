"""s87_hard_task.py — MVT-E：随深度增长的难度（解决 MVT-D v2 的平台问题）。

## 为什么需要 MVT-E

MVT-D v2（`s84_dual_task.py`）的构造对 hop≥3 是**同构**的：
两条链、一条在第 hop 步可达。这导致模型在 hop3–6 上表现落在同一平台
（P0-1 实测：0.755–0.890，落差仅 0.08–0.24），**无法区分深度能力**。

## MVT-E 的设计：让每个额外 hop 都增加真实计算量

单条主链，第 k 跳处引入 **k 条分叉**（k 随深度增长）：
- 在第 k 步，正确下一跳与 (k−1) 个干扰下一跳并列；
- 干扰分支在**不同深度**重新汇入或终止，迫使模型逐跳消歧；
- 只有走完全程才能确定终点。

计算量随深度增长：hop h 需要比较 h 个分叉点，
且每个分叉点的干扰数随位置递增（1, 2, 3, ..., h−1）。

## 为什么这是"真难度"而非"更多噪声"

关键设计：**干扰分支会在更深处汇回主链**。
模型不能靠"哪条能走到底"判断（每条都能走一段），
必须记住"我在第几步、当前在哪个节点"，这正是串行深度的核心。

## 与 MVT-D 的结构对比

| | MVT-D v2 | MVT-E |
|---|---|---|
| 主链 | 1 条 | 1 条 |
| 干扰链 | 1 条（在 hop−1 断） | h−1 条（第 k 跳处 k−1 条分叉） |
| 干扰是否可达终点 | 否 | **部分会汇回** |
| hop 增长时的计算量 | 恒定（同构） | **递增** |
| 单步可解 | 否 | 否 |

## 用法

  python s87_hard_task.py gen --out data/ldm_mvte --train-max 6 --ood-max 8
  python s87_hard_task.py audit --data data/ldm_mvte   # 捷径审计
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent
POOL = [f'E{i:03d}' for i in range(300)]


@dataclass(frozen=True)
class Fact:
    subj: str
    rel: str
    obj: str


def gen_mvte(rng: random.Random, split: str, hop: int,
             max_branch: int = 3) -> dict | None:
    """生成一个 MVT-E 样本。

    结构：
      - 主链 e0 -> m1 -> ... -> m_hop（答案）
      - 在第 k 跳（k=2..hop）的每个主链节点 m_{k-1} 处，
        挂 (min(k-1, max_branch) - 1) 条干扰分支
      - 每条干扰分支长度 L_k 随机 1..(hop-k+1)，**在较深处汇回主链或终止**
      - 候选 = {m_hop, 某条干扰分支的末端}，两候选下游长度随机（去捷径）

    唯一性：从 e0 走 hop 步，只有 m_hop 是"主链终点"；
    干扰分支的末端都在 < hop 步内终止或汇回，不产生第二个 hop 步终点。
    """
    # 实体预算：主链 hop + 干扰节点（按分叉数估算）
    n_branch_pts = max(0, hop - 1)
    avg_branch = (1 + min(hop, max_branch)) / 2
    est_distract = int(n_branch_pts * avg_branch) + hop
    need = 1 + hop + est_distract + 6
    if need > len(POOL):
        return None
    ents = rng.sample(POOL, need)
    it = iter(ents)
    e0 = next(it)
    main = [next(it) for _ in range(hop)]        # 主链 m1..m_hop

    facts = []
    prev = e0
    for i, nxt in enumerate(main):
        facts.append(Fact(prev, 'P', nxt))
        prev = nxt

    # 在 e0 处固定挂 1 条干扰（保证两候选都是 e0 的直接子节点）。
    # 干扰链独立展开、**绝不汇入主链**，且总长 < hop，
    # 这样从 e0 走 hop 步只有主链终点。
    dist0 = next(it)
    facts.append(Fact(e0, 'P', dist0))
    max_L0 = hop - 2                     # e0->dist0 已占 1 步
    L0 = rng.randint(1, max(1, max_L0)) if max_L0 >= 1 else 1
    cur = dist0
    for _ in range(max(0, L0 - 1)):
        try:
            nn = next(it)
        except StopIteration:
            return None
        facts.append(Fact(cur, 'P', nn))
        cur = nn

    # 后续主链节点 m_{k-1}（k=3..hop）挂分叉。
    #
    # 设计要点（决定任务难度）：
    #   干扰分支**在中途汇回主链**，因此它在第 hop 步确实能到达同一终点。
    #   模型不能靠"哪条能走到底"排除它，必须判断
    #   "从 e0 出发、恰好走 hop 步的连续路径是否唯一"。
    #   分叉点越多（k 越大），需要维护的并行状态越多 → 难度随深度增长。
    #
    # 唯一性保证：干扰分支汇回点必须在主链更深处，且汇回后不再分叉，
    #   这样从 e0 走 hop 步仍只有 main[-1] 一个终点。
    for k in range(3, hop + 1):
        src = main[k - 2]                        # 第 k 跳的起点
        n_br = min(k - 2, max_branch - 1)
        for _ in range(max(0, n_br)):
            try:
                nxt_ent = next(it)
            except StopIteration:
                return None
            facts.append(Fact(src, 'P', nxt_ent))
            # 干扰链长度 1..(hop-k)，然后汇回 main[j]（j > k-1）
            max_L = max(1, hop - k)
            L = rng.randint(1, max(1, max_L))
            cur = nxt_ent
            for _ in range(L - 1):
                try:
                    nn = next(it)
                except StopIteration:
                    return None
                facts.append(Fact(cur, 'P', nn))
                cur = nn
            # 汇回主链更深处（j 从 k 到 hop-1，即可保证不改变 hop 步终点）
            if k < hop:
                j = rng.randrange(k, hop)
                facts.append(Fact(cur, 'P', main[j]))

    fset = set(facts)
    # 唯一性复检：从 e0 走 hop 步应只到 m_hop
    reach = {e0}
    for _ in range(hop):
        reach = {f.obj for f in fset if f.subj in reach}
    if reach != {main[-1]}:
        return None
    # 起点必须恰好两个出口（两候选 = e0 的两个直接子节点）
    first = sorted({f.obj for f in fset if f.subj == e0})
    if len(first) != 2:
        return None
    # 问题："从 e0 走 hop 步能到达哪个候选"——答案 = main[0]
    # 选 main[0] 与 dist0 作为候选（两者都是 e0 的直接子节点）
    cands = [{'rel': 'P', 'obj': main[0]}, {'rel': 'P', 'obj': dist0}]
    order = [0, 1]
    rng.shuffle(order)
    cands = [cands[i] for i in order]
    ans_idx = next(i for i, c in enumerate(cands) if c['obj'] == main[0])

    rng.shuffle(facts)
    fact_str = ' ; '.join(f'{f.subj} {f.rel} {f.obj}' for f in facts)
    cand_str = ' , '.join(f"{c['rel']} {c['obj']}" for c in cands)
    question = (f'[STATE] {fact_str} [CURRENT] {e0} [GOAL] '
                + ' '.join(['P'] * hop)
                + f' [ACTIONS] {cand_str} [ANSWER]')
    proof = [{'subj': (e0 if i == 0 else main[i - 1]), 'rel': 'P',
              'obj': main[i]} for i in range(hop)]
    return {
        'id': f'{split}_{hop}_{rng.randint(0, 10**9):09d}',
        'split': split, 'goal_name': f'P{hop}', 'goal': ['P'] * hop,
        'hop': hop, 'e0': e0, 'e_answer': main[0],
        'candidates': cands, 'answer_index': ans_idx,
        'n_candidates': 2, 'n_distractors': 0,
        'comp_depth_by_index': {},
        'proof': proof,
        'facts': [f.__dict__ for f in facts],
        'question': question,
        'n_facts': len(facts),
        'n_branch_points': max(0, hop - 1),
    }


def gen_split(rng, split, hops, per, max_branch=3):
    out = []
    for h in hops:
        n = 0
        tries = 0
        while n < per and tries < per * 80:
            tries += 1
            ex = gen_mvte(rng, split, h, max_branch=max_branch)
            if ex is not None:
                out.append(ex)
                n += 1
        if n < per:
            print(f'  [warn] hop{h}: only {n}/{per} generated')
    return out


def audit(rows, name):
    """捷径审计：逐条朴素规则，任一接近 1.0 则任务无区分力。"""
    import collections
    tot = len(rows)
    if not tot:
        print(f'{name}: EMPTY')
        return {}
    stats = collections.Counter()
    for r in rows:
        g = {(f['subj'], f['obj']) for f in r['facts']}
        ans = int(r['answer_index'])

        def walk(s):
            cur, d, seen = s, 0, set()
            while True:
                if cur in seen:
                    break
                seen.add(cur)
                nxt = [o for a, o in g if a == cur]
                if len(nxt) != 1:
                    break
                cur, d = nxt[0], d + 1
            return d

        L = [walk(c['obj']) for c in r['candidates']]
        if L[0] != L[1]:
            stats['pick_longer'] += int((0 if L[0] > L[1] else 1) == ans)
            stats['pick_shorter'] += int((0 if L[0] < L[1] else 1) == ans)
        ins = [any(f['obj'] == c['obj'] for f in r['facts'])
               for c in r['candidates']]
        if ins[0] != ins[1]:
            stats['has_in_edge'] += int((0 if ins[0] else 1) == ans)
        outs = [any(f['subj'] == c['obj'] for f in r['facts'])
                for c in r['candidates']]
        if outs[0] != outs[1]:
            stats['has_out_edge'] += int((0 if outs[0] else 1) == ans)
        stats['pick_idx0'] += int(ans == 0)
        stats['pick_idx1'] += int(ans == 1)
    print(f'\n=== {name} (n={tot}) ===')
    res = {}
    for k in ('pick_longer', 'pick_shorter', 'has_in_edge', 'has_out_edge',
              'pick_idx0', 'pick_idx1'):
        v = stats[k] / tot
        flag = ' ← 捷径!' if v > 0.75 else ''
        print(f'  {k:<16} {v:.3f}{flag}')
        res[k] = round(v, 3)
    return res


def cmd_gen(a):
    rng = random.Random(a.seed)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    plan = {
        'train.jsonl': ('train', list(range(2, a.train_max + 1)), a.per),
        'dev_random.jsonl': ('dev', list(range(2, a.train_max + 1)), 200),
        'dev_ood.jsonl': ('dev', list(range(a.train_max + 1, a.ood_max + 1)),
                          200),
    }
    manifest = {'seed': a.seed, 'task': 'MVT-E (depth-scaled difficulty)',
                'max_branch': a.max_branch, 'splits': {}}
    for fn, (split, hops, per) in plan.items():
        rows = gen_split(rng, split, hops, per, max_branch=a.max_branch)
        (outdir / fn).write_text(
            '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n',
            encoding='utf-8')
        manifest['splits'][fn] = {'n': len(rows), 'hops': hops}
        print(f'{fn}: {len(rows)} rows, hops={hops}')
    (outdir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')


def cmd_audit(a):
    d = Path(a.data)
    for fn in ('train.jsonl', 'dev_random.jsonl', 'dev_ood.jsonl'):
        p = d / fn
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text(encoding='utf-8')
                .splitlines() if l.strip()]
        audit(rows, fn)
        # 统计每 hop 的事实数（检验难度是否随深度增长）
        import collections
        by = collections.defaultdict(list)
        for r in rows:
            by[r['hop']].append(r.get('n_facts', len(r['facts'])))
        print('  每 hop 平均事实数:',
              {h: round(sum(v) / len(v), 1) for h, v in sorted(by.items())})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gen')
    g.add_argument('--out', default='data/ldm_mvte')
    g.add_argument('--seed', type=int, default=2028)
    g.add_argument('--per', type=int, default=2000)
    g.add_argument('--train-max', type=int, default=6)
    g.add_argument('--ood-max', type=int, default=8)
    g.add_argument('--max-branch', type=int, default=3)
    g.set_defaults(fn=cmd_gen)
    au = sub.add_parser('audit')
    au.add_argument('--data', default='data/ldm_mvte')
    au.set_defaults(fn=cmd_audit)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
