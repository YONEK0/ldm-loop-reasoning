"""s66: LDM 最小可行任务（MVT）生成——纯 P 链 / 二候选 / 零背景干扰。

定位（LDM 日志 §7 判定后的选项 B 第一步）：把任务降到文献已证明可学的
pp-hop 难度级，先打通"模型能学"（可行性第 2 层），再逐轴加回难度。

与 s62（正式任务 v2/v3/v5）的差异——每条都是有意为之的难度削减：
  - 关系只有 P（有向）；无 M/S 对称性。
  - 候选固定 2 个：1 正确 + 1 深断链竞争边（断链深度 = hop-2，即必须
    追踪到最后才能排除；这是从 v2 保留的唯一难度轴——深度）。
  - 背景干扰事实 = 0（干扰轴整体移除，后续逐轴加回）。
  - 实体池 24（对齐"小字母表"精神；池 16 的 hop2 样本空间约 4.4 万，
    跨切分期望碰撞约 14%，24 降到 ~2.3%，且生成时显式排除与 train 的
    碰撞，实际为 0）。
  - 训练 hop 2..3，深度外推 hop 4..6（保留循环深度的判定轴）。

切分（MVT 专属，正式 confirm_* 冻结集不涉及、不消耗）：
  train        hop{2,3} × --train-per-hop
  dev_random   hop{2,3} × --dev-per-hop
  confirm_depth hop{4,5,6} × --depth-per-hop

输出目录 data/ldm_mvt/，字段与 s62 兼容（s63/s65 无需改动）。
纯 CPU。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data' / 'ldm_mvt'
OUT = ROOT / 'results' / 'ldm_mvt_gen.json'
SEED = 2041
POOL = [f'E{i:03d}' for i in range(32)]


@dataclass(frozen=True)
class Fact:
    subj: str
    rel: str
    obj: str


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'))


def all_reach(facts: set[Fact], start: str, n_hops: int) -> set[str]:
    frontier = {start}
    for _ in range(n_hops):
        nxt = {f.obj for f in facts
               if f.rel == 'P' and f.subj in frontier}
        frontier = nxt
        if not frontier:
            break
    return frontier


def gen_example(rng: random.Random, split: str, hop: int,
                comp_mode: str = 'deep') -> dict | None:
    k = hop
    deep = k - 2 if comp_mode == 'deep' else rng.randint(0, k - 2)
    need = (k + 1) + (1 + deep)
    ents = rng.sample(POOL, need)
    it = iter(ents)
    chain = [next(it) for _ in range(k + 1)]
    e0, e_answer = chain[0], chain[-1]
    chain_facts = [Fact(chain[i], 'P', chain[i + 1]) for i in range(k)]
    b = next(it)
    comp_facts = [Fact(e0, 'P', b)]
    prev = b
    for _ in range(deep):
        nxt = next(it)
        comp_facts.append(Fact(prev, 'P', nxt))
        prev = nxt
    facts = chain_facts + comp_facts
    rng.shuffle(facts)
    fset = set(facts)

    # solver 复检（不信任构造）：
    if all_reach(fset, e0, k) != {e_answer}:
        return None
    targets = sorted({f.obj for f in fset
                      if f.rel == 'P' and f.subj == e0})
    if len(targets) != 2 or chain[1] not in targets:
        return None
    flags = [e_answer in all_reach(fset, t, k - 1) for t in targets]
    if sum(flags) != 1:
        return None

    order = [0, 1]
    rng.shuffle(order)
    candidates = [{'rel': 'P', 'obj': targets[i]} for i in order]
    answer_index = next(i for i, c in enumerate(candidates)
                        if c['obj'] == chain[1])
    fact_str = ' ; '.join(f'{f.subj} {f.rel} {f.obj}' for f in facts)
    cand_str = ' , '.join(f"{c['rel']} {c['obj']}" for c in candidates)
    question = (f'[STATE] {fact_str} [CURRENT] {e0} [GOAL] '
                + ' '.join(['P'] * k)
                + f' [ACTIONS] {cand_str} [ANSWER]')
    key = sha256_text(canonical_json({
        'facts': sorted([f.__dict__ for f in facts],
                        key=lambda d: (d['subj'], d['rel'], d['obj'])),
        'e0': e0, 'hop': k}))
    return {
        'id': key[:16], 'split': split, 'goal_name': f'P{k}',
        'goal': ['P'] * k, 'hop': k, 'e0': e0, 'e_answer': e_answer,
        'candidates': candidates, 'answer_index': answer_index,
        'n_candidates': 2, 'n_distractors': 0,
        'comp_depth_by_index': {str(i): deep for i, c in enumerate(candidates)
                                if c['obj'] != chain[1]},
        'proof': [f.__dict__ for f in chain_facts],
        'facts': [f.__dict__ for f in facts],
        'question': question,
    }


def gen_split(rng: random.Random, split: str, hops, per_hop: int,
              ban_keys: set[str] | None = None,
              comp_mode: str = 'deep') -> list[dict]:
    out: dict[str, dict] = {}
    attempts = 0
    ban = ban_keys or set()
    while len(out) < per_hop * len(hops) and attempts < per_hop * 400:
        attempts += 1
        hop = hops[len(out) % len(hops)]      # 轮转保持 hop 均衡
        ex = gen_example(rng, split, hop, comp_mode)
        if ex is None or ex['id'] in out or ex['id'] in ban:
            continue
        out[ex['id']] = ex
    if len(out) < per_hop * len(hops):
        raise RuntimeError(f'生成不足 {split}: {len(out)}')
    return list(out.values())


def audit(rows: list[dict], name: str, train_keys: set[str]) -> dict:
    bad = 0
    for ex in rows:
        fset = {Fact(f['subj'], f['rel'], f['obj']) for f in ex['facts']}
        ok = (all_reach(fset, ex['e0'], ex['hop']) == {ex['e_answer']}
              and sum(ex['e_answer'] in all_reach(fset, c['obj'],
                                                  ex['hop'] - 1)
                      for c in ex['candidates']) == 1)
        bad += (not ok)
    keys = {ex['id'] for ex in rows}
    return {'split': name, 'n': len(rows), 'solver_bad': bad,
            'train_key_overlap': len(keys & train_keys),
            'answer_pos': dict(sorted(Counter(
                ex['answer_index'] for ex in rows).items())),
            'hop_hist': dict(sorted(Counter(ex['hop'] for ex in rows).items()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-per-hop', type=int, default=6000)
    ap.add_argument('--dev-per-hop', type=int, default=500)
    ap.add_argument('--depth-per-hop', type=int, default=250)
    ap.add_argument('--train-hops', default='2,3',
                    help='训练 hop 列表（MVT-2: 2,3,4,5）')
    ap.add_argument('--depth-hops', default='4,5,6',
                    help='外推 hop 列表（MVT-2: 6,7,8）')
    ap.add_argument('--data-dir', default='data/ldm_mvt')
    ap.add_argument('--comp-mode', default='deep', choices=['deep', 'varied'],
                    help='deep=竞争边总在最深处断（MVT/2）；varied=断链深度'
                         '均匀采样（MVT-3，摧毁轮次统计捷径）')
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()

    global DATA_DIR
    DATA_DIR = Path(args.data_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_hops = [int(h) for h in args.train_hops.split(',')]
    depth_hops = [int(h) for h in args.depth_hops.split(',')]
    assert max(train_hops) < min(depth_hops), '外推集必须深于训练集'
    splits = {}
    cm = args.comp_mode
    def _gen(rng, split, hops, per, ban=None):
        return gen_split(rng, split, hops, per, ban, comp_mode=cm)
    train = _gen(random.Random(SEED + 1), 'train', train_hops,
                 args.train_per_hop)
    splits['train'] = train
    train_keys = {ex['id'] for ex in train}
    dev = _gen(random.Random(SEED + 2), 'dev_random', train_hops,
               args.dev_per_hop, ban=train_keys)
    splits['dev_random'] = dev
    depth = _gen(random.Random(SEED + 3), 'confirm_depth', depth_hops,
                 args.depth_per_hop, ban=train_keys)
    splits['confirm_depth'] = depth

    audits = [audit(v, k, train_keys) for k, v in splits.items()]

    file_hashes = {}
    if args.write:
        for name, rows in splits.items():
            p = DATA_DIR / f'{name}.jsonl'
            with p.open('w', encoding='utf-8', newline='\n') as f:
                for ex in rows:
                    f.write(json.dumps(ex, ensure_ascii=False) + '\n')
            file_hashes[name] = sha256_text(p.read_text(encoding='utf-8'))
        (DATA_DIR / 'manifest.json').write_text(json.dumps({
            'seed': SEED, 'entity_pool': len(POOL),
            'task': 'MVT: pure-P chain, binary candidates, 0 bg, deep-break comp',
            'train_hops': [2, 3], 'depth_extrapolation_hops': [4, 5, 6],
            'comp_depth_rule': 'hop-2 (deep break only)',
            'splits': {k: {'n': len(v), 'sha256': file_hashes.get(k)}
                       for k, v in splits.items()},
            'audits': audits,
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    OUT.write_text(json.dumps({'args': vars(args), 'audits': audits},
                              ensure_ascii=False, indent=2) + '\n',
                   encoding='utf-8')
    print(json.dumps({'audits': audits,
                      'files': {k: str(DATA_DIR / f'{k}.jsonl')
                                for k in splits}}, ensure_ascii=False,
                     indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
