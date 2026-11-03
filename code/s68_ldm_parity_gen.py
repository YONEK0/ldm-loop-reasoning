"""s68: LDM-parity 任务生成（方向 3：文献已验证的循环友好任务族）。

任务：n 个槽位，每槽一位（0/1），答案 = 1 的个数的奇偶性（二分类）。
难度轴 = 1 的个数 k（PonderNet 原文协议：训练 64 槽/k∈[1,48]，
外推 96 槽/k∈[49,96]）。

为何选 parity：注意力是线性聚合，模 2 计数不可单次前向完成——
并行捷径在此原理上不可行（Hahn 2020 理论限制的经典例），与图任务
（一次内容匹配一跳）形成精确对照。若循环轮次在任何任务上有计算
角色，必在此。

编码（零改动复用 s63 encode_example_fused）：
  facts  = 每槽一条自环 (E{slot}, P|M, E{slot})，P=1 位，M=0 位
  goal   = []（无路径）；e0 固定 E000
  candidates = [(P, E290=EVEN), (M, E291=ODD)]，answer_index = parity
  hop    = k（难度即 1 的个数，供 by_hop 报告）

输出 data/ldm_parity/{train,dev_random,confirm_depth}.jsonl + manifest。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / 'results' / 'ldm_parity_gen.json'


def gen(rng: random.Random, split: str, n_slots: int, k_range) -> dict:
    k = rng.randint(*k_range)
    ones = set(rng.sample(range(n_slots), k))
    facts = [{'subj': f'E{i:03d}', 'rel': 'P' if i in ones else 'M',
              'obj': f'E{i:03d}'} for i in range(n_slots)]
    parity = k % 2
    cands = [{'rel': 'P', 'obj': 'E290'}, {'rel': 'M', 'obj': 'E291'}]
    content = {'n': n_slots, 'k': k, 'facts': facts}
    ex = {
        'id': hashlib.sha256(json.dumps(content, sort_keys=True)
                             .encode()).hexdigest()[:16],
        'split': split, 'goal_name': 'parity', 'goal': [], 'hop': k,
        'e0': 'E000', 'e_answer': 'E000',
        'candidates': cands, 'answer_index': parity, 'n_candidates': 2,
        'n_distractors': 0, 'comp_depth_by_index': {}, 'proof': [],
        'facts': facts,
        'question': f'[PARITY] n={n_slots} k={k} -> {"ODD" if parity else "EVEN"}',
    }
    return ex


def gen_split(seed: int, split: str, n_slots: int, k_range, count: int,
              ban: set[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    out: dict[str, dict] = {}
    ban = ban or set()
    attempts = 0
    while len(out) < count and attempts < count * 50:
        attempts += 1
        ex = gen(rng, split, n_slots, k_range)
        if ex['id'] in out or ex['id'] in ban:
            continue
        out[ex['id']] = ex
    if len(out) < count:
        raise RuntimeError(f'parity 生成不足 {split}')
    return list(out.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data/ldm_parity')
    ap.add_argument('--train-n', type=int, default=12000)
    ap.add_argument('--dev-n', type=int, default=1000)
    ap.add_argument('--confirm-n', type=int, default=750)
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train = gen_split(2051, 'train', 64, (1, 48), args.train_n)
    ban = {ex['id'] for ex in train}
    dev = gen_split(2052, 'dev_random', 64, (1, 48), args.dev_n, ban)
    confirm = gen_split(2053, 'confirm_depth', 96, (49, 96),
                        args.confirm_n, ban)

    audits = {}
    hashes = {}
    for name, rows in (('train', train), ('dev_random', dev),
                       ('confirm_depth', confirm)):
        par = Counter(ex['answer_index'] for ex in rows)
        ks = [ex['hop'] for ex in rows]
        audits[name] = {
            'n': len(rows), 'parity_balance': dict(par),
            'k_min': min(ks), 'k_max': max(ks),
            'k_mean': round(sum(ks) / len(ks), 1),
            'overlap_vs_train': 0 if name == 'train' else len(
                {ex['id'] for ex in rows} & ban),
        }
        if args.write:
            p = data_dir / f'{name}.jsonl'
            with p.open('w', encoding='utf-8', newline='\n') as f:
                for ex in rows:
                    f.write(json.dumps(ex, ensure_ascii=False) + '\n')
            hashes[name] = hashlib.sha256(
                p.read_text(encoding='utf-8').encode()).hexdigest()
    if args.write:
        (data_dir / 'manifest.json').write_text(json.dumps({
            'task': 'parity (PonderNet protocol)', 'seed_base': 2051,
            'train': '64 slots, k=1..48', 'confirm': '96 slots, k=49..96',
            'audits': audits, 'sha256': hashes,
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    OUT.write_text(json.dumps({'audits': audits}, ensure_ascii=False,
                              indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'audits': audits}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
