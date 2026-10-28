"""s83_shortcut_probe.py — 检验"下游链更长即答案"捷径是否被模型使用。

背景（2026-10-27 发现）：
  `gen_example(comp_mode='deep')` 让干扰链深度恒为 hop-2，即干扰候选的下游
  长度比正确候选短 1。因此"从两候选各走一遍、选更长的"在 train/dev/confirm
  **全部 100% 正确**（已实测）。这构成一个不需求解起点的捷径。

本脚本构造三组诊断集，用同一 checkpoint 对比：
  A. orig    原样 confirm（干扰短 1）—— 若模型用捷径应高分
  B. inverted 干扰链**比答案长 2**—— 若模型用捷径应系统性翻错（准确率≈0）
  C. tied     两候选下游等长—— 捷径无信息，只能真解

判读（预注册）：
  - 捷径主导：A 高（≥0.6）且 B 低（≤0.2）且 C ≈ 随机
  - 真求解：A、B 都高（B 也 ≥0.6），C 也高于随机
  - 无法区分：其余

用法：
  python s83_shortcut_probe.py --ckpt results/remote_R_CTRL_exactT_s53_ckpt.pt \
      --rounds 6 --out results/probe_s53.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

import s63_ldm_model as M
from s66_ldm_mvt_gen import Fact, all_reach, gen_example

ROOT = Path(__file__).parent
DEV = ROOT / 'data' / 'ldm_mvt5'


def build_variant(rng, hop, mode, n, comp_mode='deep'):
    """生成 hop 阶样本并重写干扰链。

    mode:
      orig     - 保持生成器输出
      inverted - 干扰链长 = hop+2（比答案长 3），答案仍是 chain[1]
      tied     - 干扰链长 = hop-1（与答案链等长）
    """
    out = []
    tries = 0
    while len(out) < n and tries < n * 40:
        tries += 1
        ex = gen_example(rng, 'probe', hop, comp_mode=comp_mode)
        if ex is None:
            continue
        if mode == 'orig':
            out.append(ex)
            continue
        # 重建：保留 answer 侧链，替换干扰侧链
        chain = [ex['e0']] + [p['obj'] for p in ex['proof']]
        ans_obj = chain[1]
        distractor = next(c['obj'] for c in ex['candidates']
                          if c['obj'] != ans_obj)
        # 干扰链要延伸的长度。注意：从 e0 出发走 hop 步只能有唯一终点，
        # 所以干扰链必须 < hop（否则产生第二个有效答案）。
        #   orig     : hop-1（生成器默认，干扰链总长 hop-1，比答案链短 1）
        #   tied     : hop-1（与答案链等长 -> 从 e0 走 hop 步到不了干扰终点）
        #   inverted : 不可行（会破坏唯一解），保留标记用于报错
        if mode == 'inverted':
            continue
        target_extra = hop - 1        # e0 -> distractor -> ... 共 hop-1 条边
        # 移除干扰侧原有事实，重建更长的链（用未占用的实体）
        used = set(chain) | {distractor}
        pool = [f'E{i:03d}' for i in range(300) if f'E{i:03d}' not in used]
        rng.shuffle(pool)
        need = target_extra - 1        # 已有 1 条 e0->distractor
        if need < 0 or len(pool) < need:
            continue
        new_facts = [Fact(ex['e0'], 'P', distractor)]
        prev = distractor
        for _ in range(max(0, need)):
            nxt = pool.pop()
            new_facts.append(Fact(prev, 'P', nxt))
            prev = nxt
        chain_facts = [Fact(chain[i], 'P', chain[i + 1])
                       for i in range(len(chain) - 1)]
        facts = chain_facts + new_facts
        fset = set(facts)
        # 复检：从 e0 走 hop 跳只有唯一答案；两候选一真一假
        if all_reach(fset, ex['e0'], hop) != {ex['e_answer']}:
            continue
        targets = sorted({f.obj for f in fset
                          if f.rel == 'P' and f.subj == ex['e0']})
        if len(targets) != 2 or ans_obj not in targets:
            continue
        flags = [ex['e_answer'] in all_reach(fset, t, hop - 1)
                 for t in targets]
        if sum(flags) != 1:
            continue
        rng.shuffle(facts)
        fact_str = ' ; '.join(f'{f.subj} {f.rel} {f.obj}' for f in facts)
        cands = [{'rel': 'P', 'obj': c['obj']} for c in ex['candidates']]
        cand_str = ' , '.join(f"{c['rel']} {c['obj']}" for c in cands)
        out.append({
            **{k: ex[k] for k in ('id', 'goal_name', 'goal', 'hop',
                                  'e0', 'e_answer', 'n_candidates',
                                  'n_distractors', 'proof')},
            'split': 'probe', 'candidates': cands,
            'answer_index': next(i for i, c in enumerate(cands)
                                 if c['obj'] == ans_obj),
            'comp_depth_by_index': {str(i): target_extra
                                    for i, c in enumerate(cands)
                                    if c['obj'] != ans_obj},
            'facts': [f.__dict__ for f in facts],
            'question': (f'[STATE] {fact_str} [CURRENT] {ex["e0"]} [GOAL] '
                         + ' '.join(['P'] * hop)
                         + f' [ACTIONS] {cand_str} [ANSWER]'),
        })
    return out


def _walk_len(facts, start):
    return len(facts)        # 占位：仅用于旧分支过滤


@torch.no_grad()
def eval_rows(model, rows, dev, r, bs=256):
    if not rows:
        return None
    ds = M.EncodedDataset(rows, max_len=256, fused=True)
    ok = 0
    for i in range(0, len(ds), bs):
        idx = list(range(i, min(i + bs, len(ds))))
        b = {k: (v.to(dev) if torch.is_tensor(v) else v)
             for k, v in ds.collate(idx).items()}
        out = model(b['ids'], b['pad_mask'], b['cand_pos'], b['cand_mask'],
                    n_rounds=r)
        lg = out['final_logits'].masked_fill(~b['cand_mask'], -1e9)
        pred = lg.argmax(1).tolist()
        ok += sum(int(p == int(ds.items[j]['label']))
                  for p, j in zip(pred, idx))
    return round(ok / len(ds), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--rounds', default='6,8')
    ap.add_argument('--hops', default='7,8')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=9001)
    ap.add_argument('--step-sup', action='store_true')
    ap.add_argument('--out', default='results/probe_shortcut.json')
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = M.build_model('loop', n_rounds=16, n_core=1, round_in_core=True,
                          pos='none', step_sup=args.step_sup).to(dev)
    sd = torch.load(args.ckpt, map_location=dev, weights_only=False)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    model.load_state_dict(sd, strict=False)
    model.eval()

    rng = random.Random(args.seed)
    hops = [int(h) for h in args.hops.split(',')]
    rounds = [int(r) for r in args.rounds.split(',')]

    # A：原样 confirm
    orig = [json.loads(l) for l in
            (DEV / 'confirm_depth.jsonl').read_text(encoding='utf-8')
            .splitlines() if l.strip()]
    payload = {'ckpt': args.ckpt, 'results': {}}
    for hop in hops:
        sub = [r for r in orig if r['hop'] == hop]
        variants = {'orig': sub}
        for mode in ('inverted', 'tied'):
            variants[mode] = build_variant(random.Random(args.seed + hop),
                                           hop, mode, args.n)
        for mode, rows in variants.items():
            for r in rounds:
                acc = eval_rows(model, rows, dev, r)
                key = f'hop{hop}_{mode}_r{r}'
                payload['results'][key] = {'acc': acc, 'n': len(rows)}
                print(json.dumps({'key': key, 'acc': acc, 'n': len(rows)}),
                      flush=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + '\n',
                              encoding='utf-8')
    print(json.dumps({'done': args.out}), flush=True)


if __name__ == '__main__':
    main()
