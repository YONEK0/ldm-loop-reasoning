"""s84_dual_task.py — 消除"下游链长"捷径的对偶任务（MVT-D）。

## 为什么要新任务

`s66` 的 MVT-5 用 `comp_mode='deep'`：干扰链恒为 hop−1 条边，正确链为 hop 条边。
于是"从两候选各走到尽头、选更长的"在全部数据上 100% 正确（实测 19750/19750）。
这是**纯链构造下保证唯一解的必然结果**，无法在原任务上消除。

实测后果（2026-10-27）：
- exact-T 的 hop7@r6 = 0.787，hop8@r6 = 0.469 —— 分界恰在"链长 ≤ r"，
  是长度比较而非深度外推；
- persist+step 的成功模型在训练深度上：hop4 orig 1.000 → 无捷径 0.607，
  hop6 orig 0.995 → 无捷径 0.520。**连"学会了"的深度也主要靠捷径。**

## 对偶任务设计（MVT-D）

保留 MVT-5 的全部形状（纯 P 链、2 候选、融合编码、单图），只改一处：
**正确答案是下游链更短的那条。**

构造：
- 长链：e0 → x1 → ... → x_{hop} （hop 条边，长度为 hop 的"诱饵"终点）
- 短链：e0 → y1 → ... → y_{hop-1} （hop−1 条边，长度 hop−1）
- 候选 = {x_hop, y_{hop-1}}，**答案 = y_{hop-1}**（更短的那条）

唯一性：从 e0 走 hop 步只能到 x_hop，y_{hop-1} 在 hop−1 步就到头。
但**任务问的是"哪条链能在 hop 步内到达终点"**——需要重新定义问句。

更简洁的等价表述（采用）：
**问句 = 从 e0 沿 P 走 hop 步，能到达哪个候选？**
- x_hop：恰好在第 hop 步到达 → 是答案；
- y_{hop-1}：第 hop−1 步到达，第 hop 步无后继 → 不是。

这样唯一解成立，且"选更长的"**仍然**是正确答案（x_hop 更长）。不解决问题。

## 最终采用：等长双链 + 只有一条连通

- 链 A：e0 → a1 → ... → a_{hop-1} → A_end（hop 条边，到 A_end）
- 链 B：e0 → b1 → ... → b_{hop-1} → B_end（hop 条边，到 B_end）
- **砍掉 B 的最后一条边**，并在 B_end 后接一段"孤立长链"（不可从 e0 到达）

候选 = {A_end, B_end}，答案 = A_end。
- 从 e0 走 hop 步：只到 A_end（B 链在第 hop−1 步后断开）
- 下游长度：A_end 之后可能还有延续；B_end 之后接的孤立长链更长
- **因此"选更长"会选错** → 捷径反转

实现时让 B_end 后的孤立链长度 = 随机 1..3（可长可短），
使"长度"与"正确性"**去相关**，而不是简单反转（反转仍是一个可利用的捷径）。

## 判据

用同一 checkpoint 对比 MVT-5（有捷径）与 MVT-D（无捷径）：
- 若 MVT-D 准确率 ≈ 随机（0.5）→ 模型此前的高分**完全依赖捷径**；
- 若 MVT-D 仍显著高于随机 → 存在真实的路径追踪能力。

用法：
  python s84_dual_task.py gen --out data/ldm_mvtd
  python s84_dual_task.py eval --ckpt results/X_ckpt.pt --rounds 6,8
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


def gen_mvtd(rng: random.Random, split: str, hop: int) -> dict | None:
    """生成一个 MVT-D 样本：两条等长链，只有一条在第 hop 步可达。

    必须消除的所有局部捷径（2026-10-27 修订，v2）：
      1. 下游链长（v1 已解决：两候选尾巴长度各自随机）
      2. **入边存在性**（v1 漏洞：干扰候选 b_end 无入边，1000/1000）
         → v2：b_end 也由一条边指向（来自一条独立短链），使"有没有入边"
           无区分度
      3. **出边存在性** → v2：两候选都挂尾巴，出边都有
      4. 出现次数/度数 → 让两候选在图中局部尽量对称

    v2 结构：
      e0 -> a1 -> ... -> a_hop           （答案链，hop 条边）
      e0 -> b1 -> ... -> b_{hop-1}       （干扰链，hop-1 条边，断在末尾）
      z  -> b_end                        （额外的入边，z 是孤立节点）
      a_end -> a_tail...                 （随机 1..3）
      b_end -> b_tail...                 （随机 1..3）
    """
    tail_a = rng.randint(1, 3)
    tail_b = rng.randint(1, 3)
    # 实体预算：e0(1) + A链(hop) + B链(hop-1) + b_end(1)
    #          + a_tail + b_tail + 入边源 z(1)
    need = 1 + hop + (hop - 1) + 1 + tail_a + tail_b + 1
    if need > len(POOL):
        return None
    ents = rng.sample(POOL, need)
    it = iter(ents)
    e0 = next(it)
    chain_a = [next(it) for _ in range(hop)]
    chain_b = [next(it) for _ in range(hop - 1)]
    b_end = next(it)
    a_tail = [next(it) for _ in range(tail_a)]
    b_tail = [next(it) for _ in range(tail_b)]
    z = next(it)

    facts = [Fact(e0, 'P', chain_a[0])]
    for i in range(hop - 1):
        facts.append(Fact(chain_a[i], 'P', chain_a[i + 1]))
    facts.append(Fact(e0, 'P', chain_b[0]))
    for i in range(hop - 2):
        facts.append(Fact(chain_b[i], 'P', chain_b[i + 1]))
    # 关键：不建 chain_b[-1] -> b_end（B 链在第 hop-1 步后断）
    # b_end 的入边来自独立节点 z，使"入边存在性"无区分度
    facts.append(Fact(z, 'P', b_end))
    # 两候选都挂尾巴，使出边存在性无区分度
    a_end = chain_a[-1]
    facts.append(Fact(a_end, 'P', a_tail[0]))
    for i in range(tail_a - 1):
        facts.append(Fact(a_tail[i], 'P', a_tail[i + 1]))
    facts.append(Fact(b_end, 'P', b_tail[0]))
    for i in range(tail_b - 1):
        facts.append(Fact(b_tail[i], 'P', b_tail[i + 1]))

    fset = set(facts)
    # 唯一性复检：从 e0 走 hop 步只能到达 a_end
    reach = {e0}
    for _ in range(hop):
        reach = {f.obj for f in fset if f.subj in reach}
        if not reach:
            return None
    if reach != {a_end}:
        return None
    targets = sorted({f.obj for f in fset if f.subj == e0})
    if set(targets) != {chain_a[0], chain_b[0]}:
        return None
    # 候选 = 两条链的"末端"（候选不是 e0 的直接子，而是路径终点）
    cands = [{'rel': 'P', 'obj': a_end}, {'rel': 'P', 'obj': b_end}]
    order = [0, 1]
    rng.shuffle(order)
    cands = [cands[i] for i in order]
    ans_idx = next(i for i, c in enumerate(cands) if c['obj'] == a_end)

    rng.shuffle(facts)
    fact_str = ' ; '.join(f'{f.subj} {f.rel} {f.obj}' for f in facts)
    cand_str = ' , '.join(f"{c['rel']} {c['obj']}" for c in cands)
    question = (f'[STATE] {fact_str} [CURRENT] {e0} [GOAL] '
                + ' '.join(['P'] * hop)
                + f' [ACTIONS] {cand_str} [ANSWER]')
    return {
        'id': f'{split}_{hop}_{rng.randint(0, 10**9):09d}',
        'split': split, 'goal_name': f'P{hop}', 'goal': ['P'] * hop,
        'hop': hop, 'e0': e0, 'e_answer': a_end,
        'candidates': cands, 'answer_index': ans_idx,
        'n_candidates': 2, 'n_distractors': 0,
        'comp_depth_by_index': {},          # 无"深度"概念
        'proof': [{'subj': chain_a[i] if i else e0, 'rel': 'P',
                   'obj': chain_a[i]}
                  for i in range(hop)],
        'facts': [f.__dict__ for f in facts],
        'question': question,
        'divergence': {'a_tail': tail_a, 'b_tail': tail_b},
    }


def gen_split(rng, split, hops, per):
    out = []
    for h in hops:
        n = 0
        tries = 0
        while n < per and tries < per * 50:
            tries += 1
            ex = gen_mvtd(rng, split, h)
            if ex is not None:
                out.append(ex)
                n += 1
    return out


def cmd_gen(a):
    rng = random.Random(a.seed)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    plan = {
        'train.jsonl': (('train',), list(range(2, a.train_max + 1)), a.per),
        'dev_random.jsonl': (('dev',), list(range(2, a.train_max + 1)), 200),
        'dev_ood.jsonl': (('dev',), list(range(a.train_max + 1, a.ood_max + 1)), 200),
    }
    manifest = {'seed': a.seed, 'task': 'MVT-D (shortcut-free)', 'splits': {}}
    for fn, ((split,), hops, per) in plan.items():
        rows = gen_split(rng, split, hops, per)
        (outdir / fn).write_text(
            '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n',
            encoding='utf-8')
        manifest['splits'][fn] = {'n': len(rows), 'hops': hops}
        print(f'{fn}: {len(rows)} rows, hops={hops}')
    (outdir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')


def cmd_eval(a):
    import torch


def _pick_device():
    """设备选择：优先 NPU（昇腾 910B），其次 CUDA，最后 CPU。"""
    import torch
    if hasattr(torch, 'npu') and getattr(torch.npu, 'is_available',
                                          lambda: False)():
        return torch.device('npu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
    import s63_ldm_model as M
    dev = _pick_device()
    model = M.build_model('loop', n_rounds=16, n_core=1, round_in_core=True,
                          pos='none', step_sup=a.step_sup).to(dev)
    sd = torch.load(a.ckpt, map_location=dev, weights_only=False)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    model.load_state_dict(sd, strict=False)
    model.eval()
    rows = [json.loads(l) for l in (Path(a.data) / a.file)
            .read_text(encoding='utf-8').splitlines() if l.strip()]
    ds = M.EncodedDataset(rows, max_len=256, fused=True)
    res = {}
    with torch.no_grad():
        for r in [int(x) for x in a.rounds.split(',')]:
            ok = 0
            by = {}
            for i in range(0, len(ds), 256):
                idx = list(range(i, min(i + 256, len(ds))))
                b = {k: (v.to(dev) if torch.is_tensor(v) else v)
                     for k, v in ds.collate(idx).items()}
                out = model(b['ids'], b['pad_mask'], b['cand_pos'],
                            b['cand_mask'], n_rounds=r)
                lg = out['final_logits'].masked_fill(~b['cand_mask'], -1e9)
                pred = lg.argmax(1).tolist()
                for p, j in zip(pred, idx):
                    it = ds.items[j]
                    hit = int(p == it['label'])
                    ok += hit
                    d = by.setdefault(str(it['hop']), [0, 0])
                    d[0] += 1
                    d[1] += hit
            res[f'r{r}'] = {'acc': round(ok / len(ds), 4),
                            'by_hop': {h: round(v[1] / v[0], 3)
                                       for h, v in sorted(by.items())}}
            print(json.dumps({f'r{r}': res[f'r{r}']}), flush=True)
    Path(a.out).write_text(json.dumps({'ckpt': a.ckpt, 'file': a.file,
                                       'results': res}, indent=2) + '\n',
                           encoding='utf-8')
    print(json.dumps({'done': a.out}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gen')
    g.add_argument('--out', default='data/ldm_mvtd')
    g.add_argument('--seed', type=int, default=2027)
    g.add_argument('--per', type=int, default=2000)
    g.add_argument('--train-max', type=int, default=6)
    g.add_argument('--ood-max', type=int, default=8)
    g.set_defaults(fn=cmd_gen)
    e = sub.add_parser('eval')
    e.add_argument('--ckpt', required=True)
    e.add_argument('--data', default='data/ldm_mvtd')
    e.add_argument('--file', default='dev_random.jsonl')
    e.add_argument('--rounds', default='6,8')
    e.add_argument('--step-sup', action='store_true')
    e.add_argument('--out', default='results/mvtd_eval.json')
    e.set_defaults(fn=cmd_eval)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
