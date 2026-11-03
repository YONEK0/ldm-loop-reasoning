"""s67: LDM 退出控制器训练与硬退出评测（第 4 层，规格 §5）。

前置证据（LDM 日志 §9 前的 §8/首轮结果）：D8 动态深度训练后模型在
任意轮数 1–6 均可作答（深度鲁棒读出），使样本级退出控制成为可测命题。

训练（单脚本内含规格 §5.3 阶段 B→C）：
  - D8：每批均匀采样 R ∈ [dyn_min, r_max]；
  - D2 逐轮监督：每轮 probe logits（共享 scorer，pre-Coda）都有 CE；
  - 软退出权重（§5.2）：q_t 为继续概率，w_t = s_t(1-q_t)，末轮吸收
    剩余质量；L = Σ w_t·CE_t + α·mean_t CE_t + λ(step)·E[rounds]；
  - λ 线性爬坡（阶段 B λ=0 → 阶段 C λ_max）。

推理（§5.4 硬退出）：
  - 每样本独立判定：t ≥ r_min 且 1-q_t ≥ τ 即退出；最迟 r_max 强制；
  - 记录逐样本退出轮数、按 hop 的平均轮数、退出直方图、精度；
  - 真实跳过口径：逐样本退出轮数是策略账；批内实际执行到批内最大
    退出轮（同批共享计算）。profiler 级真跳过验证是规格收尾项 #7。

预注册判读（跑前写定，LDM 日志 §9）：
  通过 = 硬退出精度 ≥ 全深度精度 − 2pp
       且 avg_rounds(hop3) > avg_rounds(hop2) + 0.15
       且 mean_rounds < r_max（确有节省）。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import s63_ldm_model as M

OUT_DEFAULT = Path('results/mvt_exit_s17.json')


def to_dev(batch, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def soft_exit_loss(out, cand_mask, label, alpha, lam):
    rounds = out['rounds']
    R = len(rounds)
    B = label.size(0)
    ce_list = [F.cross_entropy(r['logits'].masked_fill(~cand_mask, -1e9),
                               label) for r in rounds]
    q_list = [r['q'] for r in rounds]          # (B,) 每轮
    s = torch.ones(B, device=label.device)
    E = torch.zeros(B, device=label.device)
    L_exit = torch.zeros((), device=label.device)
    for t in range(R):
        w_t = s * (1 - q_list[t]) if t < R - 1 else s
        L_exit = L_exit + (w_t * ce_list[t]).mean() * 1.0 \
            if False else L_exit + w_t.mean() * ce_list[t]
        E = E + s
        if t < R - 1:
            s = s * q_list[t]
    L_aux = torch.stack(ce_list).mean()
    loss = L_exit + alpha * L_aux + lam * E.mean()
    return loss, {'exit_ce': float(L_exit.detach()),
                  'aux_ce': float(L_aux.detach()),
                  'E_rounds': float(E.mean().detach())}


@torch.no_grad()
def full_depth_eval(model, ds, bs, dev, max_len, r):
    model.eval()
    correct = n = 0
    order = list(range(len(ds)))
    for i in range(0, len(order), bs):
        b = to_dev(ds.collate(order[i:i + bs]), dev)
        out = model(b['ids'], b['pad_mask'], b['cand_pos'], b['cand_mask'],
                    n_rounds=r)
        logits = out['final_logits'].masked_fill(~b['cand_mask'], -1e9)
        correct += int((logits.argmax(1) == b['label']).sum())
        n += len(order[i:i + bs])
    model.train()
    return correct / n


@torch.no_grad()
def hard_exit_eval(model, ds, bs, dev, tau, r_min, r_max):
    """逐样本硬退出：返回精度、平均轮数、按 hop 轮数、退出直方图、批最大轮。"""
    model.eval()
    order = list(range(len(ds)))
    recs = []
    batch_max_rounds = []
    for i in range(0, len(order), bs):
        b = to_dev(ds.collate(order[i:i + bs]), dev)
        ids, pad_mask = b['ids'], b['pad_mask']
        cand_pos, cand_mask, label, hop = (b['cand_pos'], b['cand_mask'],
                                           b['label'], b['hop'])
        B, S = ids.shape[0], ids.shape[1]
        if ids.dim() == 3:
            x = model.tok_emb(ids).sum(2)
        else:
            x = model.tok_emb(ids)
        x = (x + model.pe[:, :S]) * pad_mask.unsqueeze(-1)
        h0 = model.ln_loop(model._run(x, model.prelude, pad_mask))
        h_raw = h0
        exited = torch.zeros(B, dtype=torch.bool, device=dev)
        exit_round = torch.full((B,), -1, dtype=torch.long, device=dev)
        h_exit = h0.clone()
        prev = None
        for t in range(r_max):
            h_raw = model._run(h_raw + h0, model.core, pad_mask)
            h_read = model.ln_loop(h_raw)
            logits_t = model._logits(h_read, cand_pos, cand_mask)
            probs = torch.softmax(logits_t.masked_fill(~cand_mask, -1e9),
                                  dim=1)
            feats = model._conf_feats(probs, cand_mask, prev)
            pooled = M.masked_mean(h_read, pad_mask)
            q = torch.sigmoid(model.exit_mlp(torch.cat(
                [pooled + model.round_emb.weight[min(t, model.round_emb.num_embeddings - 1)],
                 feats], dim=1))).squeeze(1)
            stop = (1 - q) >= tau
            forced = torch.tensor(t + 1 >= r_max, device=dev)
            do_exit = (~exited) & ((stop & torch.tensor(
                t + 1 >= r_min, device=dev)) | forced)
            if do_exit.any():
                idx = do_exit.nonzero(as_tuple=True)[0]
                exit_round[idx] = t + 1
                h_exit[idx] = h_read[idx]
            exited = exited | do_exit
            prev = probs
            if bool(exited.all()):
                break
        batch_max_rounds.append(int(exit_round.max()))
        final_h = model._run(h_exit, model.coda, pad_mask)
        final_logits = model._logits(final_h, cand_pos, cand_mask)
        pred = final_logits.masked_fill(~cand_mask, -1e9).argmax(1)
        for j in range(B):
            recs.append({'round': int(exit_round[j]),
                         'ok': int(pred[j] == label[j]),
                         'hop': int(hop[j])})
    model.train()
    rounds = [r['round'] for r in recs]
    acc = sum(r['ok'] for r in recs) / len(recs)
    by_hop = {}
    for r in recs:
        by_hop.setdefault(r['hop'], []).append((r['round'], r['ok']))
    hop_stats = {str(h): {'avg_round': round(sum(x for x, _ in v) / len(v), 3),
                          'acc': round(sum(ok for _, ok in v) / len(v), 4),
                          'n': len(v)} for h, v in sorted(by_hop.items())}
    hist = {}
    for r in rounds:
        hist[r] = hist.get(r, 0) + 1
    return {'tau': tau, 'acc': round(acc, 4), 'mean_rounds':
            round(sum(rounds) / len(rounds), 3),
            'by_hop': hop_stats, 'exit_hist': dict(sorted(hist.items())),
            'batch_max_rounds': batch_max_rounds,
            'per_sample_rounds': recs[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--alpha', type=float, default=0.3)
    ap.add_argument('--lam-max', type=float, default=0.02)
    ap.add_argument('--lam-start-frac', type=float, default=0.3)
    ap.add_argument('--lam-full-frac', type=float, default=0.7)
    ap.add_argument('--r-max', type=int, default=6)
    ap.add_argument('--r-min', type=int, default=1)
    ap.add_argument('--dyn-min', type=int, default=1)
    ap.add_argument('--taus', default='0.3,0.5,0.7,0.9')
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--eval-bs', type=int, default=256)
    ap.add_argument('--data-dir', default='data/ldm_mvt')
    ap.add_argument('--out', default=str(OUT_DEFAULT))
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    M.DATA_DIR = Path(args.data_dir)

    model = M.build_model('loop', n_rounds=args.r_max).to(dev)
    tr = M.EncodedDataset(M.load_jsonl(M.DATA_DIR / 'train.jsonl'),
                          max_len=args.max_len, fused=True)
    dv = M.EncodedDataset(M.load_jsonl(M.DATA_DIR / 'dev_random.jsonl'),
                          max_len=args.max_len, fused=True)
    print(json.dumps({'train': len(tr), 'dev': len(dv),
                      'params': model.param_report()['total'],
                      'device': str(dev)}), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.1, betas=(0.9, 0.95))
    warm = max(1, args.steps // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warm))
    rng = random.Random(args.seed)
    order = list(range(len(tr)))
    rng.shuffle(order)
    batches = [order[i:i + args.bs] for i in range(0, len(order), args.bs)]

    hist = []
    step = 0
    t0 = time.perf_counter()
    model.train()
    while step < args.steps:
        rng.shuffle(batches)
        for idx in batches:
            if step >= args.steps:
                break
            R = rng.randint(args.dyn_min, args.r_max)
            b = to_dev(tr.collate(idx), dev)
            out = model(b['ids'], b['pad_mask'], b['cand_pos'],
                        b['cand_mask'], n_rounds=R)
            if step >= args.lam_start_frac * args.steps:
                span = max(1, (args.lam_full_frac - args.lam_start_frac)
                           * args.steps)
                lam = args.lam_max * min(1.0, (step - args.lam_start_frac
                                               * args.steps) / span)
            else:
                lam = 0.0
            loss, parts = soft_exit_loss(out, b['cand_mask'], b['label'],
                                         args.alpha, lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 1000 == 0 or step == 1:
                rec = {'step': step, 'loss': round(float(loss.detach()), 4),
                       'lam': round(lam, 5), 'grad': round(float(gn), 3),
                       **{k: round(v, 4) for k, v in parts.items()},
                       'sec': round(time.perf_counter() - t0, 1)}
                hist.append(rec)
                print(json.dumps(rec), flush=True)

    full_acc = full_depth_eval(model, dv, args.eval_bs, dev,
                               args.max_len, args.r_max)
    tau_results = [hard_exit_eval(model, dv, args.eval_bs, dev, float(t),
                                  args.r_min, args.r_max)
                   for t in args.taus.split(',')]
    payload = {'args': vars(args), 'hist': hist,
               'full_depth_acc': round(full_acc, 4),
               'tau_sweep': tau_results,
               'prereg_pass': {'quality': None, 'hop_monotone': None,
                               'saving': None},
               'note': '判读：质量≥full-2pp；hop3 轮数>hop2+0.15；'
                       'mean_rounds<r_max'}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False,
                                         indent=2) + '\n', encoding='utf-8')
    torch.save(model.state_dict(), str(Path(args.out)).replace(
        '.json', '_ckpt.pt'))
    print(json.dumps({'done': args.out, 'full_acc': payload['full_depth_acc'],
                      'taus': [(r['tau'], r['acc'], r['mean_rounds'])
                               for r in tau_results]}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
