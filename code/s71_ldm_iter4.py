"""s71: 第四次迭代——混批 + 批内变长 R + 流式大样本 + Fan 课程（§18.2 预注册）。

修正 §18.1 钉死的三个混杂：
  ① 混批 per-sample exact-T：批内各 hop 混合，每样本轮数 R_i = hop_i−1，
     rounds_mask 冻结已停样本（s63 已支持）；
  ② 流式无限样本防记忆（沿用 s69 思路，MVT-5 深断链生成器现造）；
  ③ Fan 课程：hop 上限从 2 逐级升到 6（每级 1/5 训练步）。

损失：各样本在自己终轮的候选 CE（per-sample final readout）。
评测：固定 r 扫描（Fan 签名）+ by_hop + 外推 hop7-8。
预注册判读沿用 §18（承重成立 = dev_r5≥0.85 且随 r 单调升；真证伪条件同）。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
_hop_r = {}

import s63_ldm_model as M
from s66_ldm_mvt_gen import gen_example

ROOT = Path(__file__).parent
DEV_DIR = ROOT / 'data' / 'ldm_mvt5'


def _accel():
    """返回当前加速器模块（torch.cuda 或 torch.npu）。

    昇腾（910B）上用 torch_npu，它会注册 `torch.npu` 且让
    `torch.cuda.is_available()` 返回 True；这里显式优先用 npu，
    使 RNG state 的保存/恢复落在正确的设备后端上。
    """
    if hasattr(torch, 'npu') and getattr(torch.npu, 'is_available',
                                          lambda: False)():
        return torch.npu
    if torch.cuda.is_available():
        return torch.cuda
    return None


def _rng_state():
    acc = _accel()
    return acc.get_rng_state() if acc is not None else None


def _set_rng_state(st):
    acc = _accel()
    if acc is None or st is None:
        return
    acc.set_rng_state(st.cpu() if torch.is_tensor(st) else st)


def _pick_device() -> torch.device:
    """设备选择：优先 NPU（昇腾），其次 CUDA，最后 CPU。"""
    if hasattr(torch, 'npu') and getattr(torch.npu, 'is_available',
                                          lambda: False)():
        return torch.device('npu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# 任务生成器（--task mvtd 时切换到无捷径对偶任务）
_TASK = 'mvt5'


def _gen(rng, split, hop):
    if _TASK == 'mvtd':
        import s84_dual_task as D
        return D.gen_mvtd(rng, split, hop)
    return gen_example(rng, split, hop, comp_mode='deep')


def make_batch(rng: random.Random, bs: int, hop_cap: int, dev_: torch.device,
               hop_list=None, use_adj: bool = False):
    rows = []
    while len(rows) < bs:
        hop = (rng.choice(hop_list) if hop_list
               else rng.randint(2, hop_cap))
        ex = _gen(rng, 'train', hop)
        if ex is not None:
            rows.append(ex)
    ds = M.EncodedDataset(rows, max_len=256, fused=True)
    b = {k: (v.to(dev_) if torch.is_tensor(v) else v)
         for k, v in ds.collate(list(range(bs))).items()}
    if use_adj:
        # 邻接掩码（P1-2）：事实级一跳图，供逐轮"≤t 跳"约束使用
        import s88_adj_task as A
        adj, fm, _dist = A.build_adj_batch([r['facts'] for r in rows],
                                          [r['e0'] for r in rows])
        b['adj'] = adj.to(dev_)
        b['fact_mask'] = fm.to(dev_)
        b['dist'] = _dist.to(dev_)
    return b


def make_batch_range(rng: random.Random, bs: int, lo: int, hi: int,
                     dev_: torch.device):
    """hop ∈ [lo, hi] 均匀采样（课程回采用）。"""
    rows = []
    while len(rows) < bs:
        hop = rng.randint(lo, hi)
        ex = _gen(rng, 'train', hop)
        if ex is not None:
            rows.append(ex)
    ds = M.EncodedDataset(rows, max_len=256, fused=True)
    return {k: (v.to(dev_) if torch.is_tensor(v) else v)
            for k, v in ds.collate(list(range(bs))).items()}


def readout_at(model, b, R_list, r_max: int):
    """per-sample 终轮读出：rounds_mask (B, r_max)，样本 i 活跃到 R_list[i]。"""
    rm = torch.zeros(b['ids'].size(0), r_max, dtype=torch.bool,
                     device=b['ids'].device)
    for i, r in enumerate(R_list):
        rm[i, :r] = True
    out = model(b['ids'], b['pad_mask'], b['cand_pos'], b['cand_mask'],
                n_rounds=r_max, rounds_mask=rm)
    return out['final_logits'], rm


def ponder_loss(model, b, r_max: int, lam_prior: float, beta_kl: float):
    """§19.7 PonderNet 完整损失（预注册）。

    与 V3c joint-exit 的三点区别：
      ① 无 rounds_mask——全样本跑满 r_max 轮，过冲毁答案（V4 实测），
        "永远继续"从此有真代价，q 获得真实停轮梯度；
      ② 逐轮读出 = coda(h_read_t)，与硬退出推理读出严格同路径
        （V3c 用 pre-coda probe CE，训练/推理读出语义错配）；
      ③ 损失 = Σ_t p_t·CE_t（per-sample 加权，非批均值×均值权重）
        + β·KL(p_t ‖ 截断几何先验 λ_p)，λ_p=1−1/E[T]，E[T]=3。
    """
    out = model(b['ids'], b['pad_mask'], b['cand_pos'], b['cand_mask'],
                n_rounds=r_max, return_round_states=True)
    ce_rows, qs = [], []
    for rd in out['rounds']:
        h_c = model._run(rd['h_read'], model.coda, b['pad_mask'])
        lg = model._logits(h_c, b['cand_pos'], b['cand_mask'],
                           cand_emb=out['cand_emb'], seq=h_c,
                           pad_mask=b['pad_mask'])
        lg = lg.masked_fill(~b['cand_mask'], -1e9)
        ce_rows.append(F.cross_entropy(lg, b['label'], reduction='none'))
        qs.append(rd['q'])
    CE = torch.stack(ce_rows)                       # (T, B)
    Q = torch.stack(qs)                             # (T, B)
    T, B = CE.shape
    dev = CE.device
    p_rows, s = [], torch.ones(B, device=dev)
    for t in range(T):
        if t < T - 1:
            p_rows.append(s * (1 - Q[t]))
            s = s * Q[t]
        else:                                       # 末轮吸收剩余质量
            p_rows.append(s)
    P = torch.stack(p_rows)                         # (T, B)，每列和为 1
    task = (P * CE).sum(0).mean()
    prior = torch.tensor([(1 - lam_prior) * lam_prior ** t
                          for t in range(T - 1)]
                         + [lam_prior ** (T - 1)], device=dev).log()
    kl = (P * (P.clamp_min(1e-9).log() - prior.unsqueeze(1))).sum(0).mean()
    e_halt = (P * torch.arange(1, T + 1, device=dev)
              .unsqueeze(1)).sum(0).mean()
    loss = task + beta_kl * kl
    return loss, {'task': float(task.detach()),
                  'kl': float(kl.detach()),
                  'e_halt': float(e_halt.detach()),
                  'q_mean_first': float(Q[0].mean().detach()),
                  'q_mean_last': float(Q[-1].mean().detach())}


def ouro_loss(model, b, r_max: int, beta_ent: float, step_sup: bool = False,
              step_weight: float = 0.5, adj_bias_list=None):
    """Ouro 风格损失（arXiv:2510.25741，第一阶段）。

    与 ponder_loss 的两点关键差异：
      ① **均匀先验**（而非几何先验）：Ouro 用 uniform prior 配熵正则，
         鼓励模型探索所有深度、不塌到某个固定轮数。我们此前的几何先验
         `(1-λ)λ^t` 明确鼓励早停，可能正是 r≥3 无增益的原因。
      ② **额外熵奖励**：-β·H(p)，主动最大化退出分布的熵。

    损失 = Σ_t p_t·CE_t  +  β_kl·KL(p ‖ U)  −  β_ent·H(p)
    其中 p 由退出概率 q_t 生成（末轮吸收剩余质量），KL 到均匀分布
    等价于 −H(p) + log T，故两项合并为 −(β_kl+β_ent)·H(p)。

    ≥3 轮才有意义的任务上，均匀先验防止"全塌到第 2 轮"（MVT-D v2 的现状）。
    """
    out = model(b['ids'], b['pad_mask'], b['cand_pos'], b['cand_mask'],
                n_rounds=r_max, return_round_states=True,
                adj_bias_list=adj_bias_list)
    ce_rows, qs, sl_rows = [], [], []
    for rd in out['rounds']:
        h_c = model._run(rd['h_read'], model.coda, b['pad_mask'])
        lg = model._logits(h_c, b['cand_pos'], b['cand_mask'],
                           cand_emb=out['cand_emb'], seq=h_c,
                           pad_mask=b['pad_mask'])
        lg = lg.masked_fill(~b['cand_mask'], -1e9)
        ce_rows.append(F.cross_entropy(lg, b['label'], reduction='none'))
        qs.append(rd['q'])
        if rd.get('step_logits') is not None:
            sl_rows.append(rd['step_logits'])
    CE = torch.stack(ce_rows)
    Q = torch.stack(qs)
    T, B = CE.shape
    dev = CE.device
    p_rows, s = [], torch.ones(B, device=dev)
    for t in range(T):
        if t < T - 1:
            p_rows.append(s * (1 - Q[t]))
            s = s * Q[t]
        else:
            p_rows.append(s)
    P = torch.stack(p_rows)                       # (T,B) 列和为 1
    task = (P * CE).sum(0).mean()
    # 熵（对退出分布）；均匀先验下 KL(p||U) = log T - H(p)
    ent = -(P * P.clamp_min(1e-9).log()).sum(0).mean()
    loss = task + beta_ent * (torch.log(torch.tensor(float(T), device=dev))
                              - ent)
    if step_sup and sl_rows:
        se = b['step_ent']
        SL = torch.stack(sl_rows)
        Tn, Bn, Vn = SL.shape
        Tse = se.size(1)
        Tuse = min(Tn, Tse)
        if Tuse > 0:
            SLu = SL[:Tuse]
            tgt = se[:, :Tuse].t().contiguous()
            valid = tgt >= 0
            if bool(valid.any()):
                ls = F.cross_entropy(SLu.reshape(-1, Vn),
                                     tgt.reshape(-1).clamp_min(0),
                                     reduction='none').view(Tuse, Bn)
                loss = loss + step_weight * ls[valid].mean()
    e_halt = (P * torch.arange(1, T + 1, device=dev).unsqueeze(1)).sum(0).mean()
    return loss, {'task': float(task.detach()),
                  'ent': float(ent.detach()),
                  'e_halt': float(e_halt.detach()),
                  'q_mean_first': float(Q[0].mean().detach()),
                  'q_mean_last': float(Q[-1].mean().detach())}


@torch.no_grad()
def hard_exit_eval_v2(model, ds, bs, dev, tau, r_min, r_max,
                      feasible=None):
    """§19.7 硬退出评测：直接用 model.forward（含 round_emb/bottleneck，
    构造上与训练同路径）；逐样本退出轮 = 首个 t≥r_min 且 1-q_t≥τ，
    否则 r_max 强制；答案 = 各样本退出轮 h_read 经一次 coda 的读出。"""
    model.eval()
    recs = []
    order = list(range(len(ds)))
    for i in range(0, len(order), bs):
        b = {k: (v.to(dev) if torch.is_tensor(v) else v)
             for k, v in ds.collate(order[i:i + bs]).items()}
        out = model(b['ids'], b['pad_mask'], b['cand_pos'], b['cand_mask'],
                    n_rounds=r_max, return_round_states=True)
        rounds = out['rounds']
        B = b['label'].size(0)
        Q = torch.stack([r['q'] for r in rounds])           # (T, B)
        stop = (1 - Q) >= tau
        stop[:max(0, r_min - 1), :] = False                 # 轮号 < r_min 不许停
        idxm = torch.arange(Q.size(0), device=dev).unsqueeze(1)
        # 首个停轮索引（确定性：非停位填末轮索引后取 min）
        t_score = torch.where(stop, idxm,
                              torch.full_like(idxm, Q.size(0) - 1))
        first_t = t_score.min(0).values
        exit_r = (first_t + 1).clamp_max(r_max)
        # 各样本取自己退出轮的 h_read，拼成一个批次跑一次 coda
        h_g = torch.stack([rounds[exit_r[j] - 1]['h_read'][j]
                           for j in range(B)])
        h_f = model._run(h_g, model.coda, b['pad_mask'])
        lg = model._logits(h_f, b['cand_pos'], b['cand_mask'],
                           cand_emb=out['cand_emb'], seq=h_f,
                           pad_mask=b['pad_mask'])
        pred = lg.masked_fill(~b['cand_mask'], -1e9).argmax(1)
        for j in range(B):
            recs.append({'round': int(exit_r[j]), 'ok': int(pred[j] == b['label'][j]),
                         'hop': int(b['hop'][j])})
    model.train()
    n = len(recs)
    acc = sum(r['ok'] for r in recs) / n
    rounds_all = [r['round'] for r in recs]
    hist = {}
    for r in rounds_all:
        hist[r] = hist.get(r, 0) + 1
    by_hop = {}
    for r in recs:
        by_hop.setdefault(r['hop'], []).append(r)
    hop_stats = {str(h): {'avg_round': round(sum(x['round'] for x in v) / len(v), 3),
                          'acc': round(sum(x['ok'] for x in v) / len(v), 4),
                          'n': len(v)} for h, v in sorted(by_hop.items())}
    match = sum(r['round'] == min(max(1, r['hop'] - 1), r_max) for r in recs) / n
    if feasible:      # A2：对照"该 hop 首个固定 r acc≥0.95 的轮数"（无则不计）
        sub = [r for r in recs if r['hop'] in feasible]
        match_feas = (sum(r['round'] == feasible[r['hop']] for r in sub)
                      / len(sub)) if sub else None
    else:
        match_feas = None
    top = max(hist.values())
    return {'tau': tau, 'acc': round(acc, 4),
            'mean_rounds': round(sum(rounds_all) / n, 3),
            'by_hop': hop_stats, 'exit_hist': dict(sorted(hist.items())),
            'stop_match_rate': round(match, 4),
            'match_feasible_rate': (round(match_feas, 4)
                                    if match_feas is not None else None),
            'collapse': bool(top / n >= 0.95)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--r-max', type=int, default=8)
    ap.add_argument('--pos', default='sin', choices=['sin', 'none'])
    ap.add_argument('--variant', default='loop', choices=['loop', 'dense'])
    ap.add_argument('--dense-mid', type=int, default=1)
    ap.add_argument('--bottleneck-dim', type=int, default=0)
    ap.add_argument('--curriculum', default='2,3,4,5,6')
    ap.add_argument('--adj-mask', action='store_true',
                    help='邻接掩码（P1-2）：第 t 轮只允许事实看到 ≤t 跳，'
                         '强制每轮只多传播一跳')
    ap.add_argument('--curriculum-step', type=int, default=0,
                    help='课程阶段按绝对步数推进：每 N 步升一级。'
                         '0 = 旧行为（按 step/total 比例，长短训练不可比）')
    ap.add_argument('--warmup', type=int, default=0,
                    help='LR warmup 步数（0 = 旧行为 steps//50）。'
                         'warmup 长度与总步数必须可独立设置，否则'
                         '“步数效应”与“warmup 效应”无法分离（P19 踩坑）')
    ap.add_argument('--joint-exit', action='store_true',
                    help='退出头联训：骨干+exit_mlp 共训，soft-exit 加权 CE')
    ap.add_argument('--ponder', action='store_true',
                    help='§19.7 PonderNet 完整损失：无冻结跑满 r_max，'
                         'Σ p_t·CE_t(coda 读出) + β·KL(几何先验)')
    ap.add_argument('--ouro', action='store_true',
                    help='Ouro 风格损失（arXiv:2510.25741）：均匀先验 + '
                         '熵奖励，鼓励探索所有深度而非塌到固定轮数')
    ap.add_argument('--beta-ent', type=float, default=0.05,
                    help='Ouro 熵项系数（KL 到均匀分布 = logT - H(p)）')
    ap.add_argument('--lam-prior', type=float, default=2.0 / 3.0,
                    help='几何先验 λ_p，均值≈1/(1-λ)=E[T]=3')
    ap.add_argument('--beta-kl', type=float, default=0.1)
    ap.add_argument('--taus', default='0.3,0.5,0.7,0.9')
    ap.add_argument('--lam-max', type=float, default=0.002)
    ap.add_argument('--eval-every', type=int, default=5000)
    ap.add_argument('--early-stop', action='store_true',
                    help='课程覆盖的全部 hop 在 dev_r8 上均 ≥0.95 时停止'
                         '（防过冲遗忘，§19.8 seed41 实测）')
    ap.add_argument('--ema-decay', type=float, default=0.0,
                    help='权重 EMA 衰减（0=关闭；文献推荐 0.92–0.999）。'
                         '评测/存档用 EMA 权重，对抗课程遗忘')
    ap.add_argument('--width', type=int, default=192,
                    help='H-008：模型宽度 d（192=基线，384=加宽）')
    ap.add_argument('--n-core', type=int, default=1,
                    help='每轮应用的共享块层数 L（1=基线；Kohli 用 4）')
    ap.add_argument('--curriculum-mix', type=float, default=0.0,
                    help='H-010：课程平滑混合概率（0=硬阶梯，0.3=渐进过渡）')
    ap.add_argument('--n-heads-read', type=int, default=0,
                    help='H-009：per-round 独立读出头数量（0=共享单一 score）')
    ap.add_argument('--head-type', default='linear',
                    choices=['linear', 'bilinear', 'mlp', 'xattn'],
                    help='决策模块形态：linear=现状(193 参)；bilinear=与候选输入'
                         '表示做双线性；mlp=测容量；xattn=候选作 query 检索全序列')
    ap.add_argument('--film-rounds', action='store_true',
                    help='方案A：FiLM 轮次调制（per-dim scale+shift）')
    ap.add_argument('--persist-sup', action='store_true',
                    help='方案C：多轮保持监督（第 R 轮起每轮都要答对）')
    ap.add_argument('--step-sup', action='store_true',
                    help='中间路径监督：第 t 轮预测下一跳实体（scratchpad）')
    ap.add_argument('--step-weight', type=float, default=0.5,
                    help='路径监督损失权重')
    ap.add_argument('--hop-list', default='',
                    help='显式 hop 采样列表（如 2,4,6,8，用于深度插值实验）')
    ap.add_argument('--sup-cap', type=int, default=0,
                    help='persist 监督的轮数上限（0=不限制）。设为训练深度后，'
                         '超出部分不再被训成"保持"，留给推理时计算')
    ap.add_argument('--dyn-r', action='store_true',
                    help='动态轮数：每批从 [r_min_train, r_max] 均匀采样展开轮数'
                         '（Kohli et al. 2026 发现动态递归外推最强）')
    ap.add_argument('--warm-noise', type=float, default=0.0,
                    help='状态唤醒：在第 7 轮起给残差加高斯噪声（std），'
                         '扩大后期轮次的状态覆盖（Ruiz & Gu 2025）')
    ap.add_argument('--cap-reach', type=int, default=0,
                    help='课程回采深度：cap=N 时采样 hop ∈ [N-reach, N]，'
                         '使难度过渡平滑（0=关闭，均匀 2..cap）')
    ap.add_argument('--resume', action='store_true',
                    help='从 --out 对应的 _ckpt.pt 续跑（存 optimizer/sched/'
                         'rng/step/hist；课程进度与 LR 调度按总步数重算）')
    ap.add_argument('--init-step', type=int, default=0,
                    help='warm-start 模式下已有权重对应的训练步数')
    ap.add_argument('--out', default='results/mvt5_iter4.json')
    ap.add_argument('--data-dir', default='',
                    help='评测集目录（默认 data/ldm_mvt5；MVT-D 用 data/ldm_mvtd）')
    ap.add_argument('--task', default='mvt5', choices=['mvt5', 'mvtd'],
                    help='训练任务：mvt5（原任务，含下游链长捷径）或 '
                         'mvtd（无捷径对偶任务）')
    args = ap.parse_args()
    global _TASK
    _TASK = args.task

    dev_ = _pick_device()      # NPU（昇腾）/ CUDA / CPU
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    if args.variant == 'dense':
        model = M.build_model('dense', n_dense_mid=args.dense_mid,
                              pos=args.pos).to(dev_)
    else:
        _heads = args.width // 48 if args.width > 192 else 4
        model = M.build_model('loop', n_rounds=args.r_max,
                              n_core=args.n_core,
                              round_in_core=True, pos=args.pos,
                              bottleneck_dim=args.bottleneck_dim,
                              d=args.width, heads=_heads,
                              ffn=args.width * 4,
                              n_heads_read=args.n_heads_read,
                              film_rounds=args.film_rounds,
                              head_type=args.head_type,
                              step_sup=args.step_sup).to(dev_)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.1, betas=(0.9, 0.95))
    # EMA（文献 arXiv:2506.00288 §5.2）：限制参数漂移，对抗课程推进时的
    # 灾难性遗忘（实测 cap 升档后 hop4 从 0.915 崩到 0.1）。评测与存档
    # 用 EMA 权重，训练仍走原始权重。
    ema = None
    if args.ema_decay > 0:
        ema = {k: v.detach().clone()
               for k, v in model.state_dict().items()}
    def ema_update():
        if ema is None:
            return
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(args.ema_decay).add_(
                        v.detach(), alpha=1 - args.ema_decay)
                else:
                    ema[k].copy_(v)
    def apply_ema(use: bool):
        if ema is None:
            return None
        if use:
            backup = {k: v.detach().clone()
                      for k, v in model.state_dict().items()}
            model.load_state_dict(ema, strict=True)
            return backup
        return None
    # warmup 原为 steps//50，即"总步数决定 warmup 长度"——P19 实测：同 seed
    # 同配置，48k 基线 s53 的 h6=0.705，而 --steps 64000 的 run 在 48000 步处
    # h6=0.900。唯一差异是 warmup 960 vs 1280，故 warmup 必须可独立设置，
    # 否则"训练步数"与"warmup"两因素被绑死、无法归因。
    warm = args.warmup if args.warmup > 0 else max(1, args.steps // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warm))

    from s65_ldm_gpu_smoke import evaluate as _ev
    # 评测集目录可配（MVT-D 等对偶任务用 --data-dir 指定），默认 ldm_mvt5
    _dd = Path(args.data_dir) if getattr(args, 'data_dir', '') else DEV_DIR
    _cf = _dd / 'confirm_depth.jsonl'
    if not _cf.exists():
        _cf = _dd / 'dev_ood.jsonl'      # MVT-D 用 dev_ood.jsonl
    dev_ds = M.EncodedDataset(
        [json.loads(l) for l in (_dd / 'dev_random.jsonl')
         .read_text(encoding='utf-8').splitlines() if l.strip()],
        max_len=256, fused=True)
    cf_ds = M.EncodedDataset(
        [json.loads(l) for l in _cf
         .read_text(encoding='utf-8').splitlines() if l.strip()],
        max_len=256, fused=True)

    stages = [int(h) for h in args.curriculum.split(',')]
    hist = []
    evals_hist = []          # 累积 eval 记录（面板曲线 + 永久档判据）
    milestones = {}          # 转折点永久档索引
    start_step = 1
    ckpt_path = Path(args.out.replace('.json', '_ckpt.pt'))
    state_path = Path(args.out.replace('.json', '_state.pt'))
    # 独占锁：同一 --out 只允许一条训练进程（已两次因重复启动互相覆盖
    # ckpt/state/json 造成不可信读数）。进程退出后锁自动释放。
    lock_path = Path(args.out.replace('.json', '.lock'))
    lock_fh = open(lock_path, 'w')
    try:
        try:
            import msvcrt                      # Windows
            msvcrt.locking(lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
        except ImportError:
            import fcntl                       # Linux
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f'[LOCK] {args.out} 已有训练进程在写（{lock_path}）；'
                         f'先确认并用唯一 --out')
    if args.resume:
        if state_path.exists():
            st = torch.load(state_path, map_location=dev_,
                            weights_only=False)
            model.load_state_dict(st['model'])
            opt.load_state_dict(st['opt'])
            sched.load_state_dict(st['sched'])
            tr = st['torch_rng']
            torch.set_rng_state(tr.cpu() if torch.is_tensor(tr) else tr)
            _set_rng_state(st.get('cuda_rng'))   # 设备无关（CUDA / NPU）
            rng.setstate(st['py_rng'])
            hist = st.get('hist', [])
            evals_hist = st.get('evals_hist', [])
            milestones = st.get('milestones', {})
            start_step = st['step'] + 1
        elif ckpt_path.exists():
            # 旧版 ckpt 只有权重（V5b 误杀现场）：从 --init-step 续，
            # 优化器/LR/RNG 重建——等价"热启动"，非严格续跑，日志标注
            model.load_state_dict(torch.load(ckpt_path, map_location=dev_))
            start_step = args.init_step + 1
            hist = [{'step': args.init_step, 'note': 'warm-start from ckpt'}]
        else:
            raise SystemExit(f'--resume 找不到 {state_path} 或 {ckpt_path}')
        # 续跑后 LR 调度按绝对步数重对齐（否则 warmup 长度随总步数变化，
        # 会把 lr 冻在近零——warm-start 实测踩到）
        opt.param_groups[0]['lr'] = args.lr * min(1.0, start_step / warm)
        sched.last_epoch = start_step - 1
        print(json.dumps({'resumed_from': start_step - 1,
                          'to': args.steps,
                          'lr': round(opt.param_groups[0]['lr'], 6),
                          'mode': 'state' if state_path.exists()
                          else 'warm-start'}), flush=True)
    t0 = time.perf_counter()
    early_stop_rec = None      # 早停记录（最终写入时必须保留，勿被覆盖）
    model.train()
    _reach = getattr(args, 'cap_reach', 0)   # 回采深度：cap=N 时最小 hop
    # 课程节奏与总步数解耦（2026-10-27 修）：
    # 原实现按 step/total_steps 分配阶段，导致 48k 与 24k 在相同 step 处
    # 面对不同难度，长短训练不可比。现改用绝对步数阈值。
    # --curriculum-step N 表示"每 N 步升一级"；默认 0 = 沿用旧行为（按比例）。
    _cstep = int(getattr(args, 'curriculum_step', 0) or 0)

    def _cap_at(s: int) -> int:
        if _cstep > 0:
            _i = min(len(stages) - 1, s // _cstep)
            return stages[_i]
        _i = min(len(stages) - 1, s * len(stages) // max(1, args.steps))
        return stages[_i]

    for step in range(start_step, args.steps + 1):
        _si = (min(len(stages) - 1, step // _cstep) if _cstep > 0
               else min(len(stages) - 1,
                        step * len(stages) // max(1, args.steps)))
        if args.curriculum_mix > 0 and _si < len(stages) - 1 \
                and rng.random() < args.curriculum_mix:
            cap = stages[_si + 1]
        else:
            cap = stages[_si]
        _hl = ([int(x) for x in args.hop_list.split(',')]
               if args.hop_list else None)
        b = make_batch(rng, args.bs, cap, dev_, _hl,
                       use_adj=getattr(args, 'adj_mask', False))
        _bias_list = None
        if getattr(args, 'adj_mask', False) and 'adj' in b:
            import s88_adj_task as _A
            _S = b['ids'].size(1)
            _bias_list = [_A.adj_bias_for_round(b['adj'], b['fact_mask'], _t,
                                                b.get('dist'))[:, :, :_S, :_S]
                          for _t in range(1, args.r_max + 1)]
        if _reach > 0 and cap - _reach >= 2:
            # 回采：cap=N 时按 --cap-reach 回退下界，使难度跨度平滑。
            # 直击"hop5 是冷启动门槛"——让 hop4→5 之间有连续过渡带，
            # 而非从纯 hop≤4 直接跳到纯 hop≤5。
            _lo = max(2, cap - _reach)
            b = make_batch_range(rng, args.bs, _lo, cap, dev_)
        R_list = [min(max(1, h - 1), args.r_max) for h in b['hop'].tolist()]
        parts = None
        if args.variant == 'dense':
            out_d = model(b['ids'], b['pad_mask'], b['cand_pos'],
                          b['cand_mask'], return_round_states=args.step_sup)
            logits = out_d['final_logits']
            lg = logits.masked_fill(~b['cand_mask'], -1e9)
            loss = F.cross_entropy(lg, b['label'])
            # 公平性：dense 臂同样可用 step-sup（路径监督不依赖循环，
            # 只是 dense 只有一次前向，用其唯一一轮的 step_logits）。
            if args.step_sup:
                sl = out_d.get('step_logits')
                se = b.get('step_ent')
                if sl is not None and se is not None and se.numel():
                    SL = sl.unsqueeze(0)                  # (1,B,V)
                    tgt = se[:, :1].t().contiguous()      # (1,B)
                    valid = tgt >= 0
                    if bool(valid.any()):
                        ls = F.cross_entropy(
                            SL.reshape(-1, SL.size(-1)),
                            tgt.reshape(-1).clamp_min(0),
                            reduction='none').view(1, SL.size(1))
                        loss = loss + args.step_weight * ls[valid].mean()
            parts = None
        elif args.persist_sup:
            # 方案C：多轮保持监督。exact-T 只在第 R 轮监督，模型从未见过
            # "算完之后继续跑"的状态——评测多跑几轮即崩（过冲毁答案）。
            # 这里从第 R 轮起每一轮都要求答对：教会"算出来之后守住"。
            _Rb = args.r_max
            if args.dyn_r:
                _Rb = rng.randint(max(2, args.r_max // 2), args.r_max)
            out = model(b['ids'], b['pad_mask'], b['cand_pos'],
                        b['cand_mask'], n_rounds=_Rb,
                        warm_noise=args.warm_noise, warm_from=6)
            rounds = out['rounds']
            B = b['label'].size(0)
            ce_rows = [F.cross_entropy(
                r['logits'].masked_fill(~b['cand_mask'], -1e9), b['label'],
                reduction='none') for r in rounds]
            CE = torch.stack(ce_rows)                 # (T, B)
            T = CE.size(0)
            mask = torch.zeros(T, B, device=CE.device)
            pre = torch.zeros(T, B, device=CE.device)
            _cap = args.sup_cap if args.sup_cap > 0 else T
            for i, R in enumerate(R_list):
                mask[max(0, R - 1):min(T, _cap), i] = 1.0   # 第 R 起，至 sup-cap
                pre[:max(0, R - 1), i] = 1.0          # 之前轮次轻压
            loss = (CE * mask).sum() / mask.sum().clamp_min(1)
            if float(pre.sum()) > 0:                  # 防提前偷跑并行捷径
                loss = loss + 0.3 * (CE * pre).sum() / pre.sum()
            # 中间路径监督：第 t 轮应预测 proof[t].obj。
            # 注意 se 的有效列数 Tse 可能 < 轮数 T；按 min(T, Tse) 截断。
            if args.step_sup:
                se = b['step_ent']                        # (B, Tse)
                sl = [r.get('step_logits') for r in rounds]
                sl = [x for x in sl if x is not None]
                if sl and se.numel():
                    SL = torch.stack(sl)                  # (T, B, V)
                    Tn, Bn, Vn = SL.shape
                    Tse = se.size(1)
                    Tuse = min(Tn, Tse)
                    if Tuse > 0:
                        SLu = SL[:Tuse]                   # (Tuse, B, V)
                        tgt = se[:, :Tuse].t().contiguous()  # (Tuse, B)
                        valid = tgt >= 0
                        if bool(valid.any()):
                            ls = F.cross_entropy(
                                SLu.reshape(-1, Vn),
                                tgt.reshape(-1).clamp_min(0),
                                reduction='none').view(Tuse, Bn)
                            loss = loss + args.step_weight * ls[valid].mean()
            parts = {'sup_persist': float(loss.detach())}
        elif args.ponder:
            loss, parts = ponder_loss(model, b, args.r_max,
                                      args.lam_prior, args.beta_kl)
        elif args.ouro:
            loss, parts = ouro_loss(model, b, args.r_max, args.beta_ent,
                                    step_sup=args.step_sup,
                                    step_weight=args.step_weight,
                                    adj_bias_list=_bias_list)
        elif args.joint_exit:
            rm = torch.zeros(b['ids'].size(0), args.r_max, dtype=torch.bool,
                             device=b['ids'].device)
            for i, r in enumerate(R_list):
                rm[i, :r] = True
            out = model(b['ids'], b['pad_mask'], b['cand_pos'],
                        b['cand_mask'], n_rounds=args.r_max, rounds_mask=rm)
            ces = [F.cross_entropy(
                r['logits'].masked_fill(~b['cand_mask'], -1e9), b['label'])
                for r in out['rounds']]
            qs = [r['q'] for r in out['rounds']]
            B = b['label'].size(0)
            sv = torch.ones(B, device=b['ids'].device)
            Lx = torch.zeros((), device=b['ids'].device)
            for t in range(len(ces)):
                w = sv * (1 - qs[t]) if t < len(ces) - 1 else sv
                Lx = Lx + w.mean() * ces[t]
                if t < len(ces) - 1:
                    sv = sv * qs[t]
            loss = Lx
        else:
            logits, _ = readout_at(model, b, R_list, args.r_max)
            lg = logits.masked_fill(~b['cand_mask'], -1e9)
            loss = F.cross_entropy(lg, b['label'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema_update()
        if step % 1000 == 0:
            rec_h = {'step': step, 'cap': cap,
                     'loss': round(float(loss.detach()), 4)}
            if parts is not None:      # §19.7 实时观测：停轮期望/KL/q 端点
                rec_h.update({k: round(v, 4) for k, v in parts.items()})
            hist.append(rec_h)
        if step % args.eval_every == 0 or step == args.steps:
            raw_backup = apply_ema(True)     # 先切到 EMA 权重评测
            ev = {}
            for name, ds in (('dev', dev_ds), ('confirm', cf_ds)):
                for r in (1, 2, 3, 4, 5, 6, 8):
                    ev[f'{name}_r{r}'] = _ev(model, ds, 256, dev_, 256,
                                             n_rounds=r)
            global _hop_r
            _hop_r = {f'dev_r{r}_by_hop': ev[f'dev_r{r}']['by_hop']
                      for r in (1, 2, 3, 4, 5, 6, 8)}
            rec = {'step': step, 'cap': cap,
                   'sec': round(time.perf_counter() - t0, 1),
                   **{k: v['acc'] for k, v in ev.items()},
                   'dev_r5_by_hop': ev['dev_r5']['by_hop'],
                   'confirm_r6_by_hop': ev['confirm_r6']['by_hop'],
                   **_hop_r}
            torch.save(model.state_dict(), ckpt_path)   # EMA 权重（评测口径）
            if raw_backup is not None:
                model.load_state_dict(raw_backup, strict=True)
            torch.save({'model': model.state_dict(),     # 原始权重（续跑口径）
                        'opt': opt.state_dict(), 'sched': sched.state_dict(),
                        'step': step, 'hist': hist,
                        'torch_rng': torch.get_rng_state(),
                        'cuda_rng': _rng_state(),   # 设备无关（CUDA / NPU）
                        'py_rng': rng.getstate(),
                        'evals_hist': evals_hist,
                        'milestones': milestones}, state_path)
            # 里程碑存档用 EMA 权重（评测口径最佳态）
            # 转折点永久档：任一 hop 首次解到 ≥0.95 / ≥4 个 hop 达标
            # ——一旦触发就留档，不随后续覆盖丢失。
            triggers = []
            for h, acc in (rec.get('dev_r8_by_hop') or {}).items():
                key = f'diag_hop{h}'
                if acc >= 0.95 and key not in milestones:
                    milestones[key] = {'step': step, 'by_hop': rec['dev_r8_by_hop']}
                    triggers.append(key)
            diag = sum(1 for h, v in (rec.get('dev_r8_by_hop') or {}).items()
                       if v >= 0.95)
            if diag >= 4 and 'diag_4hops' not in milestones:
                milestones['diag_4hops'] = {'step': step, 'n_hops': diag}
                triggers.append('diag_4hops')
            if triggers:
                perm = Path(args.out.replace(
                    '.json', f'_MILESTONE_s{step}.pt'))
                _b = apply_ema(True)          # 存 EMA 权重
                torch.save({'model': model.state_dict(), 'step': step,
                            'triggers': triggers,
                            'evals': rec, 'milestones': dict(milestones)},
                           perm)
                if _b is not None:
                    model.load_state_dict(_b, strict=True)
                print(json.dumps({'MILESTONE': str(perm),
                                  'step': step,
                                  'triggers': triggers}), flush=True)

            # 早停判据（目标配方，非终评）：dev_r8 by_hop 在课程覆盖的
            # 全部难度级上都 ≥0.95 → 机制已解出，继续跑只会过冲/遗忘
            # （§19.8 seed41 实测 20k 峰 0.916 → 24k 0.786）。停止并存档。
            covered = {h for h in (rec.get('dev_r8_by_hop') or {})
                       if int(h) <= max(stages)}
            solved = {h for h in covered
                      if rec['dev_r8_by_hop'][h] >= 0.95}
            _can_verify = max(stages) <= 6   # dev 集最深 hop6
            if (args.early_stop and not _can_verify and not getattr(
                    args, '_es_warned', False)):
                print(json.dumps({'early_stop_disabled':
                                  'curriculum beyond dev coverage'}),
                      flush=True)
                args._es_warned = True
            if (args.early_stop and _can_verify
                    and covered and covered <= solved):
                stop_path = Path(args.out.replace('.json', '_EARLYSTOP.pt'))
                _b = apply_ema(True)
                torch.save({'model': model.state_dict(), 'step': step,
                            'by_hop': rec['dev_r8_by_hop'],
                            'reason': f'all {len(covered)} covered hops >=0.95'},
                           stop_path)
                if _b is not None:
                    model.load_state_dict(_b, strict=True)
                print(json.dumps({'EARLY_STOP': str(stop_path),
                                  'step': step,
                                  'by_hop': rec['dev_r8_by_hop']}), flush=True)
                evals_hist.append(rec)
                Path(args.out).write_text(json.dumps(
                    {'args': vars(args), 'hist': hist, 'evals': rec,
                     'evals_hist': evals_hist, 'milestones': milestones,
                     'early_stop': {'step': step,
                                    'by_hop': rec['dev_r8_by_hop']}},
                    indent=2) + '\n', encoding='utf-8')
                hist[-1]['early_stop'] = step if hist else None
                early_stop_rec = {'step': step,
                                  'by_hop': rec['dev_r8_by_hop']}
                break
            if hist:     # steps<1000 时 hist 可能为空（冒烟曾触发 IndexError）
                hist[-1].update({k: rec[k] for k in ('sec',)})
            print(json.dumps(rec), flush=True)
            evals_hist.append(rec)
            Path(args.out).write_text(json.dumps(
                {'args': vars(args), 'hist': hist, 'evals': rec,
                 'evals_hist': evals_hist, 'milestones': milestones},
                indent=2) + '\n', encoding='utf-8')

    # ---- §19.7 终评：固定 r 扫描已在 evals 里（oracle=对角线），此处补硬退出
    final = {'args': vars(args), 'hist': hist, 'last_evals': rec,
             'evals': rec, 'early_stop': early_stop_rec,
             'evals_hist': evals_hist, 'milestones': milestones,
             'milestone_files': sorted(
                 p.name for p in Path(args.out).parent.glob(
                     Path(args.out).stem + '_MILESTONE_*.pt'))}
    if args.ponder:
        rs = [1, 2, 3, 4, 5, 6, 8]
        oracle = max(rec[f'dev_r{r}'] for r in rs)   # rec 内已是 acc 浮点
        fixed_r8 = rec['dev_r8']
        # A2 对照系：各 hop 首个固定 r acc≥0.95 的轮数（评测扫描里取）
        feasible = {}
        for h in (rec['dev_r8_by_hop'] or {}):
            hit = [r for r in rs
                   if rec.get(f'dev_r{r}_by_hop', {}).get(h, 0) >= 0.95]
            if hit:
                feasible[int(h)] = min(hit)
        sweep = [hard_exit_eval_v2(model, dev_ds, 256, dev_, float(t), 1,
                                   args.r_max, feasible=feasible)
                 for t in args.taus.split(',')]
        # A1：质量门（acc ≥ oracle−2pp）内取 mean_rounds 最小 τ
        usable = [s for s in sweep if s['acc'] >= oracle - 0.02]
        best = min(usable, key=lambda s: s['mean_rounds']) if usable else None
        final['ponder_eval'] = {
            'oracle_fixed_r': round(oracle, 4),
            'fixed_r8': round(fixed_r8, 4),
            'feasible_round_by_hop': {str(k): v for k, v in
                                      sorted(feasible.items())},
            'tau_sweep': sweep,
            'usable_taus': [s['tau'] for s in usable],
            'best_tau': best,
            'prereg_verdict': {
                'quality_pass': bool(usable),
                'rounds_pass': bool(best and best['mean_rounds'] <= 3.5),
                'saving_vs_r8_pass': bool(best and best['mean_rounds']
                                          <= 0.5 * args.r_max),
                'match_feasible_pass': bool(best
                                            and (best['match_feasible_rate']
                                                 or 0) >= 0.9),
                'match_hopm1_rate': (best['stop_match_rate']
                                     if best else None),
                'collapse_A4': not usable,
            },
            'falsified_if': '质量门内无 τ（A4 塌缩）或 hop 全解前提下 '
                            'match_feasible 仍 <0.6',
        }
    Path(args.out).write_text(json.dumps(final, ensure_ascii=False,
                                         indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'done': args.out,
                      **(final.get('ponder_eval', {})
                         .get('prereg_verdict', {}))}), flush=True)


if __name__ == '__main__':
    main()
