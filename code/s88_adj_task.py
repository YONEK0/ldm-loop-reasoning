"""s88_adj_task.py — 邻接掩码任务（P1-2）：强制"每轮只多传播一跳"。

## 目的

当前 MVT-D v2 上，模型在 r=2 就饱和（r=3–8 无增益），说明它用**压缩式推理**
（一轮传播多跳），而不是"每轮推进一跳"。
因此本任务**无法检验**"循环承载串行深度"这一核心假说。

本模块给模型加**拓扑掩码**（Chen 2026, arXiv:2603.21676 的做法）：
第 t 轮，一个事实位置只能看到**距它 ≤ t 跳**的事实。
这样 hop h 的题在 r < h 时**物理上不可能**答对，阶梯成为必然。

## 实现

样本已经是 MVT-D v2 格式（`s84_dual_task.py` 生成）。
编码（`encode_example_fused`）为：每条事实占一个位置，顺序与 `ex['facts']` 一致。

对每个样本构造 (S, S) 布尔邻接：
  adj[j][i] = True  表示"事实 i 的信息可在一跳内流向事实 j"
                    （即 facts[i].obj == facts[j].subj）
另外：
  - 每个位置可看自己（自环）；
  - [STATE]/[CURRENT]/[GOAL]/[ACTIONS] 与候选位置**不受限**（它们是
    任务标记与读出点，不是内容节点）；
  - 内容传播只发生在事实位置之间。

于是 r 轮后，事实 i 的信息最远到达距它 r 跳的事实。
hop h 的答案需要从 e0 出发 h 跳，故至少需要 h 轮才能让答案事实"看到"起点。

## 用法

  from s88_adj_task import build_adj_batch
  bias = build_adj_batch(rows, round_idx=t, max_len=256)   # (B,1,S,S)
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

import s63_ldm_model as M

ROOT = Path(__file__).parent


def build_adj_batch(facts_list: list[list[dict]], e0_list: list[str],
                    max_len: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """为一个 batch 构造事实级邻接矩阵（CPU，bool）。

    返回 (adj, fact_mask)：
      adj: (B, S, S) bool，adj[b, j, i] = 事实 i 可一跳流向事实 j
      fact_mask: (B, S) bool，True 表示该位置是事实位置
    """
    B = len(facts_list)
    adj = torch.zeros(B, max_len, max_len, dtype=torch.bool)
    fact_mask = torch.zeros(B, max_len, dtype=torch.bool)
    dist = torch.full((B, max_len), 10**6, dtype=torch.long)
    for b, facts in enumerate(facts_list):
        n = min(len(facts), max_len - 1)
        # 事实位置从 1 开始（位置 0 是 [STATE]）
        subj_of = [f['subj'] for f in facts[:n]]
        obj_of = [f['obj'] for f in facts[:n]]
        for i in range(n):
            fact_mask[b, i + 1] = True
        # 自环
        for i in range(n):
            adj[b, i + 1, i + 1] = True
        # 一跳边：fact i 的 obj == fact j 的 subj  -> i 的信息流向 j
        for i in range(n):
            for j in range(n):
                if i != j and obj_of[i] == subj_of[j]:
                    adj[b, j + 1, i + 1] = True
        # 起点：e0 视为"第 0 跳的已知信息"，所有 subj==e0 的事实可一跳看到
        for j in range(n):
            if subj_of[j] == e0_list[b]:
                adj[b, j + 1, 0] = True        # 位置 0 = [STATE]，代表起点上下文
        # BFS：事实到 e0 的跳数（fact i 的 subj 距 e0 的最短距离）
        import collections
        out_edges = {}
        for i in range(n):
            out_edges.setdefault(subj_of[i], []).append(obj_of[i])
        d = {e0_list[b]: 0}
        q = collections.deque([e0_list[b]])
        while q:
            u = q.popleft()
            for v in out_edges.get(u, []):
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        for i in range(n):
            di = d.get(subj_of[i])
            if di is not None:
                dist[b, i + 1] = min(di, 10**5)
    return adj, fact_mask, dist


def adj_bias_for_round(adj: torch.Tensor, fact_mask: torch.Tensor,
                       t: int, dist: torch.Tensor | None = None
                       ) -> torch.Tensor:
    """第 t 轮（1-based）的注意力偏置 (B,1,S,S)。

    **每轮只允许 1 跳**（adj 本身，而非其 t 次幂）。
    这样 r 轮的信息传播半径恰为 r 跳 → hop h 在 r<h 时物理上不可能答对，
    阶梯成为任务约束的必然结果（H-025 的判据）。

    参数 t 仅用于兼容调用签名，不再扩大可见范围；
    如需放宽可显式改这里的 hops。
    候选/标记位置不受限（读出点），事实位置只可见 1 跳邻居+自身+
    全部非事实位置（否则 e0 进不了事实链）。
    """
    B, S, _ = adj.shape
    dev = adj.device
    hops = 1                                   # 关键：每轮恰好一跳
    reach = torch.eye(S, dtype=torch.bool, device=dev).unsqueeze(0).expand(
        B, S, S).clone()
    front = adj.clone()
    # 注意：hops=1 时也必须并入 adj 本身（旧实现用 range(hops-1) 会漏掉）
    for _ in range(max(1, hops)):
        reach = reach | front
        if _ == hops - 1:
            break
        front = torch.einsum('bij,bjk->bik', front.to(torch.float32),
                             adj.to(torch.float32)) > 0
    q_is_fact = fact_mask.unsqueeze(-1)          # (B,S,1)
    nonfact_key = (~fact_mask).unsqueeze(1)      # (B,1,S)
    allowed_fact = reach | nonfact_key
    if dist is not None:
        # 距离分层（BFS）：第 t 轮只让 dist <= t-1 的事实可见，
        # 这样 hop h 需要 dist=h-1 的答案事实，r<h 时物理上不可见
        frontier = (dist <= (t - 1)).unsqueeze(1) & fact_mask.unsqueeze(1)
        allowed_fact = (reach & frontier) | nonfact_key
        allowed_cand = frontier | nonfact_key
        allowed = torch.where(q_is_fact, allowed_fact, allowed_cand)
    else:
        allowed = torch.where(q_is_fact, allowed_fact,
                              torch.ones_like(reach))
    bias = torch.zeros(B, 1, S, S, dtype=torch.float32, device=dev)
    bias = bias.masked_fill(~allowed.unsqueeze(1), -1e9)
    return bias


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding='utf-8')
            .splitlines() if l.strip()]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/ldm_mvtd2/dev_random.jsonl')
    ap.add_argument('--n', type=int, default=8)
    args = ap.parse_args()
    rows = load_rows(Path(args.data))[:args.n]
    adj, fm, dist = build_adj_batch([r['facts'] for r in rows],
                                    [r['e0'] for r in rows])
    print('adj shape', adj.shape, 'fact positions per sample',
          fm.sum(1).tolist())
    for t in (1, 2, 3, 6):
        bias = adj_bias_for_round(adj, fm, t, dist)
        # 统计事实位置平均可看位置数
        vis = (bias[0, 0] > -1).sum(1).float()
        fvis = vis[fm[0]]
        print(f't={t}: 事实位置平均可见 {fvis.mean():.1f} / {adj.shape[1]}')


if __name__ == '__main__':
    main()
