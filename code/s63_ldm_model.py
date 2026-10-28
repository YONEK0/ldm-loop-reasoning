"""s63: LDM（循环决策模型）实现：符号分词、模型、精确参数量、单步 sizing。

规格：docs/plans/LOOPED_DECISION_MODEL_FEASIBILITY_SPEC_DRAFT_V1.md V1.1
数据：data/ldm_edge_action/（s62 冻结，manifest 含哈希）

本文件提供可导入的模型/数据接口，并支持三种 CLI 模式：
  --mode params   仅打印参数量分解（秒级）
  --mode sizing   CPU 前反传计时 + 参数量 + 峰值内存（不训练）
  --mode encode   打印一条分词结果（契约自检）

设计要点（与规格 §4 对应）：
  - 双向注意力（事实顺序随机打乱，因果掩码会破坏可用性）。
  - 正弦位置编码（无参数，长度外推友好）。
  - Core 的 2 个块在循环轮次间共享参数；Coda 只在读出处运行一次。
  - Core 与 Coda 的输出投影零初始化（D7）：初值下整条循环栈为恒等，
    H_t=(t+1)H_0，经 ln_loop 归一化后各轮读出相同；对称性由梯度打破。
  - 评分头为共享 Linear(d,1)，probe 与 Coda 后最终评分共用。
  - 退出控制器读 pooled hidden + 轮次嵌入 + 置信度特征
    (maxp, margin, 归一化熵, 与上轮分布 KL)。

纯 CPU；不加载任何预训练权重。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data' / 'ldm_edge_action'

# ---------------------------------------------------------------- vocab
PAD = 0
ENT_BASE = 1          # E000..E299 -> 1..300
REL_IDS = {'P': 301, 'M': 302, 'S': 303}
STATE, CURRENT, GOAL, ACTIONS, ACT, SEP, COMMA = 304, 305, 306, 307, 308, 309, 310
VOCAB_SIZE = 320      # 对齐到 8 的倍数，留扩展位


def ent_id(name: str) -> int:
    return ENT_BASE + int(name[1:])


def encode_example(ex: dict) -> tuple[list[int], list[int]]:
    """结构化字段 -> token ids + 候选评分位置（候选实体 token 的下标）。"""
    ids: list[int] = [STATE]
    for f in ex['facts']:
        ids.extend([ent_id(f['subj']), REL_IDS[f['rel']],
                    ent_id(f['obj']), SEP])
    ids.extend([CURRENT, ent_id(ex['e0']), GOAL])
    ids.extend(REL_IDS[r] for r in ex['goal'])
    ids.append(ACTIONS)
    pos: list[int] = []
    for c in ex['candidates']:
        ids.append(ACT)
        ids.append(REL_IDS[c['rel']])
        pos.append(len(ids))          # 实体 token 将落在此下标
        ids.append(ent_id(c['obj']))
        ids.append(COMMA)
    return ids, pos


def encode_example_fused(ex: dict) -> tuple[list[tuple[int, int, int]],
                                            list[int]]:
    """融合编码：每条事实占一个位置，输入向量 = subj+rel+obj embedding 之和。

    动机（实测诊断，LDM 日志 §5）：逐 token 编码下，第一层注意力中无任何
    位置同时携带 (实体,关系)——实体 token 不知自己的关系、关系 token 不知
    主语，模型须先自发学出位置绑定电路才能做关系 join。融合编码把 join
    变为第 1 层即可内容寻址。目标路径保持逐 token（顺序携带信息，不可求和）。
    """
    ids: list[tuple[int, int, int]] = [(STATE, 0, 0)]
    for f in ex['facts']:
        ids.append((ent_id(f['subj']), REL_IDS[f['rel']], ent_id(f['obj'])))
    ids.append((CURRENT, ent_id(ex['e0']), 0))
    ids.append((GOAL, 0, 0))
    ids.extend((REL_IDS[r], 0, 0) for r in ex['goal'])
    ids.append((ACTIONS, 0, 0))
    pos: list[int] = []
    for c in ex['candidates']:
        pos.append(len(ids))
        ids.append((ACT, REL_IDS[c['rel']], ent_id(c['obj'])))
    return ids, pos


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()
            if l.strip()]


def remap_example(ex: dict, perm: list[int]) -> dict:
    """实体编号置换：所有实体名 E### 统一按 perm 重编号，其余字段不变。
    任务逻辑恒等，仅摧毁"实体身份 -> 答案"的记忆关联。"""
    def emap(name: str) -> str:
        return f'E{perm[int(name[1:])]:03d}'

    ex = dict(ex)
    ex['e0'] = emap(ex['e0'])
    ex['e_answer'] = emap(ex['e_answer'])
    ex['facts'] = [{'subj': emap(f['subj']), 'rel': f['rel'],
                    'obj': emap(f['obj'])} for f in ex['facts']]
    ex['candidates'] = [{'rel': c['rel'], 'obj': emap(c['obj'])}
                        for c in ex['candidates']]
    ex['proof'] = [{'subj': emap(f['subj']), 'rel': f['rel'],
                    'obj': emap(f['obj'])} for f in ex.get('proof', [])]
    ex['question'] = ''
    return ex


class EncodedDataset:
    def __init__(self, rows: list[dict], max_len: int = 256,
                 fused: bool = False, perm: list[int] | None = None):
        """perm: 实体重映射表（长度 300，perm[旧编号]=新编号）。
        用于记忆捷径消融：训练时每 epoch 重掷置换，摧毁"实体对记忆"路径；
        结构（关系/标记）不变，任务逻辑恒等。"""
        self.fused = fused
        self.perm = perm
        self.items = []
        truncated = 0
        for ex in rows:
            if perm is not None:
                ex = remap_example(ex, perm)
            ids, pos = (encode_example_fused(ex) if fused
                        else encode_example(ex))
            if len(ids) > max_len:
                truncated += 1
                continue
            cand = ex['candidates']
            cdi = ex.get('comp_depth_by_index', {}) or {}
            chain_len = [
                (ex['hop'] - 1) if i == ex['answer_index']
                else int(cdi.get(str(i), cdi.get(i, -1)))
                for i in range(len(cand))]
            # 中间路径监督（scratchpad）：第 t 轮要走向的实体 id。
            # proof 给出逐步路径 [(s,r,o),...]，第 t 轮的目标是 proof[t].obj。
            _proof = ex.get('proof') or []
            step_ent = [ent_id(p['obj']) for p in _proof]
            self.items.append({
                'ids': ids, 'pos': pos, 'chain_len': chain_len,
                'n_cand': len(cand),
                'label': ex['answer_index'],
                'hop': ex['hop'],
                'goal_name': ex['goal_name'],
                'split': ex['split'],
                'id': ex['id'],
                'n_distractors': ex['n_distractors'],
                'answer_pos': ex['answer_index'],
                'step_ent': step_ent,
            })
        self.truncated = truncated

    def __len__(self):
        return len(self.items)

    def collate(self, idx: Iterable[int]):
        batch = [self.items[i] for i in idx]
        B = len(batch)
        S = max(len(b['ids']) for b in batch)
        C = max(b['n_cand'] for b in batch)
        ids = torch.zeros(B, S, dtype=torch.long)
        if self.fused:
            ids = torch.zeros(B, S, 3, dtype=torch.long)
        pad_mask = torch.zeros(B, S)
        cand_pos = torch.zeros(B, C, dtype=torch.long)
        cand_mask = torch.zeros(B, C, dtype=torch.bool)
        label = torch.zeros(B, dtype=torch.long)
        hop = torch.zeros(B, dtype=torch.long)
        chain_len = torch.full((B, C), -1.0)
        Rmax = max((len(b.get('step_ent') or []) for b in batch), default=0)
        step_ent = torch.full((B, max(Rmax, 1)), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b['ids'])
            ids[i, :L] = torch.tensor(b['ids'], dtype=torch.long)
            pad_mask[i, :L] = 1.0
            cand_pos[i, :b['n_cand']] = torch.tensor(b['pos'], dtype=torch.long)
            cand_mask[i, :b['n_cand']] = True
            cl = b.get('chain_len', [-1] * b['n_cand'])
            chain_len[i, :b['n_cand']] = torch.tensor(
                [float(x) for x in cl], dtype=torch.float)
            label[i] = b['label']
            hop[i] = b['hop']
            se = b.get('step_ent') or []
            if se:
                step_ent[i, :len(se)] = torch.tensor(se, dtype=torch.long)
        return {
            'ids': ids, 'pad_mask': pad_mask,
            'cand_pos': cand_pos, 'cand_mask': cand_mask,
            'label': label, 'hop': hop, 'chain_len': chain_len,
            'step_ent': step_ent,
            'meta': [{'id': b['id'], 'goal_name': b['goal_name'],
                      'hop': b['hop'], 'split': b['split']} for b in batch],
        }


# ---------------------------------------------------------------- model
def rope_freqs(max_len: int, head_dim: int,
               base: float = 10000.0) -> torch.Tensor:
    """RoPE 频率表：(max_len, head_dim/2)。"""
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float()
                          / head_dim))
    t = torch.arange(max_len).float()
    return torch.outer(t, inv)              # L, hd/2


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """x: B,H,S,hd -> 旋转位置编码（前 hd/2 维成对旋转）。"""
    S = x.size(-2)
    f = freqs[:S].to(x.device)              # S, hd/2
    cos = torch.cos(f).unsqueeze(0).unsqueeze(0)   # 1,1,S,hd/2
    sin = torch.sin(f).unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], dim=-1).flatten(-2)


def sinusoidal_pe(max_len: int, d: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class Block(nn.Module):
    """pre-norm 双向 Transformer 块；small_out 时残差支路以小方差初始化。

    历史：这里曾用精确零初始化（规格 D7，取自 Loop-Think-Generalize）。
    本项目实测发现精确零会切断整个子层的梯度——梯度只能到达被零化的
    最后一层投影，ln/qkv/up 全部 grad=0，可训练容量塌缩为单个 Prelude
    块，表现为训练集可拟合但 dev 停在随机基线。故改为小方差残差缩放：
    保留"初值接近恒等"的稳定意图，同时让梯度能进入子层内部。
    """

    def __init__(self, d: int, heads: int, ffn: int,
                 small_out: bool = False, out_scale: float = 0.02,
                 attn_mode: str = 'full', local_w: int = 0,
                 use_rope: bool = False):
        super().__init__()
        assert d % heads == 0
        self.d, self.heads, self.hd = d, heads, d // heads
        self.attn_mode = attn_mode
        self.local_w = local_w
        self.use_rope = use_rope
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.up = nn.Linear(d, ffn)
        self.down = nn.Linear(ffn, d)
        if small_out:
            nn.init.normal_(self.proj.weight, std=out_scale)
            nn.init.zeros_(self.proj.bias)
            nn.init.normal_(self.down.weight, std=out_scale)
            nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor,
                attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        """attn_bias: (B,1,S,S) 附加注意力偏置（如邻接掩码 -1e9），可为 None。"""
        B, S, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, S, 3, self.heads, self.hd)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)                     # B,H,S,hd
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.use_rope:
            if not hasattr(self, '_rope_f') or self._rope_f.size(0) < S:
                self._rope_f = rope_freqs(max(S, 64), self.hd)
            q = apply_rope(q, self._rope_f)
            k = apply_rope(k, self._rope_f)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.hd)
        pad_bias = (pad_mask < 0.5)[:, None, None, :]      # B,1,1,S
        scores = scores.masked_fill(pad_bias, -1e9)
        if attn_bias is not None:                 # 任务级结构约束（邻接掩码）
            ab = attn_bias
            if ab.size(-1) != S:
                ab = ab[:, :, :S, :S]
            scores = scores + ab
        if self.attn_mode == 'causal':
            causal = torch.triu(torch.ones(S, S, dtype=torch.bool,
                                           device=x.device), 1)
            scores = scores.masked_fill(causal, -1e9)
        elif self.attn_mode == 'local':
            pos = torch.arange(S, device=x.device)
            dist = (pos[:, None] - pos[None, :]).abs()
            scores = scores.masked_fill(dist > self.local_w, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, d)
        x = x + self.proj(out)
        x = x + self.down(F.gelu(self.up(self.ln2(x))))
        return x


def masked_mean(h: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    w = pad_mask.unsqueeze(-1)
    return (h * w).sum(1) / w.sum(1).clamp_min(1.0)


def gather_pos(h: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    B, S, d = h.shape
    idx = pos.unsqueeze(-1).expand(B, pos.size(1), d)
    return h.gather(1, idx)


class LoopedDecisionModel(nn.Module):
    """variant='loop'：prelude(1) + core(n_core, 共享) + coda(1)；
    variant='dense'：prelude(1) + mid(n_dense_mid, 独立) + coda(1)。"""

    def __init__(self, d: int = 192, heads: int = 4, ffn: int = 768,
                 n_prelude: int = 1, n_core: int = 2, n_coda: int = 1,
                 n_rounds: int = 4, variant: str = 'loop',
                 n_dense_mid: int = 2, max_len: int = 512,
                 zero_init_loop: bool = True,
                 round_in_core: bool = False,
                 pos: str = 'sin', attn_mode: str = 'full',
                 local_w: int = 0, bottleneck_dim: int = 0,
                 use_rope: bool = True, n_heads_read: int = 0,
                 film_rounds: bool = False, step_sup: bool = False,
                 head_type: str = 'linear'):
        super().__init__()
        self.d, self.variant = d, variant
        self.n_rounds = n_rounds
        self.round_in_core = round_in_core
        self.pos = pos
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d, padding_idx=PAD)
        if pos == 'sin':
            self.register_buffer('pe', sinusoidal_pe(max_len, d),
                                 persistent=False)
        else:              # rope / none：不加绝对PE（none 全零，Fan NoPE）
            self.register_buffer('pe', torch.zeros(1, max_len, d),
                                 persistent=False)
        self.prelude = nn.ModuleList([Block(d, heads, ffn, attn_mode=attn_mode,
                                            local_w=local_w, use_rope=(pos == 'rope' and use_rope))
                                      for _ in range(n_prelude)])
        zi = zero_init_loop
        if variant == 'loop':
            self.core = nn.ModuleList([Block(d, heads, ffn, small_out=zi,
                                             attn_mode=attn_mode,
                                             local_w=local_w,
                                             use_rope=(pos == 'rope' and use_rope))
                                       for _ in range(n_core)])
            self.mid = None
        else:
            self.mid = nn.ModuleList([Block(d, heads, ffn, small_out=zi,
                                            attn_mode=attn_mode,
                                            local_w=local_w,
                                            use_rope=(pos == 'rope' and use_rope))
                                      for _ in range(n_dense_mid)])
            self.core = None
        self.coda = nn.ModuleList([Block(d, heads, ffn, small_out=zi,
                                         attn_mode=attn_mode,
                                         local_w=local_w,
                                         use_rope=(pos == 'rope' and use_rope))
                                   for _ in range(n_coda)])
        self.ln_loop = nn.LayerNorm(d)
        self.bottleneck_dim = bottleneck_dim
        if bottleneck_dim > 0:      # 轮间信息瓶颈（承重结构）：
            # 跨轮只传 W_down→W_up 的低维状态；层维度无法在单轮内
            # 偷跑多步——下一轮可见的只有压缩后的 d_b 维信息。
            self.b_down = nn.Linear(d, bottleneck_dim)
            self.b_up = nn.Linear(bottleneck_dim, d)
        self.score = nn.Linear(d, 1)
        # 决策模块形态（读出侧单变量）：默认 linear 是现状——193 参数、
        # 只读候选 token 自己的隐状态。其余三种用来测"这 0.013% 的参数
        # 是不是承重"：bilinear 显式比较候选身份与状态；mlp 测容量；
        # xattn 让读出端自己检索全序列（而不是依赖 core 把答案压进候选位）。
        self.head_type = head_type
        if head_type == 'bilinear':
            self.head_bilinear = nn.Linear(d, d, bias=False)
        elif head_type == 'mlp':
            self.head_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                          nn.Linear(d, 1))
        elif head_type == 'xattn':
            self.head_q = nn.Linear(d, d)
            self.head_k = nn.Linear(d, d)
            self.head_v = nn.Linear(d, d)
            self.head_out = nn.Linear(d, 1)
        # H-009：per-round 读出头（0=关闭，共享单一 score）
        self.n_heads_read = n_heads_read
        if n_heads_read > 0:
            self.score_round = nn.ModuleList(
                [nn.Linear(d, 1) for _ in range(n_heads_read)])
        # 方案 A：FiLM 轮次调制。比 additive round_emb 强得多——每轮对
        # 残差状态做 per-dim scale+shift，使共享 core 能真正按轮次条件化，
        # 直接对抗"第 k 轮与第 k+1 轮抢同一表示"的寻址冲突（H-007）。
        self.film_rounds = film_rounds
        if film_rounds:
            self.film_gamma = nn.Embedding(max(n_rounds, 16), d)
            self.film_beta = nn.Embedding(max(n_rounds, 16), d)
            nn.init.zeros_(self.film_gamma.weight)   # 初始化=恒等，不破坏起点
            nn.init.zeros_(self.film_beta.weight)
        # 退出控制器（仅 loop 变体；dense 诊断臂不构造，避免参数账虚增）
        if variant == 'loop':
            self.round_emb = nn.Embedding(max(n_rounds, 16), d)
            self.exit_mlp = nn.Sequential(nn.Linear(d + 4, 64), nn.GELU(),
                                          nn.Linear(64, 1))
        else:
            self.round_emb = None
            self.exit_mlp = None
        self.proc_head = nn.Linear(d, 1)   # 过程监督：候选链存活位
        # 中间路径监督：预测"下一跳走向哪个实体"（vocab 上的分类头）。
        # 给每轮提供稠密的学习目标，破解 hop5 的冷启动门槛——
        # 模型不必先偶然猜对整个答案才能获得梯度。
        self.step_head = nn.Linear(d, VOCAB_SIZE) if step_sup else None
        self.step_sup = step_sup

    @staticmethod
    def _adj_bias(adj: torch.Tensor, steps: int,
                  pad_mask: torch.Tensor,
                  start_idx: int = 0) -> torch.Tensor:
        """把邻接矩阵转成"距起点 ≤ steps 跳"的注意力偏置 (B,1,S,S)。

        adj: (B,S,S) bool，adj[b,i,j]=True 表示 token j 可从 token i 一跳到达。
        用布尔矩阵幂累积：reach = I | A | A² | ... | A^steps。
        不可达位置加 -1e9，使该轮只能看到 steps 跳内的节点。
        """
        B, S, _ = adj.shape
        dev = adj.device
        eye = torch.eye(S, dtype=torch.bool, device=dev).unsqueeze(0)
        reach = eye.expand(B, S, S).clone()
        front = reach.clone()
        for _ in range(steps):
            front = torch.einsum('bij,bjk->bik', front.to(torch.float32),
                                 adj.to(torch.float32)) > 0
            reach = reach | front
        bias = torch.zeros(B, 1, S, S, dtype=torch.float32, device=dev)
        bias = bias.masked_fill(~reach.unsqueeze(1), -1e9)
        return bias

    # ---- helpers
    def _run(self, x: torch.Tensor, blocks: nn.ModuleList,
             pad_mask: torch.Tensor,
             attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        for b in blocks:
            x = b(x, pad_mask, attn_bias)
        return x * pad_mask.unsqueeze(-1)

    @staticmethod
    def _conf_feats(probs: torch.Tensor, cand_mask: torch.Tensor,
                    prev: torch.Tensor | None):
        n_valid = cand_mask.sum(1, keepdim=True).clamp_min(1).float()
        p = probs.clamp_min(1e-9)
        logp = p.log()
        ent = -(p * logp).sum(1) / n_valid.squeeze(1).log().clamp_min(1e-6)
        neg = (~cand_mask).float()
        masked = probs.masked_fill(neg.bool(), -1.0)
        top2 = masked.topk(2, dim=1).values
        maxp = top2[:, 0].clamp_min(0)
        margin = (top2[:, 0] - top2[:, 1]).clamp_min(0)
        if prev is None:
            kl = torch.zeros_like(maxp)
        else:
            kl = (prev.clamp_min(1e-9) * (prev.clamp_min(1e-9).log() - logp)
                  ).sum(1).clamp_min(0)
        return torch.stack([maxp, margin, ent, kl], dim=1)

    def _logits(self, h: torch.Tensor, cand_pos: torch.Tensor,
                cand_mask: torch.Tensor, t_idx: int = -1,
                cand_emb: torch.Tensor | None = None,
                seq: torch.Tensor | None = None,
                pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # H-009：per-round 独立读出头。t_idx>=0 且启用分头时，用该轮的
        # 专属 score 层——使相邻轮次的目标表示不再共享同一组参数，
        # 直接消解"学到轮 k 就毁掉轮 k+1"的竞争（E-001/H-007）。
        # 分头只对 linear 形态有意义，其余形态走各自分支。
        g = gather_pos(h, cand_pos)          # B,C,d
        if self.head_type == 'linear':
            if getattr(self, 'n_heads_read', 0) > 0 and t_idx >= 0:
                layer = self.score_round[min(t_idx, self.n_heads_read - 1)]
                return layer(g).squeeze(-1)
            return self.score(g).squeeze(-1)     # B,C
        if self.head_type == 'bilinear':
            # cand_emb 是候选位置的输入表示（融合编码下 = ACT+REL+OBJ），
            # 用它和隐状态做双线性比较，而不是只让线性头读隐状态
            return (self.head_bilinear(g) * cand_emb).sum(-1)
        if self.head_type == 'mlp':
            return self.head_mlp(g).squeeze(-1)
        # xattn：候选位当 query，对整条序列做单头注意力后打分
        q = self.head_q(g)
        k, v = self.head_k(seq), self.head_v(seq)
        att = (q @ k.transpose(1, 2)) / math.sqrt(self.d)
        att = att.masked_fill(pad_mask.unsqueeze(1) <= 0, -1e9)
        ctx = torch.softmax(att, dim=-1) @ v
        return self.head_out(ctx).squeeze(-1)

    # ---- forward
    def forward(self, ids, pad_mask, cand_pos, cand_mask, n_rounds=None,
                return_final_only=False, rounds_mask=None,
                return_round_states=False, return_diag=False,
                warm_noise: float = 0.0, warm_from: int = 6,
                adj_mask=None, hop_vec=None, adj_bias_list=None):
        """adj_mask: (B,S,S) bool 邻接矩阵（True=可达一跳）。

        若同时给出 hop_vec（(B,) 目标跳数），则第 t 轮的注意力被限制在
        "距起点 ≤ t 跳"的节点集合内 —— 强制每轮只多传播一跳
        （Chen 2026 的 topological interface 做法）。
        这样"r 轮最多推进 r 跳"成为物理约束，可用来检验串行深度。
        """
        if ids.dim() == 3:      # 融合编码：每位置 3 个 id 求和（PAD=0 贡献 0）
            x = self.tok_emb(ids).sum(dim=2)
        else:
            x = self.tok_emb(ids)
        x = x + self.pe[:, :ids.size(1)]
        x = x * pad_mask.unsqueeze(-1)
        cand_emb = gather_pos(x, cand_pos)   # 候选位置的输入表示（判别头用）
        h0 = self.ln_loop(self._run(x, self.prelude, pad_mask))
        h_raw = h0
        rounds = []
        prev_probs = None
        diag = {'h0_norm': float(h0.norm(dim=-1).mean()),
                'round_norm': [], 'h0_share': []} if return_diag else None
        if self.variant == 'loop':
            steps = n_rounds or self.n_rounds
            units = [self.core] * steps
        else:
            steps = len(self.mid)
            units = [self.mid]
        idx = 0
        for blk in units:
            if self.variant == 'loop':
                if self.bottleneck_dim > 0 and idx > 0:
                    carry = self.b_up(self.b_down(h_raw))
                else:
                    carry = h_raw
                inp = carry + h0
                if self.round_in_core:
                    inp = inp + self.round_emb.weight[
                        min(idx, self.round_emb.num_embeddings - 1)]
                # 邻接掩码（P1-2 用）：第 idx 轮只允许看 ≤ idx+1 跳内
                _bias = None
                if adj_bias_list is not None:
                    # 预计算的逐轮偏置（可精细控制哪些查询位置受限制）
                    _bias = (adj_bias_list[idx]
                             if idx < len(adj_bias_list) else None)
                elif adj_mask is not None:
                    _bias = self._adj_bias(adj_mask, idx + 1, pad_mask)
                h_new = self._run(inp, blk, pad_mask, _bias)
                if warm_noise > 0 and idx >= warm_from:
                    # 状态唤醒：在后期轮次注入噪声，扩大可达状态覆盖
                    h_new = h_new + torch.randn_like(h_new) * warm_noise
                if self.film_rounds:
                    # FiLM：按轮次调制（scale+shift），初始化恒等
                    _t = min(idx, self.round_emb.num_embeddings - 1)
                    h_new = h_new * (1.0 + self.film_gamma.weight[_t]) \
                        + self.film_beta.weight[_t]
                if rounds_mask is not None:
                    act = rounds_mask[:, idx].float().view(-1, 1, 1)
                    h_raw = act * h_new + (1.0 - act) * h_raw
                else:
                    h_raw = h_new
            else:
                h_raw = self._run(h_raw, blk, pad_mask)
            h_read = self.ln_loop(h_raw)
            if diag is not None:
                # 文献诊断量（arXiv:2605.18797）：残差范数随轮次是否膨胀，
                # 以及 h0 重注入占残差的比例（residual explosion 的直接指标）
                with torch.no_grad():
                    rn = h_raw.norm(dim=-1).mean()
                    hn = h0.norm(dim=-1).mean().clamp_min(1e-6)
                    diag['round_norm'].append(float(rn))
                    diag['h0_share'].append(float(rn / hn))
            if return_final_only and idx < steps - 1:
                idx += 1
                continue
            logits = self._logits(h_read, cand_pos, cand_mask, t_idx=idx,
                                  cand_emb=cand_emb, seq=h_read,
                                  pad_mask=pad_mask)
            if self.variant == 'loop':
                probs = torch.softmax(
                    logits.masked_fill(~cand_mask, -1e9), dim=1)
                feats = self._conf_feats(probs, cand_mask, prev_probs)
                pooled = masked_mean(h_read, pad_mask)
                t_idx = min(idx, self.round_emb.num_embeddings - 1)
                r_in = pooled + self.round_emb.weight[t_idx].unsqueeze(0)
                q = torch.sigmoid(
                    self.exit_mlp(torch.cat([r_in, feats], dim=1))).squeeze(1)
                surv = self.proc_head(
                    gather_pos(h_read, cand_pos)).squeeze(-1)
                _rd = {'logits': logits, 'q': q, 'feats': feats,
                       'probs': probs, 'surv': surv}
                if self.step_head is not None:
                    # 用当前轮读出态预测"这一跳走向哪个实体"
                    _rd['step_logits'] = self.step_head(
                        masked_mean(h_read, pad_mask))
                rounds.append(_rd)
                if return_round_states:
                    # §19.7：暴露逐轮读出态，供 PonderNet 逐轮 coda 读出与
                    # 硬退出评测取"该样本退出轮"的 h_read——评测从此直接
                    # 走 forward，不再手写展开（s67 展开缺 round_emb 的教训）
                    rounds[-1]['h_read'] = h_read
                prev_probs = probs.detach()
            else:
                # dense：无循环，但同样可提供 step-sup 监督所需的
                # 单轮 step_logits（保证 loop/dense 对比的监督公平性）
                _rd = {'logits': logits}
                if self.step_head is not None:
                    _rd['step_logits'] = self.step_head(
                        masked_mean(h_read, pad_mask))
                rounds.append(_rd)
            idx += 1
        final_h = self._run(h_read, self.coda, pad_mask)
        final_logits = self._logits(final_h, cand_pos, cand_mask,
                                    t_idx=idx, cand_emb=cand_emb,
                                    seq=final_h, pad_mask=pad_mask)
        out = {'rounds': rounds, 'final_logits': final_logits,
               'final_h': final_h, 'cand_emb': cand_emb}
        # dense 便捷出口：单轮 step_logits（loop 请走 rounds[t]）
        if self.step_head is not None and len(rounds) == 1:
            out['step_logits'] = rounds[0].get('step_logits')
        if diag is not None:
            out['diag'] = diag
        return out

    # ---- param accounting
    def param_report(self) -> dict:
        groups: dict[str, int] = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            g = name.split('.')[0]
            if name.startswith('prelude'):
                g = 'prelude'
            elif name.startswith('core'):
                g = 'core(shared)'
            elif name.startswith('mid'):
                g = f'mid({len(self.mid)} unique)'
            elif name.startswith('coda'):
                g = 'coda'
            elif name.startswith('tok_emb'):
                g = 'token_emb'
            groups[g] = groups.get(g, 0) + p.numel()
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        unrolled = None
        if self.variant == 'loop':
            core = groups.get('core(shared)', 0)
            unrolled = total + core * (self.n_rounds - 1)
        return {'groups': groups, 'total': total,
                'core_params': groups.get('core(shared)', 0),
                'unrolled_equiv_params': unrolled}


def build_model(variant: str, n_rounds: int = 4, n_dense_mid: int = 2,
                d: int = 192, heads: int = 4, ffn: int = 768,
                n_prelude: int = 1, n_core: int = 2, n_coda: int = 1,
                zero_init_loop: bool = True,
                round_in_core: bool = False,
                pos: str = 'sin', attn_mode: str = 'full',
                local_w: int = 0,
                bottleneck_dim: int = 0,
                n_heads_read: int = 0,
                film_rounds: bool = False,
                step_sup: bool = False,
                head_type: str = 'linear') -> LoopedDecisionModel:
    return LoopedDecisionModel(
        d=d, heads=heads, ffn=ffn, n_prelude=n_prelude, n_core=n_core,
        n_coda=n_coda, n_rounds=n_rounds, variant=variant,
        n_dense_mid=n_dense_mid, zero_init_loop=zero_init_loop,
        round_in_core=round_in_core, pos=pos, attn_mode=attn_mode,
        local_w=local_w, bottleneck_dim=bottleneck_dim,
        n_heads_read=n_heads_read, film_rounds=film_rounds,
        step_sup=step_sup, head_type=head_type)


# ---------------------------------------------------------------- CLI
def _batch_from(split: str, bs: int, seed: int = 0):
    rows = load_jsonl(DATA_DIR / f'{split}.jsonl')
    ds = EncodedDataset(rows)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:bs].tolist()
    return ds.collate(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='params',
                    choices=['params', 'sizing', 'encode'])
    ap.add_argument('--variant', default='loop', choices=['loop', 'dense'])
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--dense-mid', type=int, default=2)
    ap.add_argument('--d', type=int, default=192)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--ffn', type=int, default=768)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--split', default='train')
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    if args.mode == 'encode':
        rows = load_jsonl(DATA_DIR / 'pilot.jsonl')
        ex = rows[0]
        ids, pos = encode_example(ex)
        print(json.dumps({'n_tokens': len(ids), 'ids': ids,
                          'cand_pos': pos,
                          'candidates': ex['candidates'],
                          'answer_index': ex['answer_index']}, indent=2))
        print('---')
        print(ex['question'][:400])
        return 0

    if args.mode == 'params':
        combos = []
        for variant, rounds, mid in [('loop', 1, 2), ('loop', 2, 2),
                                     ('loop', 4, 2), ('loop', 6, 2),
                                     ('dense', 1, 2), ('dense', 1, 8)]:
            m = build_model(variant, n_rounds=rounds, n_dense_mid=mid,
                            d=args.d, heads=args.heads, ffn=args.ffn)
            rep = m.param_report()
            combos.append({'variant': variant, 'rounds': rounds,
                           'dense_mid': mid, 'total': rep['total'],
                           'core_shared': rep['core_params'],
                           'groups': rep['groups']})
        print(json.dumps({'d': args.d, 'heads': args.heads, 'ffn': args.ffn,
                          'vocab': VOCAB_SIZE, 'combos': combos}, indent=2))
        return 0

    # sizing
    m = build_model(args.variant, n_rounds=args.rounds,
                    n_dense_mid=args.dense_mid, d=args.d,
                    heads=args.heads, ffn=args.ffn)
    rep = m.param_report()
    batch = _batch_from(args.split, args.bs)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    times = []
    peak_rss = 0.0
    for it in range(args.iters):
        t0 = time.perf_counter()
        out = m(batch['ids'], batch['pad_mask'], batch['cand_pos'],
                batch['cand_mask'])
        ce = F.cross_entropy(
            out['final_logits'].masked_fill(~batch['cand_mask'], -1e9),
            batch['label'])
        loss = ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        times.append(time.perf_counter() - t0)
        try:
            import resource
            peak_rss = max(peak_rss, resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss / 1024.0)
        except Exception:
            pass
    S = batch['ids'].size(1)
    tokens = batch['ids'].size(0) * S
    depth = (args.rounds * 2 + 2) if args.variant == 'loop' else \
        (args.dense_mid + 2)
    macs_per_token_block = 4 * args.d * args.d + 2 * args.d * args.ffn
    fwd_macs = tokens * depth * macs_per_token_block
    rec = {
        'variant': args.variant, 'rounds': args.rounds,
        'dense_mid': args.dense_mid, 'd': args.d, 'heads': args.heads,
        'ffn': args.ffn, 'bs': args.bs, 'seq_len': S, 'depth_blocks': depth,
        'params': rep, 'step_sec_mean': round(sum(times) / len(times), 4),
        'step_sec_all': [round(t, 4) for t in times],
        'peak_rss_mb': round(peak_rss, 1),
        'fwd_macs_est': int(fwd_macs),
        'fwd_macs_note': 'attn term未计; 4d^2为qkv+proj, 2d*ffn为mlp',
    }
    print(json.dumps(rec, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=2) + '\n',
                                  encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
