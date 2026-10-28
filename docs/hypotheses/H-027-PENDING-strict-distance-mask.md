# H-027（PENDING）：用“距离前沿”严格掩码强制每轮一跳

状态：PENDING（代码未实现；判据跑前写定）
日期：2026-10-28
关联：H-026（REFUTED）、P13、P12_adj、H-025

## 假说（一句话）

H-026 的冒烟之所以在 r=3 就解出 hop6，是因为当前掩码只限制
“事实 query → 事实 key 的图可达性”，而候选/非事实位置始终可见、
`prelude`/`coda` 未掩码、且融合事实 token 同时携带边的两端，导致
**一层注意力可以核对两条边（每轮约推进 2 跳）**。若把掩码改成
“第 t 轮事实 query 只能看到距 e0 **恰好 t-1 跳**的事实前沿”，并让
候选/标记 query 只能看到 `dist≤t` 的事实、`prelude` 用 t=0、
`coda` 用最后一轮偏置，则 `r<h` 应回到随机，`r≥h` 才出现爬升。

## 证据（来自 P13 冒烟，2026-10-28）

- 修复 `s88` 的“≤t 跳”漏 1 跳 bug 后，2×3000 步冒烟仍 FAIL：
  seed17/53 的 `dev_r1=0.650/0.714`；r=3 时 hop6 已 0.985/0.985。
- 同一 ckpt 的 mask on/off：r=3 时 with mask 0.987、no mask 0.466，
  说明评测确实在传掩码，高准确率依赖掩码，但不是阶梯。
- eval-only 探针（在旧 ckpt 上换更严格的掩码，不重训）：
  - “候选只能看 `dist≤t`、事实看不到候选 key”：r=3 仍 0.896，
    hop6 0.89 → **仅限制候选不够**；
  - “距离前沿”掩码：r=1..4 全部塌到 0.32–0.49（旧模型没学过这种
    输入分布，不能据此说新掩码可学，但说明旧解依赖“每轮多跳”的
    可达性，而不是真正的单跳递推）。
- 结论：问题不在评测透传，也不只在候选可见性；**掩码的粒度**
  （图可达 vs 距离前沿）和未掩码的 `prelude`/`coda` 是主嫌疑。

## 文献依据

1. **Chen 2026, arXiv:2603.21676**，
   *Thinking Deeper, Not Longer: Memory-Efficient Test-Time Reasoning with
   Depth-Recurrent Transformers for Compositional Generalization*。
   用 adjacency masking 把 thinking steps 与图距离绑定；其图通常是
   “节点=token”，一层传播一跳。我们的融合编码把一条边压成一个
   token（subj+rel+obj），一层可以同时核对两条边，这是与文献的
   关键差异，也是需要“距离前沿”而非“邻接可达”的原因。
   <https://arxiv.org/abs/2603.21676>
2. **Zhu et al. 2025/2026, arXiv:2510.25741**（Ouro）：
   只用于解释退出分布/熵正则，不用于支持掩码有效性。
   <https://arxiv.org/abs/2510.25741>

## 判定实验

### 阶段 0：严格掩码冒烟（先决条件）

实现（建议最小改动）：

- `s88_adj_task.py` 新增 `distance_from_e0(adj, fact_mask, max_t)` 与
  `adj_bias_for_round_strict(adj, fact_mask, cand_mask, t)`：
  - 事实 query：只允许 `self`、`dist == t-1` 的事实、以及标记 key
    （`[STATE]/[CURRENT]/[GOAL]/[ACTIONS]`）；**不允许候选 key**；
  - 候选/标记 query：只允许 `self`、标记、`dist <= t` 的事实；
  - `dist` 由 e0 在事实图上 BFS 得到（链式数据唯一）。
- `s63_ldm_model.forward` 增加 `adj_bias_prelude` / `adj_bias_coda`：
  - `prelude` 用 t=0 偏置（无事实-事实边，只留标记/自环）；
  - `coda` 用最后一轮偏置（防止读出前多走一跳）。
- `s71_ldm_iter4.py` 增加 `--adj-strict`，训练/评测/硬退出全部走
  strict 构造函数；`s65.evaluate` 增加 `adj_strict` 参数。
- 旧 `--adj-mask` 行为保持不变，便于复现 P12/P13。

运行：

```bash
cd /root/ldm
# 新 tag，唯一 --out；2 seed × 6000 步，cap=2
TAG=P14_strictmask STEPS=6000 SEEDS="17 53" bash _p14_strict_smoke.sh
```

（脚本可按 `_smoke_adj_eval.sh` 复制，命令加 `--adj-strict`；
`results/P14_strictmask_s17.json`、`..._s53.json` 已存在则整体拒绝。）

### 预注册判据（跑前写定，跑完不改）

- **S1 无泄漏**：两个 seed、所有 `h∈{2,3,4,5,6}`、所有 `r<h`：
  `acc ≤ 0.60`（2 候选随机=0.50，留 0.10 噪声）。
- **S2 可学**：两个 seed 的 hop2 在 `r≥2` 上
  `max acc ≥ 0.70`；且每个 seed `dev_r1 ≤ 0.55`。
- **S3 阶梯**：两个 seed 的 hop2 都满足 `dev_r1 ≤ 0.55`、
  `max_{r≥2} acc ≥ 0.70`、落差 `≥0.15`。
- **PASS** = 两个 seed 都 S1∧S2∧S3。
- 若 6000 步仍 `task≈ln2`（不学）但 S1 通过：只说明“掩码没泄漏”，
  不宣布可学；延长到 12k 再判，不直接上 48k。
- 若 S1 仍失败（`r<h` 有 cell >0.60）：停止“距离前沿”分支，转查
  融合编码本身（需要把事实拆成 entity/relation token，另立假设）。

### 阶段 1：正式实验（仅阶段 0 PASS 后）

- 8 seed（11,13,19,29,17,53,37,71）×48k，`--adj-mask --adj-strict`，
  唯一 `--out=results/P14_adj_strict_s<S>.json`，并发 ≤6。
- 主判据沿用 H-026 的 P1/P2/P3（不改口径）：
  - P1：≥4/8 满足 `h7≥0.80` 且 `h7≥h6−0.05`，
    `h7=confirm_r8_by_hop["7"]`、`h6=dev_r8_by_hop["6"]`；
  - P2：h∈{4,5,6} 中至少 2 个满足 `acc(r=h,h)≥0.85`、
    `mean_{r<h}≤0.55`、`r*(h)` 随 h 单调不减；
  - P3：所有 seed、所有 h、所有 `r<h`：`acc≤0.65`。
- 对照：P05（同配方无掩码）与 P13 冒烟（lax 掩码）。

## 成立/不成立的后续影响

- **成立**：说明“融合边 token + 距离前沿掩码”可以强制串行深度，
  H-025 的 `r*(h)` 首次可被干净检验；下一步做 `--step-sup` 消融
  （Chen 2026 报告 step 级监督损害外推）。
- **不成立（S1 失败）**：说明即使距离前沿也挡不住融合编码的
  一层多跳，需改编码（事实拆成 entity/relation token）或改读出
  （候选只读当前前沿的单一事实），不能再靠调掩码强度。
- **不成立（S1 过、S2 不过）**：严格掩码可防泄漏但优化太难；
  优先查课程/步数/监督信号，而不是回到 lax 掩码。
