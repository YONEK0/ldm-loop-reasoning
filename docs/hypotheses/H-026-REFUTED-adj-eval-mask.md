# H-026（REFUTED）：邻接掩码的 dev 假随机来自评测漏传掩码

状态：**REFUTED（2026-10-28）**——评测漏传 `adj_bias_list` 被证实为真实
bug，但“补掩码后应出现 `r<h≈随机、r≥h 显著更高` 阶梯”的核心预测
被冒烟 C3 证伪（r=3 已把 hop6 解到 0.985；详见文末“最终判读”）。
日期：2026-10-28
关联：P12_adj、P13、H-027（严格掩码，下一步）、H-025（简报引用的邻接掩码阶梯；文件不在当前快照）

## 假说（一句话）

`P12_adj` 的 dev 恒 ≈0.50 是**评测路径没有传 `adj_bias_list`** 造成的
分布错配，而不是掩码阻断了学习；把同一逐轮邻接偏置传进评测前向后，
带掩码训练的模型应表现出“`r<h` ≈ 随机、`r≥h` 显著高于随机”的阶梯，
并在 48k 正式实验中恢复可学性。

## 文献依据

1. **Chen 2026, arXiv:2603.21676**，
   *Thinking Deeper, Not Longer: Memory-Efficient Test-Time Reasoning with
   Depth-Recurrent Transformers for Compositional Generalization*。
   该文在 graph reachability 上用 adjacency masking 把传播深度变成
   可验证约束，报告“thinking steps 达到任务复杂度后准确率才爬升”的
   **computational frontier**，且深度循环可外推到训练范围之外。
   与本项目最直接相关；也提示 **intermediate per-step supervision 会
   损害该任务上的外推**——我们的配方带 `--step-sup`，正式实验应把
   “step-sup 是否抵消掩码收益”列为待检验的混淆项（另立假设，不在
   H-026 里事后改口径）。
   <https://arxiv.org/abs/2603.21676>
2. **Zhu et al. 2025/2026, arXiv:2510.25741**，
   *Scaling Latent Reasoning via Looped Language Models*（Ouro）。
   用熵正则的深度分配目标训练循环模型；本项目的 `--ouro` 直接采用其
   均匀先验 + 熵奖励。该文支持“退出分布需要显式正则，否则塌到固定轮数”，
   但不涉及邻接掩码；因此不能用来支持掩码有效性，只用于解释训练目标。
   <https://arxiv.org/abs/2510.25741>

## 判定实验

### 阶段 0：冒烟（先决条件，未通过不得进入阶段 1）

```bash
cd /root/ldm
bash _smoke_adj_eval.sh
```

- 2 seed（17, 53）× 3000 步，标准配方 + `--adj-mask`
  + `--curriculum-step 8000`（3000 步内课程停在 cap=2，只改训练步数）。
- 每个 run 唯一 `--out=results/P13_smoke_adjfix_s<S>.json`；
  `_smoke_adj_eval.sh` 发现文件已存在会整体拒绝。
- 判据由 `s90_check_adj_smoke.py` 固化（跑前写定）：
  - **C1 非随机**：某个已训练 hop 在某 r 上 acc ≥ 0.70，
    且 `dev_r1 ≤ 0.55`；
  - **C2 阶梯**：存在已训练 hop h，`max_{r≥h} acc ≥ 0.65`，
    `r<h` 的平均 acc ≤ 0.55，落差 ≥ 0.15（用 max 而非只取 r=h：
    Ouro 退出分布偏后，r=h 的读数可能尚未校准；冒烟只验证方向）；
  - **C3 无泄漏**：所有 hop 在 `r<h` 时 acc ≤ 0.65。
- **成立**：两个 seed 都 C1∧C2∧C3。
- **不成立**：任一 seed 失败 → 停，不启动 48k；先区分
  “评测仍错”（C1 失败）与“掩码泄漏/可学性差”（C2 或 C3 失败）。

### 阶段 1：正式实验（仅在阶段 0 PASS 后）

```bash
bash _p13_launch.sh
```

- 8 seed：11, 13, 19, 29, 17, 53, 37, 71；48k；标准配方 + `--adj-mask`；
  唯一 `--out=results/P13_adj_evalfix_s<S>.json`；并发 ≤6（两批）。
- 对照：`P05_48k_*`（同配方、无掩码，seed 11/13/19/29）。
  与 P13 相比只改 `--adj-mask` 一个变量；P05 未受本次评测修复影响
  （它本来就没有掩码）。
- 指标：`dev_r{r}_by_hop`、`confirm_r8_by_hop`、`dev_r1`、训练 `task`。
  **掩码下 r<h 物理上不可能，因此确认深度必须用 r=8，不能再用
  `confirm_r6_by_hop`（hop7 在 r=6 时 r<h，旧指标会系统性低估）。**

## 预注册判据（跑完不改口径）

**P1 配方可靠性（主判据）**
- 8 个 seed 中 ≥4 个（50%）满足：
  `h7 ≥ 0.80` 且 `h7 ≥ h6 − 0.05`；
  其中 `h7 = confirm_r8_by_hop["7"]`（r=8 ≥ h=7），
  `h6 = dev_r8_by_hop["6"]`。
- 强复现（§8.6 的“≥5/8”）：同一批 run 中 ≥5 个满足上式。
- 辅助报告 `h8 = confirm_r8_by_hop["8"]`，不单独作为通过门槛
  （hop8 需要 r=8，正好在 r_max 边界，允许更低的期望）。

**P2 串行阶梯（机制判据）**
- 在 P1 通过的 seed 上，对 h ∈ {4,5,6}：
  - `acc(r=h, h) ≥ 0.85`；
  - `mean_{r<h} acc(r, h) ≤ 0.55`；
  - `r*(h) = min{r : acc(r,h) ≥ 0.85}` 随 h **单调不减**。
- 三个 h 中至少 2 个满足才算 P2 通过（允许单个 hop 的 seed 噪声）。

**P3 掩码完整性**
- 所有 seed、所有 h、所有 `r<h`：`acc(r,h) ≤ 0.65`。
- 任一 cell 超过 0.65 → 视为掩码泄漏，P2 的“物理不可能”解释不成立。

**判定**
- H-026 **成立**：阶段 0 PASS，且 P1 通过，且 P3 无泄漏。
- H-026 **部分成立**：阶段 0 PASS（评测 bug 确证），但 P1 通过率 <50%；
  此时不得再归因于评测，应记录“掩码下优化/泛化仍不可靠”。
- H-026 **被推翻**：阶段 0 的 C1 失败（补掩码后仍恒 ≈0.50 且训练
  task≈0）→ 评测漏传不是主因，回到掩码实现/优化退化假设。
- 若 P3 失败而 P1 仍高：说明模型可能利用候选/非事实位置的可见性，
  需另立“候选读出也纳入可达性约束”的假设（要重训，不能事后改 H-026）。

## 成立/不成立的后续影响

- **成立**：P12 作废读数的原因被定位为评测口径；可以放心用
  `r*(h)` 检验“循环承载串行深度”（H-025 的方向），并把
  `--step-sup` 与掩码的交互列为下一个单变量消融（Chen 2026 明确报告
  step 级监督损害该任务外推）。
- **部分成立**：掩码不是免费午餐；优先做 `--step-sup` 消融与
  掩码强度/候选可见性设计，而不是继续堆 seed。
- **被推翻**：停止 P13，转查 `s88_adj_task` 的图构造（实体重名、
  截断、e0 边）与优化退化；不得再把 P12 的 0.50 当评测 bug 处理。

## 最终判读（2026-10-28，沙箱解除后实测）

### 执行

修复 `s88_adj_task.adj_bias_for_round` 的“≤t 跳”漏 1 跳 bug 后，运行
预注册冒烟 `bash _smoke_adj_eval.sh`（2 seed×3000 步，cap=2）。
结果：`results/P13_smoke_adjfix_s17.json`、
`..._s53.json`；checker 输出 `results/P13_smoke_adjfix_check.txt`。

### 结果：SMOKE FAIL（0/2 seeds）

| seed | dev_r1 | best trained acc | C1 | C2 | C3 |
|---|---|---|---|---|---|
| 17 | 0.650 | 0.995 | FAIL | FAIL | FAIL |
| 53 | 0.714 | 0.995 | FAIL | FAIL | FAIL |

最关键的 C3 泄漏：seed17 的 hop6 在 r=3 已 **0.985**（r1/r2 也有
0.645）；seed53 的 hop6 在 r=3 为 0.985。即 **3 轮就能解 6 跳**，
“r<h 物理上不可能”不成立。

同一 ckpt 的 mask on/off 对照（s17，dev_random）：

| r | with mask | no mask |
|---|---|---|
| 1 | .650 | .482 |
| 2 | .639 | .484 |
| 3 | .987 | .466 |
| 4 | .995 | .466 |

- **评测漏传确实是真 bug**：关掉掩码立即塌回随机；修复后掩码模型
  不再恒 0.50。
- **但 H-026 的阶梯预测被证伪**：高准确率不是“r≥h 才出现”，而是
  r=3 对所有 hop 一起跳变；C3 明确泄漏。
- 训练侧并非不学：task loss 在 3000 步末已降到 0.236（s17）/
  0.005（s53），所以不能解释为“掩码阻断学习”。

### 结论

按预注册判据，阶段 0 未通过，且失败模式是 H-026 文末预告的
**C3 泄漏 / 候选可见性**分支，不是“评测仍错”。因此：

- H-026 作为“补掩码后出现 r<h≈随机、r≥h 显著更高阶梯”的假设：
  **REFUTED**；
- 其中“评测漏传 `adj_bias_list`”的子结论：**CONFIRMED（真实 bug，
  已修）**；
- 另发现并修复 `s88` 的“≤t 跳”漏 1 跳 bug（训练/评测同受影响）。
- **不启动 P13 48k**（`_p13_launch.sh` 的冒烟闸门也会拒绝）。
  P12 的 0.50 不能再简单归因于评测；P13 的正式实验需等 H-027
  的严格掩码通过冒烟。

### Loop 恢复（本次已按新阶段配置）

旧 spec 的 `B_adjacency_mask` 指向 `P12_adj_*`（已停在 12k，
`target_step=48000`），直接恢复会永久等待。现改为指向下一步的
`P14_strictmask_*`（H-027，2 seed×6000 步），并把
`auto_loop_state.json` 的 `stage_idx` 置到该阶段；这样 loop 恢复后
不会重复启动 P12，也不会在漏掩码上启动 48k。
