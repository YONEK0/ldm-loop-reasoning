# 假说索引（HYPOTHESES INDEX）

> **三态命名约定**（文件名体现状态，便于路由）：
> - `H-xxx-PENDING-<slug>.md` —— 待验证
> - `H-xxx-CONFIRMED-<slug>.md` —— 已证实
> - `H-xxx-REFUTED-<slug>.md` —— 已证伪
>
> 状态变更时**重命名文件**并更新本索引。证伪的假说保留，因为它们排除了搜索空间。
>
> **编号唯一性教训（2026-10-28）**：H-026/H-027 曾被两方同时使用——
> overnight 迭代链（`H-026-REFUTED-adj-eval-mask` / `H-027-PENDING-strict-distance-mask`，
> 09:58/10:01 建立）与当日人工假说（原 H-026 决策模块 / 原 H-027 优化停滞）。
> 现按建立时间裁定：**H-026/H-027 归迭代链**，人工假说改号为 **H-028/H-029**。
> 今后新假说必须先查本表最大编号再分配。

---

> **⚠️ 2026-10-27 重大更正**：MVT-5 任务存在**下游链长捷径**（干扰链恒比答案链短 1，
> 使"选更长的候选"在全部 19750 个样本上 100% 正确，见 H-023）。
> **涉及 MVT-5 任务语义的结论（H-001/H-003/H-011/H-013/H-014/H-022）全部需要重验**；
> 现有模型在无捷径任务 MVT-D 上准确率 = 随机（0.52–0.55）。
> 调查报告：`../notes/SHORTCUT_INVESTIGATION_20261027.md`
>
> **替代任务已就绪**：MVT-D v2（`s84_dual_task.py`）消除了全部局部捷径
> （pick_longer=0.3305，两候选结构对称）。首轮结果见 H-024。

## 当前假说表

| 编号 | 状态 | 假说（一句话） | 关键证据 | 文件 |
|---|---|---|---|---|
| **H-029** | **待验证** | 失败 seed 的根因是**优化停滞**（训练分布没拟合上），不是外推能力不足 | 13 个 48k run：task loss <0.15 → 8/8 个 h7≥0.80；≥0.21 → 5/5 个 <0.80。P19 的 s53 把 task loss 从 0.2102 压到 0.0031，h7 随之 0.695→0.805 | [H-029-PENDING-optimization-stall.md](H-029-PENDING-optimization-stall.md) |
| **H-028** | **待验证（单 seed 已命中证实档）** | 决策模块的 193 参数线性读出是 hop6 收敛的瓶颈之一 | seed 17 四臂：linear h6=0.80/h7=0.78；**mlp h6=0.97/h7=0.875**（无跷跷板，hop2–5 全 ≥0.99）；bilinear 0.735/0.51；xattn loss 精确停在 ln2。需 ≥3 seed 复验 | [H-028-PENDING-decision-head-capacity.md](H-028-PENDING-decision-head-capacity.md) |
| **H-027** | **待验证** | 用"距离前沿"严格掩码（事实只看 dist=t−1、候选只看 dist≤t、prelude 用 t=0、coda 用末轮）可强制每轮一跳 | eval-only 探针：换严格掩码后 r=1..4 全塌到 0.32–0.49，说明旧解依赖"每轮多跳"可达性。代码未实现，判据已跑前写定 | [H-027-PENDING-strict-distance-mask.md](H-027-PENDING-strict-distance-mask.md) |
| **H-026** | **证伪（但子结论成立）** | 邻接掩码的 dev 恒 0.50 源于**评测漏传 `adj_bias_list`**；补上后应出现 `r<h≈随机、r≥h 显著更高` 阶梯 | 漏传确认为真 bug（关掩码即塌回随机：r=3 with 0.987 / no 0.466）；但补修后 **r=3 已解 hop6=0.985**，C3 泄漏判据 FAIL，阶梯预测被证伪 | [H-026-REFUTED-adj-eval-mask.md](H-026-REFUTED-adj-eval-mask.md) |
| **H-025** | **证伪（检验手段不成立）** | 用邻接掩码检验"每轮一跳"在本架构可行 | 三变体实测：t跳/轮 r=3 解 hop6（未约束）；1跳/轮 loss 2.11 恒定全随机；BFS距离分层 hop2 在 r=2/r=8 均 0.50。根因：节点=事实、候选=实体，不同图 | [H-025-REFUTED-mask-test-infeasible.md](H-025-REFUTED-mask-test-infeasible.md) |
| **H-024** | **证实** | 无捷径任务上存在真实循环依赖与深度外推；Ouro 损失更优 | persist：三 seed **r=1 全随机**（0.494–0.508），s71 r=2=0.868、hop7=0.830/hop8=0.785；**Ouro 20k 即 hop7=0.85/hop8=0.88、r=1 从 0.51 升到 0.797** | [H-024-CONFIRMED-clean-loop-dependence.md](H-024-CONFIRMED-clean-loop-dependence.md) |
| H-023 | **证实** | MVT-5 存在"下游链长"捷径，且模型确实在用 | "选更长候选"规则 train/dev/confirm 全 1.0000（19750/19750）；exact-T hop7@r6=0.787 但 r=6<7 跳物理不可能；hop8@r6=0.469 分界在链长处 | [H-023-CONFIRMED-downstream-length-shortcut.md](H-023-CONFIRMED-downstream-length-shortcut.md) |
| H-022 | **证伪** | 深度可插值：训练 {2,4,6,8} 应能解未见的 {3,5,7} | 三 seed 未训练深度均值 hop3=0.373/hop5=0.417/hop7=0.350（≈或低于随机）；**s17 训练 hop6=0.995、hop8=0.984 全学会，夹中间的 hop5=0.205、hop7=0.091 远低于随机**；step_head 首跳命中 0.20–0.67 vs 训练深度 0.82–1.00；content_follow=1.000 排除位置偏置 | [H-022-REFUTED-depth-interpolation.md](H-022-REFUTED-depth-interpolation.md) |
| H-021 | **证伪** | 增加每轮层数 L 能提升可学习深度 | L=1/2/4 在课程到 8 时 hop7 均 0.97–1.00；L=4 参数 ×1.9 无优势 | [H-021-REFUTED-L-depth-capacity.md](H-021-REFUTED-L-depth-capacity.md) |
| H-020 | **待验证（强证据）** | persist 的"保持"监督把后期轮次训成守成，主动封死外推 | exact-T hop8 随 r 6→16 升 0.533→0.731；persist 平坦 0.501→0.512 | [H-020-PENDING-hold-blocks-extrapolation.md](H-020-PENDING-hold-blocks-extrapolation.md) |
| H-015 | **证伪** | confirm 集偏置导致 hop7 低于随机 | 诊断实测：位置偏好 0.496、候选对调内容跟随 1.000 —— 无偏置，外推失败是真实模型局限 | [H-015-REFUTED-confirm-bias.md](H-015-REFUTED-confirm-bias.md) |
| H-014 | **证伪** | persist+step 配方可外推到 hop7/8 | 3 seed confirm_r8=0.44–0.47，全程 0.41–0.54 无 >0.6 的点；hop7 反低于随机 | [H-014-REFUTED-extrapolation.md](H-014-REFUTED-extrapolation.md) |
| H-013 | **证实** | **中间路径监督（step-sup）跨越 hop5 冷启动门槛** | **3/4 必败种子被救回（s71=1.000），总成功率 4/8→7/8** | [H-013-CONFIRMED-step-supervision.md](H-013-CONFIRMED-step-supervision.md) |
| H-012 | **证伪** | 细化课程（回采 cap-reach）可跨过 hop5 门槛 | 4 个失败种子峰值 0.763–0.808，hop5 仍 0.495–0.56，无一突破 | [H-012-REFUTED-cap-reach.md](H-012-REFUTED-cap-reach.md) |
| H-011 | **证实** | **多轮保持监督（persist-sup）解决轮次竞争：从第 R 轮起每轮都要答对** | **3/6 seed 五级全解（1.000/0.999/0.993），归零事件 5→0；dense 同条件仍解不出 hop5/6** | [H-011-CONFIRMED-persist-supervision.md](H-011-CONFIRMED-persist-supervision.md) |
| H-010 | **证伪** | 课程硬切换（cap 阶梯）触发遗忘 | 三种 mix（0/0.3/0.6）峰值 0.671/0.664/0.664，无改善；渐进版峰值反而更早（2k） | [H-010-REFUTED-curriculum-shift.md](H-010-REFUTED-curriculum-shift.md) |
| H-009 | **待验证（弱支持）** | 每 hop 独立读出头可解 H-007（读出层分支） | 峰值 +0.089（0.671→0.760）、hop4 0.815→0.90；但 hop5/6 未达标、跷跷板仍在。**注意：该实验跑在 MVT-5 上（有 100% 捷径），结论不可用，需在 MVT-D v2 重做** | [H-009-PENDING-per-hop-head.md](H-009-PENDING-per-hop-head.md) |
| H-008 | **证伪** | 加宽模型（d×2，参数×3.9）可解 H-007（容量不足分支） | 服务器实测峰值 0.669 vs 基线 0.671，零收益且跷跷板更剧烈 | [H-008-REFUTED-wider-fixes.md](H-008-REFUTED-wider-fixes.md) |
| H-007 | **待验证（收窄）** | 根因是相邻 hop 的轮次索引竞争，**主要发生在共享 core 状态更新路径**（读出层为次要耦合点） | hop5=0.0 同时 hop6=1.0（seed17+EMA@22k）；hop2=0.005 同时 hop3=0.985（seed31+EMA@8k）；服务器基线同样出现 | [H-007-PENDING-round-index-competition.md](H-007-PENDING-round-index-competition.md) |
| H-006 | **证伪** | 参数漂移导致遗忘，EMA 可解 | EMA 救 seed53（0.49→0.86）但毁 seed17（全解→0.691），净效果为零 | [H-006-REFUTED-ema-fixes.md](H-006-REFUTED-ema-fixes.md) |
| H-005 | **证伪** | 训练失败源于学习率过高 | 1e-3/5e-4/3e-4 全失败，3e-4 最差（0.176） | [H-005-REFUTED-lr-too-high.md](H-005-REFUTED-lr-too-high.md) |
| H-004 | **证伪** | 训练失败源于残差爆炸（residual explosion，arXiv:2605.18797） | 成功/失败残差增长 6.78× vs 6.63×，无区分度；增长最小的 seed17 最成功 | [H-004-REFUTED-residual-explosion.md](H-004-REFUTED-residual-explosion.md) |
| H-003 | **证实** | 1× 参数的 loop 优于同参/1.7× 稠密臂（参数效率） | loop 1× 解 hop5/6=0.99/0.96；dense 1×/1.7× 停在 0.5 | [H-003-CONFIRMED-param-efficiency.md](H-003-CONFIRMED-param-efficiency.md) |
| H-002 | **证实** | NoPE（无位置编码）是循环承重的必要条件 | NoPE 双 seed 4/5+5/5；sin 编码双 seed 0/5 | [H-002-CONFIRMED-nope-necessary.md](H-002-CONFIRMED-nope-necessary.md) |
| H-001 | **已撤销（待重验）** | 循环轮次承载串行深度：需要 T 跳的样本在第 T 轮答对（对角线） | **原证据受任务捷径污染**：阶梯可用"链长 ≤ r 才能比完"解释；去捷径后 s17 的 hop4/hop6 从 1.000/0.995 跌到 0.607/0.520 | [H-001-REVOKED-loop-carries-serial-depth.md](H-001-REVOKED-loop-carries-serial-depth.md) |

---

## H-025 的重要限定（2026-10-28 由 H-026 补充）

H-025 原文把"1 跳/轮 loss 2.11 恒定、全随机"作为"掩码阻断学习"的证据。
H-026 的冒烟显示**该判定只对"每轮恰好 1 跳"这一变体成立**；
而"≤t 跳可达"变体（P12/P13 所用）**训练侧是能学的**
（3000 步末 task loss 0.236 / 0.005），问题在于**泄漏**
（r=3 即解 hop6=0.985），不是学不动。

因此邻接掩码的失败模式应表述为两类，不可混为一谈：
1. **过严（恰好 1 跳/轮）** → 阻断学习，loss 恒定；
2. **过松（≤t 跳可达）** → 能学但泄漏，"每轮一跳"的物理约束不成立。

二者共同指向同一结论：本架构下（融合编码使一个 token 携带一条边的两端，
一层注意力可核对两条边）**掩码粒度无法同时满足"可学"与"约束单跳"**。

---

## 判定标准（预注册，防事后解释）

- **证实**：至少 2 个独立 seed 或 2 种独立测量方式支持，且无相反证据。
- **证伪**：至少 1 次预注册的对照实验给出相反结果，且该实验本身经过复现。
- **待验证**：有支持性观察但样本不足，或仅有单一测量方式。

**禁止**：把"看起来合理"写成证实；把"一次跑出来"写成可复现。

## 证伪记录的价值

H-004 / H-005 / H-006 虽被证伪，但各自排除了一个主流方向（残差、超参、参数漂移），
把搜索空间缩小到架构层面（H-007 系列）。论文中应作为"系统排除"呈现。
