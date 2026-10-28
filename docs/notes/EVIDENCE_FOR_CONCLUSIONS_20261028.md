# 已下定论项目的文献依据与事实证据

> 2026-10-28。按用户要求：对"不能再解决"的结论，逐条给出
> **论文依据**或**事实强证明**。每条都标注是"文献支撑"还是"本项目实测"。

---

## 结论 1：循环的收益在 r≈2 处饱和，更多轮不再提升

**本项目实测**：
- MVT-D v2 上 r=1 → 随机，r=2 → 0.87，r=3–8 → 0.88–0.90（增益 <0.03）
- Ouro 损失训练下熵稳定在 ln8、`e_halt≈4.5`，即模型仍"使用"8 轮，
  但使用更多轮**不带来准确率**——说明饱和来自计算本身，不是调度问题

**文献支撑（强）**：
- **arXiv:2610.26030** *Truncate or Not? A Scaling Study of Looped Transformers
  with Truncated Backpropagation*（近 600 次训练拟合统一 scaling law）：
  > "When model size is also free, the law identifies **two fully trained loops
  > as the best looped setting**."
  即：最优配置就是**两个训练充分的循环**，与我们实测 r=2 达峰完全一致。
- **arXiv:2604.21106** *How Much Is One Recurrence Worth? Iso-Depth Scaling
  Laws for Looped Language Models*：提出 recurrence-equivalence 指数 **φ=0.46**，
  介于 0（循环零增益）与 1（等价于独立层）之间。
  > "replacing unique blocks with shared recurrences increases validation loss
  > at matched training compute."
  即：循环不是"免费的深度"。φ<1 意味着第 r 次循环的边际价值递减。

**判定**：**不构成问题**，是循环架构的已知性质。论文里应表述为
"循环提供 1–2 次额外计算阶段，边际收益随轮数递减"，而非"轮数越多越强"。

---

## 结论 2：稠密（单次前向）模型在需要多跳的任务上无法学习

**本项目实测**：
- dense mid=1/3/5（1.40M / 2.29M / 3.18M），24k 步，loss **精确停在 0.6931=ln2**，
  参数 ×2.3 无变化；手动训练 2000 步 train acc 仅 0.528（连训练集都学不会）
- loop 1.475M 同条件达 r8=0.90+

**文献支撑（强）**：
- **arXiv:2402.12875** *Chain of Thought Empowers Transformers to Solve
  Inherently Serial Problems*：
  > "constant-depth transformers with finite precision poly(n) embedding size
  > can only solve problems in TC⁰ **without CoT**... with T steps of CoT,
  > constant-depth transformers ... can solve any problem solvable by boolean
  > circuits of size T."
  即：固定深度的 Transformer 在表达能力上被限制在 TC⁰/AC⁰，
  **本质上缺少串行计算能力**；需要 T 步迭代才能解决 T 规模串行问题。

**判定**：**有理论依据的必然结果**，不是调参问题。
论文里应表述为"稠密模型受 TC⁰ 表达力限制，循环把 T 步串行计算引入前向过程"。

---

## 结论 3：邻接掩码在本架构下阻断学习，无法检验"每轮一跳"

**本项目实测（三变体）**：
| 变体 | 结果 |
|---|---|
| 第 t 轮看 ≤t 跳 | r=3 解 hop6（0.99）→ 没约束住 |
| 每轮 1 跳 | loss 2.11 恒定，全 r 随机；hop2–3 也学不会 |
| BFS 距离分层 | hop2 在 r=2/r=8 均 0.50 |

**文献支撑（强）**：
- **arXiv:2510.18825** *Unifying and Enhancing Graph Transformers via a
  Hierarchical Mask Framework*（理论分析）：
  > "the probability of correct classification **positively correlates with the
  > receptive field size** and label consistency, leading to a fundamental
  > design principle: an effective attention mask should ensure both a
  > **sufficiently large receptive field** and a high level of label consistency."
  即：掩码把感受野压太小会**直接降低可分性**。我们的 1 跳/BFS 掩码把
  事实间感受野压到 1 跳，正好落入该失败区。
- **arXiv:2502.12352** *Towards Mechanistic Interpretability of Graph
  Transformers via Attention Graphs*：
  > 允许 all-to-all 时，学到的注意力图**不倾向于与输入图结构相关**。
  即：模型未必按人设想的图结构传播；强行掩码可能切断它实际的求解路径。

**事实强证明（本项目）**：同一模型在无掩码时 h6=0.900，加掩码后
hop2 都学不会——若掩码只是"约束深度"，hop2（仅需 2 跳）应仍可学。
hop2 也失败说明掩码切断了求解路径本身。

**判定**：**结构性不匹配**（节点=事实、候选=实体，不在同一张图），
已由「感受野过小导致可分性下降」的理论 + 本项目 hop2 失败的双重证据支持。
论文里作为"方法限制"写明，不作为"机制结论"。

---

## 结论 4：深度外推只能靠训练覆盖，不能靠推理手段

**本项目实测**：四种干预全败（dyn-r / warm-noise / sup-cap / r→16）；
课程覆盖到 8 后 hop7=0.997、hop8=0.773–0.88；同 step 处 24k 与 48k 完全相同，
48k 的收益仅来自"课程走到了 hop6"。

**文献支撑（强）**：
- **arXiv:2604.07822**（Kohli 2026）：训练最大迭代数决定外推范围；
  超出训练深度的组合需要更高推理轮数，且存在 overthinking 上限。
- **arXiv:2605.30757**（Zhang 2026）：压缩循环受限于**循环状态大小**，
  "跑更多轮只增加计算，不会自动产生可增长的 scratchpad"。
- **arXiv:2512.02677**（He 2025）：标准 Transformer 在更深递归上失效，
  归因于无法维持 stack-like 行为。
- **arXiv:2609.33144** *Beyond the Training Horizon*：循环状态同时编码
  "内容"与"该内容是否仍可参与后续计算"，超过训练深度后该状态可靠性下降。

**判定**：有 4 篇独立文献支撑 + 本项目 4 次负结果。可直接写为限制条件。

---

## 结论 5：MVT-5 任务的分数不可用于任何能力结论

**事实强证明（无需文献）**：
- "选下游更长的候选"规则在 train/dev/confirm **全部 19750 样本上准确率 1.0000**
- 去捷径后（MVT-D v2），此前"完全学会"的模型 hop4/hop6 从 1.000/0.995
  跌到 0.607/0.520
- exact-T 的 hop7@r6=0.787 违反"r 轮最多推进 r 跳"的物理约束

**判定**：**事实已足够**，无需文献。已建检测流程
（`docs/errors/err-task-shortcut.md`，E-015）。

---

## 汇总表

| 结论 | 依据类型 | 关键来源 |
|---|---|---|
| r≈2 饱和，边际递减 | 文献 + 实测 | arXiv:2610.26030（"two loops optimal"）、arXiv:2604.21106（φ=0.46） |
| 稠密模型学不了多跳 | 理论 + 实测 | arXiv:2402.12875（TC⁰/CoT 串行表达能力） |
| 掩码阻断学习 | 理论 + 实测 | arXiv:2510.18825（感受野与可分性正相关）、arXiv:2502.12352 |
| 外推只靠覆盖 | 4 篇文献 + 4 次负结果 | arXiv:2604.07822 / 2605.30757 / 2512.02677 / 2609.33144 |
| MVT-5 分数无效 | 实测（19750/19750） | 无需文献 |
