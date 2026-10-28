# Jaume 700M 原生循环 × MoE 训练文档(V6R 核验修订版)

> 版本:V6R,2026-10-09。**状态:纯规划,未启动任何训练,未占用 GPU。**
> 依据:会话 sess_b0feaf27 全程讨论(含父会话 Jaume 主线)+ 对 `TRAINING_PLAN_V6` 的逐条外部核验。
> 取代:V6 的论文证据与架构数字;V6 仍可参考,但其引用与初始化方案以本文为准。
> 标注体系:
> - **[✓]** 已对论文原文 / 官方仓库 / ModelScope config.json 独立核实
> - **[△]** 会话内核查过、本文未独立复核
> - **[✎]** V6 或对话中的错误/偏差,本文已修正(详见 §9)

---

## 0. 先读:V6 甄别结论

V6 引用的 11 篇论文**全部真实存在、arXiv 编号无误**,三篇核心论文(Sparse Layers / Tied Expert / RMoE)的关键数字**全部核实通过**。但 V6 存在 1 处严重内部矛盾、2 处引用失准、1 处数字未核实、1 处可行性缺失,共 9 项,速览:

| # | 严重度 | 问题 |
|---|---|---|
| P1 | **严重** | 初始化自相矛盾:hidden 512→768、16→24 层,却要从 350m 权重"复制/继承"——跨宽度无法复制权重 |
| P2 | 中 | 参数账本不自洽:"active 420–470M" 被低估,按已核实的 vocab=151,647 估算 active ≈500M+ |
| P3 | 中 | Nomic v2 数据量失准:"235M 才稳定获益"混淆了消融与生产(生产用 1.6B pairs) |
| P4 | 小 | router 归属错误:Loop-MoE 原文是**权重共享单 router**,决策自然分化;untie router 收益≈0 |
| P5 | 小 | LARM "4-block 7.6M / WER 26.78" 对照在官方 README 中不存在 |
| P6 | 小 | 漏引直接相关文献 SMELT(2609.01343) |
| P7 | 流程 | 无硬件可行性章节:项目约束 RTX 5060 8GB,700M 全参训练不可行 |
| P8 | 小 | S1 验收门"同 step CE"跨数据配方不可比 |
| P9 | 小 | Jaume "STS22 多语言退化"说法与模型卡终值矛盾,出处(PLAN.md)未能复核 |

---

## 1. 讨论脉络与最终共识(主对话梳理)

会话共 7 轮,演化路径:

| 轮 | 用户意图 | 结论 |
|---|---|---|
| 1 | Jaume 能否加循环深度 | 可行;它是双向 Encoder + MoE,输出句向量,不能照搬 Qwen/LoopUS 方案;推荐 E/M/D 切分 |
| 2 | 澄清:**原生**架构,非后训练 | 原生更优;三种"效率"(参数/token/速度)必须分开,最可信的是参数效率 |
| 3 | 效率=参数效率,去找文献 | 4 个先例:LARM(Encoder 循环)、Universal Transformer、RecursiveGPT、Ouro |
| 4 | **参数量不变、效果变好** | 同参数重分配:省下的层参数投入更宽 hidden / 更大 expert / 更强共享核心 |
| 5 | 目标更正 **700M**;是否先稠密后稀疏 | 不必须完整 dense→MoE;推荐原生 + 短 R=1 warmup + 局部 upcycling |
| 6 | MoE×循环路由混乱 + 小数据过拟合 | 承认风险真实;渐进打开(shared expert→soft routing→top-2)、监控四指标 |
| 7 | **"借鉴已有论文呢?(moe下运用depthloop)"** | 找到 3 篇直接论文:Loop-MoE(2605.09165)、Tied Expert(2606.16825)、RMoE(2408.06793)——本文 §3 详述 |

**最终共识**:循环深度 × MoE 的风险真实存在,但已有论文证明可控可行,核心条件是**跨循环路由分化**(既不塌缩也不漂移);对策是"共享 expert 权重 + 独立 attention + 路由分化作为一等监控指标"。

---

## 2. 底模事实(Jaume-Embedding)[✓]

来源:ModelScope `LightNeko/Jaume-Embedding` 的 config.json(350m-stage3)与模型卡,2026-10-09 独立抓取核实。

| 项 | 值 | 状态 |
|---|---|---|
| 架构 | 双向 Transformer Encoder(BTE),GQA + RoPE(theta 1e6)+ RMSNorm | ✓ |
| 层 / hidden / intermediate | 16 / 512 / 2048 | ✓ |
| 注意力 | 8 heads / 4 KV heads | ✓ |
| FFN | **已是 MoE:4 experts,top-2**(config 原始键名 `expert_num`/`topk`) | ✓ |
| 序列长度 | 8192 | ✓ |
| vocab | **151,647**(Qwen 系词表) | ✓ |
| 输出 | attention pooling → projection(512 维)→ L2 norm | ✓ |
| Matryoshka dims | [128, 256, 384, 512] | ✓ |
| 负载均衡 | `load_balance_weight = 0.01` | ✓ |
| 官方 checkpoint | 350m-mlm / mlm-v2(加 6 语种)/ stage2(±ml)/ stage3(±ml)/ decision-v2 | ✓ |
| 指标 | stage3:STS-B 0.6223@512D、STS22 0.3554;stage3-ml:STS-B 0.6214、STS22 0.3581 | ✓ |
| decision-v2 | choice acc 0.9975(ECE 0.0017)、noul AUROC 0.837;score 头 Pearson 0.29(弱) | ✓ |
| 700M | 模型卡明确:700M **存在但未发布**("is not included here") | ✓ |

**[✎] 两处修正**:
1. V6 称"英文对比训练导致 STS22 多语言退化(官方 PLAN 记录)"——模型卡终值显示 stage3-ml 的 STS22(**0.3581**)> stage3(0.3554),退化说法与终值矛盾;若指中间实验,需回看仓库 PLAN.md 原文(本次抓取失败)。**在 PLAN.md 复核前,不得把"多语言退化"当作已确认事实。**
2. V6 的"Stage4 Matryoshka → Stage5 decision"五阶段枚举是简化;实际阶段为 stage2/ stage3 / decision-v2,Matryoshka 是输出特性而非独立阶段。

**关键事实**:Jaume 本身就是 MoE 模型。问题不是"要不要加 MoE",而是"已有 4-expert MoE 如何与循环深度共存而不路由塌缩"。

---

## 3. 论文证据库(逐篇核验)

### 3.1 Sparse Layers are Critical to Scaling Looped Language Models(arXiv:2605.09165)[✓]

Loop-MoE 的直接主证,Lee/Biloki/Hu/May,2026。

- **dense looped 模型 scaling 不如标准基线;换成 sparse MoE 后 Loop-MoE scaling 反超标准基线**。机理:同一物理层在不同循环中选择不同专家,跨循环路由分化恢复了共享参数损失的表达力。✓
- 配置:**E=8,k=2(仿 Mixtral)**;Switch 式 load-balance + ST-MoE 式 router z-loss。✓
- 跨循环 expert overlap 实测:layers 1–6、8 上 **25–53% 的 token 两圈专家完全不重叠,仅 4–14% 完全相同**;layer 7 例外(37% 完全相同、仅 3% 零重叠)。✓(对话漏掉了 layer 7 例外,本文补上)
- **[✎] router 归属修正**:原文是**权重共享的单 router** 跨圈复用,只是输入状态不同导致决策分化——**不是**"每个循环层保留独立 router"。这反而更支持轻量设计。
- loop boundary 是天然 early-exit 点。✓

### 3.2 Tied Expert Layers in MoE LMs(arXiv:2606.16825)[✓]

回答"哪些组件能共享、哪些不能"。

- 共享 expert FFN 权重(跨相邻层 tie),**attention 与 router 保持每层独立**;评测 OLMoE / Qwen3-MoE / DeepSeekMoE,含 OLMoE-1B-7B 全量。✓
- 参数节省:g=2 → 29–35%,**g=4 → 43–52%**(≈V6 所说"约 50%"✓),摘要口径"内存近乎 2×"。✓
- **2+2 untied prelude/coda"压倒所有其他架构选择"(Finding 4)**——首尾各 2 层不共享是最重要设计。✓
- ablation:**untie router 收益≈0(loss 差 ≤0.009),untie attention 收益 0.030–0.058**——原文结论"per-layer attention, not routing, is what differentiates layers"。✓
- 注意(本文补充的适用性限定):该文是**标准深度下的层间 tie**,不是循环执行场景;迁移到循环核心属合理外推,不是同设定证据。

### 3.3 RMoE:Layerwise Recurrent Router(arXiv:2408.06793)[✓]

- 普通 MoE 每层 router 独立决策 → 路由不协调、expert 多样性不足;RMoE 用 **低维投影 + GRU 维护跨层路由状态**(`r_i = GRU(Proj_i(x_i), r_{i-1})`),超过普通 SMoE,开销小。✓(ICLR 2025)
- 迁移方式(本文设计,非原文):把"跨层状态"改成"跨圈状态"。

### 3.4 Nomic Embed v2(arXiv:2502.07972)[✓,含 ✎ 修正]

Embedding-MoE 的直接先例。

- **数据量是硬前提**:100M pairs 时 MoE 持续劣于参数相当的 dense(原文引 Krajewski et al. 2024);235M pairs(单语消融)时 MoE 反超;**生产训练集为 1.6B pairs**。✓
- **[✎] 修正 V6**:V6 写"235M pairs 才稳定获益"并规划"≥200M"——把消融值当成了充分条件。正确口径:**≥200M 是消融下限;Nomic v2 生产级用了 1.6B**。S5 数据规划按此双口径写。
- 结构:base 为 XLM-R Base + RoPE(mNomic-BERT),**隔层(alternating)转 MoE,8 experts、top-2,从第 2 层开始**;ablation:只转 6 层 > 全部 12 层。✓
- 规模参照:475M total / 305M active,输出 768 维(256 Matryoshka)——700M Jaume 的同量级参照物。

### 3.5 原生循环 Encoder 侧

- **LARM**(arXiv:2606.04678,Idiap)[✓ 定性 + 主要数字]:稀疏 CTC checkpoint + supervision-clock embedding + **FiLM 深度条件** + 延迟软后验反馈。100h LibriSpeech test-clean(greedy):**LARM d=384(7.7M)WER 11.34 < 16-block(28.9M)14.43 < 48-block(85.7M)12.24** 之外——即 7.7M 循环模型超过 28.9M、9.58(d=768 版,28.9M)甚至超过 85.7M;960h 同样领先。✓
  **[✎]**:对话中"4-block 7.6M / WER 26.78"这条对照在官方 README 中**不存在**(README 基线只有 16/48-block),引用前必须查论文正文,否则删除。V6 未引用该数字,不受影响。
- **Universal Transformer**(arXiv:1807.03819)[✓]:共享 block 循环 + ACT 逐位置停机,同参数优于普通 Transformer——原生循环 + 动态深度的谱系源头。
- **RecursiveGPT**(*Squeezing More from Limited Data with Recursive Transformers*,arXiv:2608.26973)[✓ 数字精确]:10M words 平均分 **45.80(27.6M 循环)> 45.07(28.7M+factorized emb)> 44.93(41.1M 标准)**;100M words 时 124.0M 循环模型(61.92)与 1.22B 标准(63.47)竞争、BLiMP 反超(80.06 vs 79.71);ALBERT 式 factorized embedding 是关键组件(去掉后 46.16→44.90)。结论:**有限数据下应把表示容量与每 token 计算解耦**。
- **Ouro / LoopLM**(*Scaling Latent Reasoning via Looped LMs*,arXiv:2510.25741)[✓]:1.4B/2.6B 原生循环预训练(7.7T tokens),证明循环应从预训练建立、熵正则训练 exit gate。

### 3.6 稀疏化初始化

- **Sparse Upcycling**(arXiv:2212.05055)[✓]:从 dense checkpoint 复制初始化 experts;load balance 过小死专家、过大压主任务。
- **Upcycling LLMs into MoE**(arXiv:2410.07524)[✓]:virtual group 初始化、weight scaling、softmax-then-topK;upcycling 不必然优于从零,取决于预算。
- **前提警示(本文补充)**:两篇 upcycling 的复制都发生在**同结构内**;跨 hidden 宽度/跨层数复制权重不在其证据范围内(直接引出 §9 P1)。

### 3.7 V6 漏引

- **SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers**(arXiv:2609.01343)[✓ 存在]:MoE 循环 Transformer 的 compute-matched scaling law,"looping middle layers in sparse MoE improves training efficiency while matching per-token FLOPs"——与本项目主案同设定,做 S2/S3 对照设计前应精读。

---

## 4. 核心风险 → 对策映射

| 风险 | 证据 | 对策(落点) |
|---|---|---|
| 跨循环路由塌缩(每圈选同一批专家→循环退化为重复计算) | 2605.09165 | cross-loop overlap 列为一等监控(§7);深度条件 + 跨圈路由状态 |
| 专家死/过载、马太效应 | upcycling 系列 | shared expert 保底 + load-balance(起步 0.01,与 Jaume 现值一致)+ z-loss |
| 小数据 MoE 劣于 dense | Nomic v2(100M 失败) | S5 数据量双口径:≥200M 下限 / 生产级 1.6B 目标 |
| 共享范围错误(全 tie 或 tie 错组件) | Tied Expert | 只共享 expert FFN;attention/norm 独立;2+2 首尾不共享 |
| 路由决策孤立、无历史 | RMoE | GRU 跨圈状态(或圈数 embedding 轻量版) |
| 退出与路由梯度冲突 | V5 实测(硬切 +0.273 CE) | adaptive exit 放最后(S7),soft decay,不与 MoE 联训 |
| 复读/漂移 | V5 L1-v2 | 第 1 圈恒等,第 2 圈起 h0 再注入 |

---

## 5. 架构方案(V6R 修正版)

### 5.1 E/M/D 切分(保留 V6 骨架)

```
E 段:L0–5,6 层,dense FFN,不循环
M 段:L6–17,12 层物理结构,循环 R∈{1..8}
      FFN = shared expert(1)+ routed experts(4, top-2)
      attention / norm per-layer 独立(Tied Expert)
D 段:L18–23,6 层,dense FFN,不循环
深度条件:FiLM(圈数 embedding 生成 scale/shift)作用于 M 段 attention 输出(LARM)
第 1 圈恒等;第 2 圈起 h + h0(V5 L1-v2)
```

设计依据:Tied Expert 的 2+2 首尾原则(此处 6+6 更保守)、Nomic v2 只转中间层、2605.09165 的 MoE-only-in-loop 主案。

### 5.2 router 设计(修正表述后)

- **每物理层一个 router,权重跨圈共享**——2605.09165 原文做法,决策靠输入状态自然分化;Tied Expert 证明 untie router 收益≈0,**不值得为每圈独立 router 付参数**。**[✎ P4]**
- router 输入拼接**圈数 embedding**;S4 起可选加 GRU 跨圈状态(RMoE 式外推):`r_t = GRU(Proj(h_t), r_{t-1})`。
- 损失三件套:load-balance(系数 0.01 起步,Jaume 原值)+ router z-loss(1e-3)+ softmax-then-topK。

### 5.3 参数核算(必须先跑脚本,本文给出公式)[✎ P2]

vocab 已核实为 151,647,可精确计算。设 hidden=H、intermediate=I:

```
attention/层 ≈ 4H²(GQA 8:4 时 q,o=H²;k,v=H²/2)
dense FFN/层 = 3HI
MoE 存储/层(M 段)= 5 × 3HI(shared 1 + routed 4)
active/层(M 段)= 3HI × (1+2) + attention   # shared + top-2
embedding = 151647 × H(输出头若不 tie 则 ×2)
```

按 V6 原案(H=768, I=3072)粗估:总存储 ≈671M(tie 输出头)~787M(不 tie),**active ≈500M(tie)~615M(不 tie)——V6 的"active 420–470M"明显低估**。H/I/层数需由核算脚本定稿,再回填本节;若需压 active,优先减 M 层数或 expert 数,不动 E/D。

### 5.4 初始化(P1 修正:三选一,不得含糊)

V6 的"从 350m 复制权重"与 hidden 512→768 冲突,**三选一**:

- **方案 a(推荐)**:保持 H=512、I=2048 不变,把 350m 的 16 层直接映射进 E(取前 6)/M(取中 12 中的 8?)/D(取后 6)——纯结构重组,权重可完整继承;700M 由加深 M 段物理层与扩 expert 达成;
- 方案 b:宽度生长手术(如 Net2Net 式 H 512→768)+ 文档写明手术细节与验证(前后输出 KL 对齐);
- 方案 c:放弃继承,从零训练(预算最高,不推荐作为首选)。

选定后写入 §9 结论表再进 S1。

---

## 6. 训练阶段(S0–S7,修正后)

沿用 V5 转折点存档协议 + atomic checkpoint + append-only 结论表。**任何阶段启动须用户明确指令。**

| 阶段 | 内容 | 修改点 | go/no-go 门 |
|---|---|---|---|
| S0 | 基线冻结:350m 官方权重跑 MTEB/STS22 基线,核实各阶段 ckpt 指标 | 无 | 产出 `results/v6r_s0_baseline.json` |
| S1 | dense-like warmup(R=1,700M 结构,权重继承) | **初始化按 §5.4 定案** | MLM CE(固定验证集)**[✎ P8:弃"同 step"口径]** ≤ 350m 的 110% |
| S2 | 原生循环 dense(对照组):R∈{1,2,4,8}+LoopUS B=20/K=5+FiLM | 精读 SMELT 后定对照预算 | 记录 CE(d) 曲线;预期劣化(2605.09165 前提复现) |
| S3 | M 段开 MoE(shared+4 routed top-2,upcycling 初始化) | — | CE(最优 d) 优于 S2;无死专家(load≥1%) |
| S4 | router 加圈数 embedding(+可选 GRU 跨圈状态) | — | cross-loop overlap 显著低于 S3 且 CE 不劣化;不降则换 per-loop router head(备选) |
| S5 | 多语言大规模对比(MNRL/hard neg/Matryoshka) | **数据双口径 [✎ P3]:≥200M 下限,1.6B 为生产目标** | MTEB 多语言 ≥ 350m 官方 95%;Matryoshka 各维不塌 |
| S6 | decision 微调 | — | choice acc ≥0.99;noul AUROC ≥0.83 |
| S7 | adaptive exit(v4-c soft decay,最后启用) | — | 策略损伤=0;平均深度 ≤50% 时质量损失 ≤1% |

---

## 7. 监控指标(S3 起全程落盘)

| 指标 | 定义 | 越界动作 |
|---|---|---|
| **cross-loop overlap** | 同 token 相邻圈 top-2 集合 Jaccard, batch 均值 | >0.8 持续 → 触发 S4 改造;参照 2605.09165 实测区间 0.2–0.6 |
| expert load | 每 expert token 占比 | <1% 连续 1k step → 存档 + 提 aux 系数 |
| router entropy | batch 级分布熵 | <0.3× 初值 → 塌缩预警 |
| per-depth routing | 圈×expert 分布矩阵 | 某圈与其他圈相同 → 该圈退化标记 |
| CE(d) / 对比损失(d) | 深度条件曲线 | d 增大不降 → 深度监督复查 |
| Matryoshka 各维 | 截断维度质量 | 低维塌 → 加权复查 |
| 生成探针 | 协议 v1.1 段界 + 复读率 | rep4>4 → 复查 h0 注入 |

---

## 8. 对照与验收(四组,同数据同预算同协议)

| 组 | 架构 | 角色 |
|---|---|---|
| C0 | Jaume 350m 官方权重 | 外部基线 |
| C1 | 700M dense MoE(R=1) | 参数量基线(S1) |
| C2 | 700M native loop dense(S2) | 复现"dense looped 劣化" |
| C3 | 700M native loop MoE + 深度条件 router(S4) | 主案 |

**成功判定**(全部满足):C3≥C1;C3>C2;C3 多语言 ≥C0 的 95%;overlap 落在 0.2–0.6;S7 深度减半质量损失 ≤1%。任一不满足如实记负结果。

---

## 9. V6 问题清单(甄别详情)

| # | 位置 | 问题 | 核验证据 | 本文处置 |
|---|---|---|---|---|
| P1 | §4.1 vs §4.2/S1 | hidden 512→768、16→24 层 vs "从 350m 复制/继承权重"——跨宽度跨层数不可复制,upcycling 论文前提是同结构 | config.json 实测 350m 为 512 宽/16 层/4 专家 | §5.4 三选一定案 |
| P2 | §4.1 | "active 420–470M"低估;vocab 151,647 下估算 active ≈500M+(tie 头) | config.json vocab 核实 | §5.3 公式+强制脚本核算 |
| P3 | §3.1 表/§5 S5 | "235M pairs 才稳定获益"/"≥200M 规划"混淆消融与生产 | Nomic v2 原文:100M 失败、235M 单语消融反超、生产 1.6B | §3.4/§6 双口径 |
| P4 | §3.2/§4.2 依据 | 把"每循环独立 router"当 Loop-MoE 做法 | 2605.09165 原文:权重共享单 router;Tied Expert:untie router ≤0.009 | §5.2 修正 |
| P5 | (对话 item 2917/2919,V6 未引) | LARM "4-block 7.6M / WER 26.78" 无出处 | 官方 README 基线仅 16/48-block | §3.5 标注禁引 |
| P6 | 附录索引 | 漏引 SMELT(2609.01343,2026-09,同设定 scaling law) | 检索证实存在 | §3.7 补入 |
| P7 | 全文 | 无硬件可行性章节;项目约束 RTX 5060 8GB(V5 为此全程 NF4+LoRA+冻结),700M 全参 MLM/对比训练在该卡不可行 | V5 项目史 | 本节警示:S1–S5 需显存预算方案(LoRA/分阶段冻结/梯度检查点)或外部算力,**开工前必须补** |
| P8 | §5 S1 门 | "CE 不差于 350m 同 step 的 110%"——跨数据配方 step 不可比 | — | §6 改固定验证集 |
| P9 | §1 表 | "英文对比训练致 STS22 多语言退化(官方 PLAN 记录)"与模型卡终值矛盾(ml 版 0.3581>0.3554) | 模型卡实测;PLAN.md 抓取失败 | §2 标注待复核,不作既定事实 |

---

## 10. 开工前核验清单

1. [ ] 抓取仓库 `PLAN.md`/`SPEC.md` 原文,复核"多语言退化"表述(P9);
2. [ ] 参数核算脚本跑通,回填 §5.3 并定稿 H/I/层数/expert 数(P2);
3. [ ] 初始化方案 a/b/c 定案(P1);
4. [ ] 精读 SMELT(2609.01343)与 2605.09165 全文,确认 S2/S3 预算与 overlap 区间(P6);
5. [ ] 补硬件/显存可行性章节(P7);
6. [ ] 用户确认后才能启动任何训练(沿 V5 惯例)。

---

## 附:论文依据索引(全部核实存在)

| arXiv | 简称 | 核验状态 |
|---|---|---|
| 2605.09165 | Sparse Layers / Loop-MoE | ✓ 编号+关键数字 |
| 2606.16825 | Tied Expert Layers | ✓ 编号+关键数字 |
| 2408.06793 | RMoE | ✓ 编号+结论 |
| 2502.07972 | Nomic Embed v2 | ✓ 编号+数字(含修正) |
| 2606.04678 | LARM | ✓ 编号+定性+主数字(一条对照未核实) |
| 1807.03819 | Universal Transformer | ✓ |
| 2608.26973 | RecursiveGPT | ✓ 编号+数字精确 |
| 2510.25741 | Ouro / LoopLM | ✓ |
| 2212.05055 | Sparse Upcycling | ✓ |
| 2410.07524 | Upcycling into MoE | ✓ |
| 2609.01343 | SMELT(V6 漏引,本文补) | ✓ 存在 |
