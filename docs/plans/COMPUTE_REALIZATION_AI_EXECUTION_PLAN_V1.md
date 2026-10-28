# 算力兑现线：AI 执行计划 V1

> 配套契约：`docs/plans/COMPUTE_REALIZATION_EXECUTION_CONTRACT_V1.md`
> 契约索引：C-01–C-11；阶段索引：E0–E5。
> 目标模型：Qwen3.5-0.8B-Base
> 当前授权：只授权本算力兑现线；不新增训练，除非后续明确授权。
> 当前状态：计划已写，执行尚未启动。

---

## 1. 给执行 AI 的总指令

你是本任务的执行 AI。开始前必须读取：

1. 本文件；
2. `docs/plans/COMPUTE_REALIZATION_EXECUTION_CONTRACT_V1.md`；
3. `docs/plans/TRAINING_PLAN_V7_0P8B.md` V5；
4. `docs/plans/S08_CAMPAIGN_LOG.md` §13–§15；
5. `results/s31_s08_hard_exit_gate.json`；
6. `results/s33_s08_seq_adaptive_gen.json`；
7. `s34_gdn_step_fix.py`。

不得把历史结果当作本次执行已通过的门禁。每个阶段必须产生本阶段结果文件，读到结果后再决定是否进入下一阶段。任何正确性门失败、显存异常、输出乱码、外部游戏/GPU 负载进入正式计时，都必须暂停并写问题报告，不得继续跑正式矩阵。

本计划不要求重新训练。真实 token 级 skip 的失败原因已归档；优先复用已验证的 GDN 单步修复和序列级自适应路径。

---

## 2. 固定资产与不可替换输入

### 2.1 模型资产

| 资产 | 路径 | 用途 |
|---|---|---|
| 非循环臂 A checkpoint | `runs/s08_armA/ckpt_latest.pt` | 质量/性能对照 |
| 循环臂 B checkpoint | `runs/s08_armB/ckpt_latest.pt` | 固定深度对照 |
| Q1 router checkpoint | `runs/s08_qat_exit/ckpt_latest.pt` | 序列自适应/Phase E |
| GDN 单步一致性补丁 | `s34_gdn_step_fix.py` | 增量推理必须加载 |
| tokenizer | 本地 Qwen3.5-0.8B snapshot | 所有模式固定同一 tokenizer |

### 2.2 结构契约

```text
24 layers
E = [0, 4)     # 4 layers
M = [4, 16)    # 12 layers
D = [16, 24)   # 8 layers
B_max = 8
full attention in M: physical layers 7, 11, 15
```

实际层调用代理：

```text
base single-pass = 4 + 12 + 8 = 24
fixed d           = 12 + 12*d
fixed d2          = 36
fixed d8          = 108
sequence d*       = 12 + 12*d*
```

这些是 layer-call 代理，不等于 FLOPs。GDN/全注意力/FFN/KV/补投影要单独计数。旧结果中的 `20+12d` 公式不再使用。

### 2.3 数据与质量资产

- 数据 manifest：`data/gsm8k/s08/manifest.json`
- 质量协议：`scorer_strict.py` v1.1
- 既有对照：
  - arm A strict 21/40，完成 36/40；
  - arm B fixed-d2 strict 22/40，完成 37/40；
  - sequence-adaptive 已有结果：strict 21/40，完成 36/40，d*=2/3。
- GSM8K test 不得用于本线的阈值选择、模式选择或调参。

---

## 3. 阶段 E0：环境与资产门禁

### 3.1 任务

1. 记录 GPU 型号、显存、CUDA、PyTorch、Transformers、Python 版本。
2. 记录模型 snapshot、checkpoint、脚本和补丁的 SHA256/hash。
3. 确认没有游戏或其他应用占用独显进行正式测试。
4. 确认 `runs/s08_armA/ckpt_latest.pt`、`runs/s08_armB/ckpt_latest.pt`、`runs/s08_qat_exit/ckpt_latest.pt` 可加载。
5. 启动只读 dashboard，不能因 dashboard 启动而改变模型设备或计时配置。

### 3.2 命令模板

```bash
cd /e/develop/project_files/llm-test
nvidia-smi
HF_HUB_OFFLINE=1 loopus_env/Scripts/python.exe -m py_compile \
  s34_gdn_step_fix.py s33_patched.py s26_s08_eval.py
loopus_env/Scripts/python.exe dashboard.py --port 8765 --run <run-name>
```

### 3.3 产物

- `results/e0_compute_env.json`
- `results/e0_asset_manifest.json`
- `runs/e0_compute.log`

### 3.5 E0 阻断项

以下任一项失败，不得进入 E1：

- 解释器不是 `loopus_env` 或 Transformers 不是项目锁定的 5.3.0；
- checkpoint/代码 hash 与 manifest 不一致；
- 检测到游戏或其他独显负载；
- cache API 与 `s34_gdn_step_fix.py` forward 签名不匹配；
- 旧脚本中的 `physical_layer-8` 映射仍被复用（本模型 M 起点为 4）；
- E1 测试没有完整 E/M/D prefill、首 token、后续多步对照。

E0 产物中必须标注历史结果 `legacy_pending_revalidation`，尤其是旧 61.7% 节省值。



### 4.1 目标

确认 GDN 单步补丁和官方 cache 分区在固定深度下可用，才能开始计时。优先测试 d=2，因为它是质量和生成对照的主深度；随后测试 d=8。

### 4.2 实现要求

- prompt 必须完整 prefill；
- **prefill 和 decode 使用相同的固定 d 语义**：d2 测试必须 d2 prefill + d2 decode，d8 测试必须 d8 prefill + d8 decode；不得 soft/全圈 prefill 后接固定 d decode；
- 首个生成 token 必须来自 prefill 最后位置 logits，不得把 prompt 最后 token 再喂一次；
- 后续 token 使用绝对位置 `pos=len(prefix)`；
- 使用 `s34_gdn_step_fix.patch_gdn_step()`，并做 patch 开/关 A/B；
- E/D 各一份 cache；M 每个逻辑深度独立 cache；
- GDN conv/recurrent state 按物理层和逻辑深度隔离；
- 请求开始时初始化、结束时释放，不能跨题污染；
- cache seed 必须在对应 prefill 完成后按层检查并记录，不得对空 cache seed 后宣称历史状态已恢复。
- 物理层到 M 相对索引使用 `physical_layer - 4`，必须断言层身份。

### 4.3 测试矩阵

先用 3 题、短输出上限 16，再扩大到：

| 维度 | 值 |
|---|---|
| prompt 长度 | 128、512 |
| 生成上限 | 16、64 |
| 深度 | d=2、d=8 |
| 对照 | 全前缀重算、exact 增量 |

### 4.4 必测字段

```text
max_logits_diff
mean_logits_diff
CE_diff
argmax_flip_count
Top-5 overlap
GDN conv state diff
GDN recurrent state diff
full-attention KV length
```

### 4.5 产物和门禁

- `results/e1_cache_correctness.json`
- `runs/e1_cache_correctness.log`

通过条件：教师强制下无 argmax 翻转；CE/logits 差异处于已记录 dtype/backend 容差；cache 生命周期无污染。失败则停止，不进入 E2。

---

## 5. 阶段 E2：序列级自适应真实算力

> 这是当前最稳的真实兑现路径。不要把失败的 token 级 skip 实现直接复用。

### 5.1 模式定义

#### Mode A：fixed-d2 exact incremental

- prompt prefill；
- 生成每个 token 执行 E + M×2 + D；
- 使用 GDN 单步补丁；
- 质量基线。

#### Mode B：fixed-d8 exact incremental

- prompt prefill；
- 生成每个 token 执行 E + M×8 + D；
- 作为高计算基线。

#### Mode C：sequence-adaptive exact incremental

- prompt prefill 后，用 Q1 router 聚合 prompt 路由信号决定 d*；
- d* 只使用已校准集合 `{2, 3}`；
- 整个生成请求固定使用 d*；
- 不在生成中途让单个 token 缺席 GDN 深度；
- 因此 GDN 增量状态语义与 fixed-d* 一致。

### 5.2 质量矩阵

使用固定 40 题 validation 子集：

- arm A 单遍；
- fixed-d2；
- fixed-d8；
- sequence-adaptive。

每题记录：

```text
question_id
gold
pred
strict_correct
finished
generated_tokens
d_star
```

### 5.3 算力与时间矩阵

每个模式报告：

```text
prefill layer calls
steady-state decode layer calls
total M layer calls
total layer-equivalent calls
TTFT
mean decode ms/token
P50/P95 decode ms/token
whole-request ms
peak VRAM
tokens/s
```

正式墙钟规则：

- 关闭游戏和其他独显负载；
- warm-up 至少 3 题；
- 正式题目至少 20 题；
- 同一机器、同一 dtype、同一 batch、同一 prompt/输出配置；
- CUDA event 或设备同步；
- 计时不包括模型加载；
- 日志写盘不进入核心计时。

### 5.4 产物

- `results/e2_sequence_adaptive_quality.json`
- `results/e2_sequence_adaptive_compute.json`
- `runs/e2_sequence_adaptive.log`

### 5.5 通过条件

- sequence-adaptive strict 不低于 fixed-d2 的预设容差；
- 完成率、重复率不恶化；
- decode 实际 M layer calls 和 steady-state 时间低于 fixed-d8；
- 结果必须同时报告“相对 fixed-d8 节省”和“相对 arm A 的额外成本”。

---

## 6. 阶段 E3：缓存工程优化

只有 E1、E2 通过后执行。每个优化必须单独实验，不得一次叠加：

### E3.1 预分配 cache

- 用固定容量/分块容量替代每 token `torch.cat`；
- 验证 cache 内容与 E1 完全一致；
- 测 TTFT、decode ms/token、峰值显存。

### E3.2 减少 Python 调度

- 将静态层列表、位置对象和 cache index 预计算；
- 不改变模型数学语义；
- profiler 记录 CPU 调度时间。

### E3.3 融合/图执行候选

- 只有目标硬件支持且数值门通过才启用；
- 记录 kernel、fallback、dtype 和 backend。

产物：

- `results/e3_cache_engine_<variant>.json`
- `runs/e3_cache_engine_<variant>.log`

---

## 7. 阶段 E4：token 级 skip 条件分支

当前 token 级 skip 已发现 GDN 单步/分块不等价。只有满足以下任一条件才可重新启动：

### E4-A：部署语义一致性训练

训练期直接使用部署单步 GDN 语义，新增 checkpoint、token budget、质量门和看板。此分支必须另行授权训练。

### E4-B：严格一致的自定义 GDN 单步实现

完成数学级/逐层级证明和 E1 复验后，才允许 token 级 skip。

### E4-C：收官不做 token 级 skip

保留序列级自适应作为正式结果，并把 token 级 skip 失败写入论文限制。

执行 AI 不得自行从 E4-C 切回 E4-A 或 E4-B。

---

## 8. 停止、恢复和看板

### 8.1 立即停止条件

- cache 正确性失败；
- 输出出现重复、乱码、固定 token；
- GPU 被游戏或外部程序抢占；
- OOM、进程退出、checkpoint 损坏；
- strict/完成率超过预设质量损失；
- 计数器与 profiler 不一致。

### 8.2 恢复规则

- 代码测试可从最近 JSON/逐题结果继续；
- 计时测试若有外部负载或配置变化，整格重测，不拼接无效结果；
- 训练类工作只从完整 checkpoint 恢复；
- 每 25 步保存 latest/prev，转折点永久保存；
- 恢复时记录 `resume_from`、旧进程、旧配置和重做区间。

### 8.3 dashboard

看板必须显示：

```text
mode
phase
completed_items
strict/finished
mean d*
M layer calls
layer-equivalent calls
TTFT
decode ms/token
P50/P95
VRAM
GPU utilization
correctness gate
```

看板是监控，不是结果档；原始 JSON 和日志才是证据。

---

## 9. E5 报告与论文归档

### 9.1 结果标签

独立标注，不得合并：

- `quality_parity_passed`
- `exact_cache_correctness_passed`
- `sequence_adaptive_layer_saving_passed`
- `wall_clock_speedup_passed`
- `token_level_skip_passed`
- `npu_port_passed`

### 9.2 主表

| 模式 | strict | 完成率 | 平均 d* | M calls/token | TTFT | decode ms/token | whole request | VRAM | 参照系 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| arm A | | | | | | | | | |
| fixed d2 | | | | | | | | | |
| fixed d8 | | | | | | | | | |
| sequence adaptive | | | | | | | | | |

### 9.3 归档

追加：

- `docs/plans/S08_CAMPAIGN_LOG.md` 新章节；
- `results/` 全部 JSON；
- 环境和资产 manifest；
- 代码 hash；
- profiler 摘要；
- 失败实验和停止原因；
- checkpoint 和转折点清单。

论文表述必须明确：

- “层前向节省”不等于“墙钟加速”；
- “相对 fixed d8 节省”不等于“相对非循环模型省”；
- token 级 skip 未通过时，不能把 sequence-adaptive 的结果写成 token-level sparse execution；
- 旧 `results/s33_s08_seq_adaptive_gen.json` 的 61.7% 记为 `legacy_pending_revalidation`，新结果覆盖后方可引用新比例；
- 层数的正确基准是 E=4、M=12、D=8；禁止再用 `20+12d`、32 或 116 等旧口径。

---

## 10. 任务完成条件

算力兑现线只有在以下信息齐全后才算完成：

1. E0 环境和资产 manifest 完成，且阻断项全部清除；
2. E1 exact cache 正确性通过（含 patch 开/关、完整 E/M/D prefill、多步对照）；
3. E2 sequence-adaptive 的质量和真实 hook 计数完成；
4. 至少一个固定模式和一个自适应模式完成同步墙钟测量；
5. 质量、层调用、FLOPs 代理、墙钟、显存、吞吐均带参照系；
6. 失败的 token-level skip 与 GDN 根因保留；
7. 结果 JSON、日志、环境、代码 hash 和台账全部归档；
8. 未通过的标签明确写为 `false`/`not tested`，不能留空或推断为通过。

## 11. 执行前必须先修正的已知代码风险

执行 AI 在 E1 前必须逐条确认，否则不得开始计时：

| 风险 | 位置 | 必须动作 |
|---|---|---|
| 层数/公式错误 | `s33_patched.py` 等旧脚本 | 改用 E=4/M=12/D=8 与 `12+12d`；重算 |
| 计数非 hook | 旧 s33 为公式累加 | 用 forward hook 计真实 layer 调用，分开 prefill/decode |
| prefill/decode 不同深度 | 旧 s33 soft 八圈 prefill + 固定 d* decode | 用同 d* 重新 prefill；探测 pass 单列成本 |
| cache seed 对空 cache | `seed22` 在 prefill 前调用 | prefill 后按层检查再 seed，记录每层状态 |
| 物理层映射 | `s32_s08_exit_skip_gen.py` 的 `li-8` | 改为 `li-4` 并断言层身份；token skip 本身暂停 |
| soft depth 轴 | `s30_s08_adaptive_gen.py` 的 `ps[:,-1]` | 改为 `ps[:, :, -1].sum(0).mean()` |
| 旧 s31 不真实跳过 | `s31_s08_hard_exit_gate.py` | 只作前置参考；不得引用 83.3% 为实测节省 |
| d* 舍入/集合 | 旧 s33 为 `round`、范围 2–8 | 先锁定规则与 `{2,3}` 候选，再校准 |

## 12. 执行 AI 必须新建的脚本与接口契约

不要直接修改或复用旧 `s33_patched.py` / `s32_s08_exit_skip_gen.py` 作为正式结果来源；它们保留为历史。新建以下脚本，文件名固定：

### 12.1 `s40_e0_manifest.py`

职责：采集环境、hash、资产可读性。

必须输出 `results/e0_compute_env.json` 与 `results/e0_asset_manifest.json`，字段至少包括：

```text
python, torch, transformers, cuda, gpu_name, gpu_total_mib
model_snapshot_path
checkpoint_sha256: {armA, armB, qat}
script_sha256: {s08_arch, s34_gdn_step_fix, scorer_strict, new scripts}
transformers_cache_class
external_gpu_processes
```

### 12.2 `s41_e1_cache_gate.py`

职责：exact 增量正确性门。

接口要求：

```text
--mode {full_recompute, incremental}
--depth {2, 8}
--patch {on, off}
--n 3
--max-new 16
```

内部必须：

1. 完整 prefill（与 decode 相同 depth 语义）；
2. 首 token 来自 prefill 末位 logits；
3. 后续 token 用绝对位置增量；
4. 用 forward hook 统计每层调用次数；
5. 对比全前缀重算的 teacher-forced logits；
6. 记录 conv/recurrent/KV 状态。

输出 `results/e1_cache_correctness.json`，包含每个 combo 的通过/失败与全部最大差。

### 12.3 `s42_e2_benchmark.py`

职责：质量 + 真实计数 + 墙钟。

接口：

```text
--modes {armA, fixed_d2, fixed_d8, seq_adaptive}
--n-gen 40
--max-new 200
--repeat 3        # 计时重复
--timing {on, off}
```

强制：

- 生成与计时分离，先生成后计时；
- 计时前 warm-up 3 题；
- `--timing on` 时若检测到外部 GPU 负载（nvidia-smi 计算进程数 > 1 或 util 异常），写 `invalid` 并停止；
- 每题逐题落盘；
- 记录 hook 计数与 CUDA event 时间。

输出：

```text
results/e2_quality_<mode>.json
results/e2_compute_<mode>.json
results/e2_timing_<mode>.json
```

### 12.4 `s43_report.py`

职责：合并 E0–E3 结果，生成 E5 主表和契约标签。

输出：

```text
results/e5_compute_report.json
docs/plans/S08_CAMPAIGN_LOG.md 追加章节（由执行 AI 手写或脚本生成）
```

## 13. 决策树

```text
E0 通过？
 否 -> 停止，问题报告，结束
 是 -> E1

E1 exact d2 通过？
 否 -> 先分 patch=off 和 patch=on：
        off 也失败 -> 缓存管线实现错误，修
        off 通过、on 失败 -> GDN 补丁引入差异，禁止进入 E2
        on 无效（与 off 相同）-> seed 时机或补丁未生效，修
 是 -> E1 d8 -> 通过则 E2

E2 fixed_d2 质量通过？
 否 -> 记录，停止（不进入自适应对比）
 是 -> E2 seq_adaptive

seq_adaptive 质量在容差内且 decode 时间 < fixed_d8？
 是 -> 标记 layer_saving / wall_clock 通过 -> E3
 否 -> 标记未通过，报告实际差距 -> E3 或直接 E5（按预算）

E4 token skip：默认不做。只有 E4-A/B 另行授权才进入。
```

## 14. 执行记录格式

每个阶段结束时必须写一条 JSON：

```json
{
  "stage": "E1",
  "timestamp": "...",
  "status": "passed|failed|invalid",
  "contracts": {"C-03": true, "C-04": true, "C-11": true},
  "inputs": {"checkpoint": "...", "script_hash": "..."},
  "results_file": "results/e1_cache_correctness.json",
  "notes": "..."
}
```

追加到 `results/e_series_stage_log.jsonl`，append-only。

## 契约索引速查

- C-01：模型/权重
- C-02：数据隔离
- C-03：正确性
- C-04：GDN 单步
- C-05：算力计数
- C-06：墙钟
- C-07：质量
- C-08：阶段边界
- C-09：存档
- C-10：停止
- C-11：旧结果与语义阻断
- E0：环境/资产
- E1：exact cache
- E2：序列级自适应
- E3：缓存工程
- E4：token skip 条件分支
- E5：报告归档

## 15. 看板与遥测契约（s44）

算力线看板为 `s44_dashboard_compute.py`（只读，端口 8765，替代 4B 训练监控页）：

```bash
loopus_env/Scripts/python.exe s44_dashboard_compute.py --port 8765
```

看板数据源：`results/e_series_stage_log.jsonl`、`results/e0_*.json`、
`results/e1_*.json`、`results/e2_*.json`、`runs/<run>/history.jsonl`、实时 nvidia-smi。

**s41/s42 必须按以下遥测契约写 `runs/<run>/history.jsonl`**，否则看板无法显示：

```text
{"event":"start","args":{"phase_label":..., "mode":..., "depth":..., "patch":...}}
{"event":"item","mode":...,"depth":...,"patch":...,"ok":...,"finished":...,
 "gen_tokens":...,"d_star":...,"layers":...,"ms_per_token":...,"ts":...}
{"event":"eval","mode":...,"n":...,"strict":...,"finished":...,
 "mean_depth":...,"m_layer_calls":...,"layer_equivalent":...,
 "ttft_ms":...,"decode_ms_mean":...,"decode_ms_p50":...,"decode_ms_p95":...,
 "peak_vram_mib":...,"correctness_gate":"pass|fail|na"}
{"event":"turn","name":..., ...}      # 转折点
{"event":"final", ...}                # 收尾
```

E1 的每个 combo 结束后写一条 `eval` 记录，`correctness_gate` 必须显式给出；
E2 每个模式结束后写一条 `eval` 记录（含 §5.3 全部计时字段）。
逐题 `item` 记录必须实时 flush，不能等整轮结束。

看板是监控，不是结果档；正式结果仍以 `results/*.json` 与阶段日志为准。
