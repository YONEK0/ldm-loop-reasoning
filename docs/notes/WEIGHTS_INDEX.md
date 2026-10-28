# 权重归档索引（19 个关键 checkpoint，108 MB）

> 2026-10-28。为"每个结论都能回溯到具体权重"而建。
> 全部为 **EMA 权重**（与 JSON 里 `evals` 的评测口径一致）。
> 存放：魔搭 `Yoneko/ldm-feasibility-testkit` 的 `transfer/ckpt/` 下；
> GitHub 仓库**不含** `.pt`（`.gitignore` 排除，避免仓库膨胀）。

## 为什么只留这 19 个

本地共 356 个 `.pt`（6.6 GB），绝大部分是中间 MILESTONE 与 `state.pt`
（含优化器状态，仅续跑用）。结论回溯只需要**每个关键条件的最终权重**：

| 组 | 数量 | 用途 |
|---|---|---|
| 13 seed 主表 | 13 | P0-1 通过率 8/13 的原始权重；也是 H-029（task loss ↔ hop7）的样本 |
| H1 决策模块四臂 | 4 | H-028：mlp 证实的原始权重 |
| H2 warmup 双值 | 2 | H2 判读：同 seed 只改 warmup 的对照权重 |

## 清单

### 13 seed 主表（48k，Ouro + step-sup，`--curriculum-step 8000`，warmup 960）

命名规则 `MAIN13_{PASS|fail}_s{seed}.pt`，PASS 判据 `h7 ≥ 0.80`。

| 文件 | seed | hop6 | hop7 | task loss | 判定 |
|---|---|---|---|---|---|
| MAIN13_PASS_s71.pt | 71 | 0.900 | 0.915 | 0.1348 | PASS |
| MAIN13_PASS_s83.pt | 83 | 0.895 | 0.905 | 0.1733 | PASS |
| MAIN13_PASS_s89.pt | 89 | 0.895 | 0.900 | 0.0707 | PASS |
| MAIN13_PASS_s61.pt | 61 | 0.900 | 0.895 | 0.2392 | PASS |
| MAIN13_PASS_s11.pt | 11 | 0.860 | 0.840 | 0.0528 | PASS |
| MAIN13_PASS_s67.pt | 67 | 1.000 | 0.835 | 0.0006 | PASS |
| MAIN13_PASS_s79.pt | 79 | 0.995 | 0.825 | 0.0027 | PASS |
| MAIN13_PASS_s29.pt | 29 | 1.000 | 0.805 | 0.0019 | PASS |
| MAIN13_fail_s73.pt | 73 | 0.745 | 0.770 | 0.2507 | fail |
| MAIN13_fail_s13.pt | 13 | 0.780 | 0.765 | 0.2267 | fail |
| MAIN13_fail_s19.pt | 19 | 0.780 | 0.745 | 0.3970 | fail |
| MAIN13_fail_s17.pt | 17 | 0.725 | 0.740 | 0.4134 | fail |
| MAIN13_fail_s53.pt | 53 | 0.720 | 0.695 | 0.2102 | fail |

对应结果：`results/remote_P04_cs48k_s{71,17,53}.json`、
`results/remote_P05_48k_s{11,13,19,29}.json`、
`results/remote_P18_seed_s{61,67,73,79,83,89}.json`。

### H1 决策模块四臂（seed 17，48k，warmup 960）

| 文件 | 读出形态 | 新增参数 | hop6 | hop7 | 用途 |
|---|---|---|---|---|---|
| H1_head_linear_s17.pt | `Linear(192→1)`，193 参 | 0 | 0.80 | 0.78 | 对照 |
| H1_head_mlp_s17.pt | `192→192→GELU→1` | +37,249 | **0.97** | **0.875** | H-028 证实档 |
| H1_head_bilinear_s17.pt | `(W·h)⊙e` | +36,864 | 0.735 | 0.51 | "拟合了但不外推" |
| H1_head_xattn_s17.pt | 候选作 query 全序列注意力 | +111,361 | 0.505 | 0.545 | "停在 ln2，学不动" |

对应结果：`results/head_ablation/H1_*_s17.json`。

### H2 warmup 双值（seed 53，48k，唯一变量 warmup）

| 文件 | warmup | hop6 | hop7 | 判定 |
|---|---|---|---|---|
| H2_w960_s53.pt | 960 | 0.750 | 0.795 | fail |
| H2_w1280_s53.pt | 1280 | 0.915 | 0.890 | PASS |

对应结果：`results/warmup/H2_w{960,1280}_s53.json`。

## 已知缺口

**P19_long64k 五个 run（64k 步）的权重没有归档**：服务器已关机，本地从未取回。
`state.pt`/`ckpt.pt`/`MILESTONE` 全部只在服务器磁盘上（共约 1.9 GB，含其它 run）。
若平台关机后仍保留数据盘，下次开机可补齐；否则这部分只能用 JSON 指标回溯，
无法做"按 hop 的 exit 分布 / 注意力模式"这类需要权重的分析。

**这五个 run 的 JSON 已完整归档**，指标层面不受影响。

## 用法

```python
import torch
from s63_ldm_model import build_model
m = build_model('loop', n_rounds=8, n_core=1, round_in_core=True,
                pos='none', step_sup=True, head_type='linear')  # H1 臂见文件名
sd = torch.load('MAIN13_PASS_s71.pt', map_location='cpu', weights_only=False)
if isinstance(sd, dict) and 'model' in sd:
    sd = sd['model']
m.load_state_dict(sd, strict=False)
m.eval()
```

注意：H2 两个权重是在 `--warmup` 不同的两个 run 里训出来的，但模型结构完全相同，
加载方式一致。H1 的 bilinear/mlp/xattn 三个权重需要对应 `head_type` 才能加载。
