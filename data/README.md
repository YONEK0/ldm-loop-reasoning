# 数据集状态总表

> 全部条目由 `s109_observable_shortcut.py` 实测（规则：只看 `facts`，
> 从每个候选走到尽头，取下游链最长者）。命中率 1.0000 ⇒ 存在 100% 捷径 ⇒ 弃用。

| 数据集 | 状态 | 实测捷径命中率 |
|---|---|---|
| `ldm_edge_action` | **弃用（DEPRECATED）** | 2400/2400 = 1.0000（train） |
| `ldm_edge_action_v3` | **弃用（DEPRECATED）** | 24000/24000 = 1.0000（train） |
| `ldm_edge_action_v4p48` | **弃用（DEPRECATED）** | 24000/24000 = 1.0000（train） |
| `ldm_edge_action_v5` | **弃用（DEPRECATED）** | 192000/192000 = 1.0000（train） |
| `ldm_mvt` | **弃用（DEPRECATED）** | 12000/12000 = 1.0000（train） |
| `ldm_mvt2` | **弃用（DEPRECATED）** | 12000/12000 = 1.0000（train） |
| `ldm_mvt3` | **弃用（DEPRECATED）** | 12000/12000 = 1.0000（train） |
| `ldm_mvt4` | **弃用（DEPRECATED）** | 11900/11900 = 1.0000（train） |
| `ldm_mvt5` | **弃用（DEPRECATED）** | 12000/12000 = 1.0000（train） |
| `ldm_mvtd` | 可用 | max 0.2535 / min 0.4953（train，均未超随机水平） |
| `ldm_mvtd2` | 可用 | max 0.3355 / min 0.3309（train，均未超随机水平） |
| `ldm_mvte` | **弃用（DEPRECATED）** | 200/200 = 1.0000（train） |
| `ldm_parity` | 可用 | 该规则不适用（图中无终止叶，可判定样本 0） |
| `synth_multihop` | 可用 | 无 facts/candidates 结构，规则不适用 |

## 规则说明

- **弃用**的数据集仍保留在仓库中，作为**错误数据 / 负面案例**用于复现与回归测试；
  但**不得**用于任何能力结论、arm 对比或论文主张。
- 每个弃用目录下都有 `DEPRECATED.md` 写明证据与替代方案。
- 新合成任务必须先跑 `s109_observable_shortcut.py`，命中率接近 1.0 直接废弃重做。
