# 昇腾 910B 迁移清单（32G × 200h）

> 用户 2026-10-27 确认：训练配方验证通过后可迁移到昇腾 910B（32G 显存，200 小时配额）。
> 本文档列出迁移前必须处理的事项，以及迁移后的验证标准。

---

## 一、已完成的代码改造（2026-10-27）

### 1.1 设备选择统一

新增 `_pick_device()`，优先级 **NPU → CUDA → CPU**：

```python
def _pick_device():
    """设备选择：优先 NPU（昇腾 910B），其次 CUDA，最后 CPU。"""
    import torch
    if hasattr(torch, 'npu') and getattr(torch.npu, 'is_available',
                                          lambda: False)():
        return torch.device('npu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
```

已应用于：`s71_ldm_iter4.py`、`s84_dual_task.py`、`s81_eval_rounds.py`、
`s79_confirm_bias.py`。

### 1.2 RNG state 设备无关

续跑需要保存/恢复加速器 RNG。原代码硬编码 `torch.cuda.*`：

```python
def _accel():
    """返回当前加速器模块（torch.cuda 或 torch.npu）。"""
    if hasattr(torch, 'npu') and getattr(torch.npu, 'is_available',
                                          lambda: False)():
        return torch.npu
    if torch.cuda.is_available():
        return torch.cuda
    return None

def _rng_state():       # 保存
    acc = _accel()
    return acc.get_rng_state() if acc is not None else None

def _set_rng_state(st): # 恢复
    acc = _accel()
    if acc is None or st is None:
        return
    acc.set_rng_state(st.cpu() if torch.is_tensor(st) else st)
```

**注意**：checkpoint 里字段名仍叫 `cuda_rng`（保持向后兼容），
但语义已改为"加速器 RNG"。

### 1.3 验证

本地 CUDA 冒烟通过（200 步，loss/准确率正常，无异常）。
语法检查 4 个文件全部通过。

---

## 二、迁移前必须在服务器上确认的事

### 2.1 环境

```bash
# 昇腾环境检查
npu-smi info                    # 确认 8×910B 或单卡可见
python -c "import torch_npu; print(torch_npu.__version__)"
python -c "import torch; print(torch.__version__, torch.npu.is_available())"
```

**版本匹配**：torch 与 torch_npu 必须版本对应（如 torch 2.1.0 ↔ torch_npu 2.1.0.post*）。
本项目代码用了 `torch.get_rng_state` / `torch.set_rng_state`（CPU 侧，通用）。

### 2.2 关键算子可用性

| 算子 | 用途 | 检查方式 |
|---|---|---|
| `scaled_dot_product_attention` | 注意力 | 昇腾较新版本支持；若不支持需手写 QK^T/softmax/V |
| `index_add_` | 融合编码（facts → 词表） | 检查数据类型和确定性 |
| `masked_fill` + `softmax` | 候选打分 | 通用，风险低 |
| `topk` | 评测 margin | 通用 |
| `torch.randn_like` | warm_noise（已证伪，不用） | 不涉及 |

**自查脚本**（迁移后第一件事）：

```python
import torch, torch_npu
dev = torch.device('npu')
x = torch.randn(64, 192, device=dev)
q = torch.randn(64, 4, 32, device=dev)
# 1) SDPA
try:
    o = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    print('SDPA OK', o.shape)
except Exception as e:
    print('SDPA FAIL -> 需手写注意力:', e)
# 2) index_add
y = torch.zeros(320, 192, device=dev)
idx = torch.randint(0, 320, (64,), device=dev)
y.index_add_(0, idx, x)
print('index_add OK')
```

### 2.3 数据与代码传输

**走魔搭**（`s85_ms_transfer.py`），不 SFTP 直传：

```bash
# 魔搭 → 昇腾机
python s85_ms_transfer.py pull --files data/ldm_mvtd2/train.jsonl ...
```
（若昇腾机不能访问魔搭，才考虑其他方式，但要先问用户。）

---

## 三、迁移验证标准（必须全部满足才能开新实验）

### 3.1 数值一致性

用 `MVTD2_s71` 的配置（persist+step，非 Ouro）在 910B 上跑 4k 步：

| 指标 | CUDA（已有） | 910B（实测） | 容差 |
|---|---|---|---|
| `dev_r1` @4k | 0.503 | ? | ±0.02 |
| `dev_r2` @4k | 0.510 | ? | ±0.02 |
| `dev_r8` @4k | 0.513 | ? | ±0.02 |
| train loss @4k | 1.384 | ? | ±0.05 |

**偏差超容差 → 迁移不成功，需排查**（数值精度、算子实现差异）。

### 3.2 性能基线

| 指标 | CUDA 2080Ti | 910B 目标 |
|---|---|---|
| 单 run 24k 步耗时（1 路） | ~28 分钟 | 记录即可 |
| 6 路并发吞吐 | 4.26× | 记录即可 |

### 3.3 确定性

同 seed 跑两次 4k 步，`dev_r*` 应逐位相同（CUDA 上已验证）。
NPU 上若不完全确定（某些算子非确定性），记录下来并调整复现标准。

---

## 四、910B 32G 带来的新可能

显存从 11G → 32G，可以：

1. **放大 batch size**：bs 64 → 256，训练更快、梯度更稳。
2. **加宽模型**：d 192 → 384（参数 ×3.9），但 H-008 已证伪"加宽有效"，
   在无捷径任务上需重新验证。
3. **并行更多 run**：并发上限从 6 提到 12–16（需实测吞吐拐点）。
4. **更长训练**：如果 P0-3 显示 48k 更好，200h 足够跑很多轮。

**建议**：先用**完全相同配置**复现基线，确认迁移正确后再放大。

---

## 五、迁移前的代码待办

- [x] `_pick_device()` 设备选择（4 个脚本）
- [x] RNG state 设备无关（`_accel` / `_rng_state` / `_set_rng_state`）
- [ ] SDPA 兼容层（若 910B 不支持，fallback 到手写注意力）
- [ ] `torch.cuda.empty_cache()` → 设备无关（检查是否有残留）
- [ ] 混合精度：910B 上 bf16 支持情况（当前代码用 fp32）

---

## 六、风险

| 风险 | 缓解 |
|---|---|
| SDPA 在 NPU 上不可用 | 写 fallback 手写注意力（小模型 d=192，性能可接受） |
| 数值精度差异导致结果偏移 | 用 4k 步基线对比，容差 ±0.02 |
| 算子非确定性 | 记录，调整"逐位相同"的复现要求为"±0.01" |
| 魔搭在昇腾机不可达 | 需先问用户是否有其他传输通道 |
| 200h 配额被无关实验消耗 | 先在 2080Ti 上把配方定死（P0 三项），再上 910B 做规模放大 |
