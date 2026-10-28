# err-task-shortcut.md — 任务存在捷径，模型的高分不代表学会目标能力

## 症状（你会看到什么）

- 某配置在评测集上达到 0.95–1.00，看起来"完全学会"；
- 但一旦换一个语义等价、去捷径的对照集，准确率塌到 **0.5 左右**；
- 异常信号：**准确率违反了任务的物理约束**。
  例：hop7 需要 7 跳，却在 r=6（每轮最多推进一跳）时拿到 0.787。

## 复现方式

本项目实例（2026-10-27）：

```bash
# 1. 检查任务是否有简单规则可满分
python -c "
import json
from pathlib import Path
rows=[json.loads(l) for l in Path('data/ldm_mvt5/train.jsonl')
      .read_text(encoding='utf-8').splitlines() if l.strip()]
# MVT-5: '选下游更长的候选'
...  # 结果 12000/12000 = 1.000
"

# 2. 用去捷径对照集复测现有 ckpt
python s84_dual_task.py eval --ckpt results/X_ckpt.pt --file dev_random.jsonl --rounds 8

# 3. 用 s83 构造 orig/tied 变体对比
python s83_shortcut_probe.py --ckpt results/X_ckpt.pt --rounds 6
```

## 根因（已证实）

`s66_ldm_mvt_gen.py:72`：
```python
deep = k - 2 if comp_mode == 'deep' else rng.randint(0, k - 2)
```
正确候选链长 = hop−1，干扰候选链长 = hop−1（恒差 1）。
所以"从两候选各走到尽头、选更长的"在全部数据上 **100% 正确**。

**这不是设计疏忽，而是纯链任务保证唯一解的必然要求**：
若干扰链 ≥ hop，从 e0 走 hop 步会到达两个终点，唯一答案不成立。

## 为什么难以发现

- 训练/验证准确率都正常上升，没有任何异常；
- 损失曲线健康，by_hop 指标漂亮，阶梯形"看起来"符合预期；
- 只有做**物理约束检查**（r=6 解 hop7）才会暴露。

## 修法（已验证 / 候选）

### 已验证
1. **构造去捷径对照集**（`s84_dual_task.py` MVT-D v2）：
   - 两条链从 e0 引出，只有一条在第 hop 步可达；
   - **两候选结构完全对称**：都有入边、都有出边、下游长度随机；
   - 验证：pick_longer = 0.336、pick_shorter = 0.517、标签均衡 0.49。
2. **旧模型在对照集上复测**：预期跌到随机，作为捷径主导的证据。

### 候选（未验证）
3. 加**反向边 / 环**，使"长度"与"可达性"彻底解耦；
4. 引入**多候选**（>2），让位置/长度信号更弱；
5. 训练时**混合两种方向**的样本（干扰更长 / 更短各半），
   使长度信号与标签去相关。

## 预防措施（写进流程）

**任何新任务上线前，必须跑"最简单启发式规则"测试：**

```python
# 伪代码
for rule in [pick_first, pick_longer, pick_shorter, pick_has_in_edge, ...]:
    acc = evaluate_rule(rule, train_set)
    assert acc < 0.7, f'任务有捷径：{rule} 得 {acc}'
```

若某规则接近 1.0，任务没有区分力，**不要开始训练**。

## 相关假说

- `../hypotheses/H-023-CONFIRMED-downstream-length-shortcut.md`（本错误直接产物）
- `../hypotheses/H-001-REVOKED-loop-carries-serial-depth.md`（受污染结论）
- 调查报告：`../notes/SHORTCUT_INVESTIGATION_20261027.md`

## 触发文件

- `s66_ldm_mvt_gen.py`（任务生成器，`comp_mode='deep'`）
- `s83_shortcut_probe.py`（捷径探针）
- `s84_dual_task.py`（无捷径对偶任务 MVT-D）

## 影响范围（重要）

MVT-5 上的**所有**涉及任务语义的结论都需重验：
H-001、H-003、H-011、H-013、H-014、H-022。
仅训练动力学/架构消融（H-004/005/006/008/010/012/021）的相对比较仍有效。
