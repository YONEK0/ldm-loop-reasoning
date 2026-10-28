# 上下文快照（压缩恢复用）· 2026-10-28 13:45

> 工作目录 `E:\develop\project_files\llm-test`。读完本文件 + 下列三份即恢复全部上下文：
> 1. `docs/notes/P0_FINAL_VERDICT_20261028.md`（P0 判读，配方可靠性）
> 2. `docs/notes/EVIDENCE_FOR_CONCLUSIONS_20261028.md`（各结论的文献依据）
> 3. `docs/hypotheses/INDEX.md`（假说表，三态）
> 4. `docs/plans/PAPER_SKELETON_20261027.md`（论文骨架 v3）

---

## 0. 现在正在做的事（最重要）

**P19 长训实验**：5 个"失败 seed"用 **64k 步**（原 48k）重跑，验证是否能把
hop6 推到 0.85 以上、从而把通过率从 61.5% 提上去。

```bash
# 服务器上已在跑（13:30 启动，5 进程，GPU 98%）
P19_long64k_s{17,53,13,19,73}    # 24k→64k，课程 --curriculum-step 8000
# 首个 eval 在 8000 步（约 13:45），全跑完预计 15:30–16:00
```

**判读方式**：
```bash
python s76_remote.py shell "cd /root/ldm && /root/miniconda3/bin/python s86_progress.py --pattern 'P19_long64k*'"
# 关键列 h6：若从 0.72–0.78 升到 ≥0.85，说明失败是训练不足、可修
# 若仍停在 0.72–0.78，说明是 seed 级不可修复，需改初始化/课程
```

**这一步的意义**：13 个 seed 的通过率是 61.5%（8/13），已达"可启动真实训练"的门槛；
若能修掉那 5 个失败 seed，就能说明"配方可靠"而非"碰运气"。

---

## 1. 项目目标与当前阶段

**目标**：验证"循环（looped/recurrent depth）是否承载串行推理"，产出可发表的
机制结论 + 可用训练配方。

**已完成**：P0 三项预注册判据全部通过（详见 §3）。**下一步**是迁移到
昇腾 910B 做基线复现，然后启动真实训练。

---

## 2. 任务与模型（当前配置）

**任务 MVT-D v2**（`s84_dual_task.py`）：两条等长链从 e0 引出，只有一条在第 hop
步可达；候选结构完全对称（都有入边/出边、下游长度随机），**已消除三类局部捷径**
（下游链长、入边存在性、出边存在性）。审计：pick_longer=0.336、标签均衡 0.49。

**模型**：Prelude(1) + Core(1 层共享) × R + Coda(1)，d=192 heads=4 ffn=768，
**NoPE**，融合编码（每事实一位），词表 320，**1,474,307 参数**（含 step_head 61,760）。

**标准配方**（48k）：
```bash
python s71_ldm_iter4.py --task mvtd --data-dir data/ldm_mvtd2 --pos none \
  --ouro --step-sup --curriculum 2,3,4,5,6 --curriculum-step 8000 \
  --steps 48000 --bs 64 --lr 1e-3 --r-max 8 --eval-every 4000 \
  --seed <S> --out results/<TAG>_s<S>.json
```

---

## 3. 已确立结论（有证据，不要重复验证）

### 3.1 P0 判据全部通过

| 判据 | 结果 |
|---|---|
| P0-1 配方可靠（≥50% seed 达 hop7≥0.80） | **8/13 = 61.5%** ✓ |
| P0-2 loop vs dense | dense 1.40/2.29/3.18M 的 loss **精确停在 ln2**，参数 ×2.3 无效 ✓ |
| P0-3 训练长度 | 课程解耦后同 step 无差异；48k 收益来自覆盖 hop6 ✓ |

**13 seed 明细（48k）**：
PASS：s71=.915、s89=.900、s83=.905、s61=.895、s11=.840、s67=.835、s79=.825、s29=.805
fail：s73=.770、s13=.765、s19=.745、s17=.740、s53=.695
（h7 均值 .818、中位 .825；h8 均值 .779）

### 3.2 瓶颈定位（关键发现）

**完美分离**：hop6 ≥ 0.85 的 8 个 seed → hop7 全部 ≥0.80；
hop6 < 0.85 的 5 个 seed → hop7 全部 <0.80。h7−h6 差值多在 ±0.05 内。

→ **外推从来不是瓶颈，hop6 收敛才是。** 提高通过率应改训练（步数/课程/lr），
不要再去改外推手段。

### 3.3 其他已确立

- **循环被真实使用**：三个 seed 的 r=1 全部随机（0.494–0.508）
- **r≈2 饱和**：r=2 达 0.87，r=3–8 仅 0.88–0.90（增益 <0.03）
- **Ouro 损失优于 persist**：6 seed 全脱离随机；熵实测 2.079=ln8、e_halt≈4.5
- **MVT-5 作废**：捷径在全部 19750 样本上准确率 1.0000

---

## 4. 已证伪 / 已结案（不要再试）

| 编号/方向 | 结论 |
|---|---|
| **H-025 邻接掩码** | **检验手段不成立**（三变体：t跳/轮未约束、1跳/轮阻断学习、BFS分层 hop2 也 0.50）。根因：节点=事实、候选=实体，不同图 |
| H-023 MVT-5 捷径 | 已证实，MVT-5 所有能力结论作废 |
| H-014 外推 | 四种干预（dyn-r/warm-noise/sup-cap/r→16）全败，只靠训练覆盖 |
| H-022 深度插值 | 证伪（训练 {2,4,6,8} 解不了 {3,5,7}） |
| H-001 阶梯形 | 撤销（原证据受 MVT-5 捷径污染） |
| H-021 每轮层数 L | L=1/2/4 等价，架构定论 L=1 |
| 其他 | 残差爆炸、lr 扫描、EMA、加宽 d、课程硬切换、课程回采、per-hop 读头、FiLM |

**已下定论的文献依据**（详见 `EVIDENCE_FOR_CONCLUSIONS_20261028.md`）：
- r≈2 饱和 ← arXiv:2610.26030（"two fully trained loops optimal"）、2604.21106（φ=0.46）
- 稠密学不了多跳 ← arXiv:2402.12875（TC⁰ / CoT 串行表达能力）
- 掩码阻断学习 ← arXiv:2510.18825（感受野与可分性正相关）、2502.12352
- 外推只靠覆盖 ← arXiv:2604.07822 / 2605.30757 / 2512.02677 / 2609.33144

---

## 5. 基础设施（关键，容易踩坑）

### 服务器
- `ssh -p 53402 root@region-41.seetacloud.com`（密码在 `s76_remote.py` 内）
- **RTX 3080，10240 MiB**（不是 2080Ti 11G）
- 工作目录 `/root/ldm`；`/root/miniconda3/bin/python`（torch 2.5.1+cu124）

### 智能体（dsh）——已打通，但有两个坑
- Node v22.20.0 → `/usr/local/node22`（软链到 `/usr/local/bin`）
- **`@deepseek-ai/dsh` 0.2.1-rc.2**（`npm i -g --registry=https://registry.npmmirror.com`，617 包 24 秒）
- 用法：`dsh headless "<任务>"`（一次调用，答完退出）
- **坑 1：沙箱**。默认 `workspace-write` 但服务器无沙箱后端 → 拒绝执行 bash。
  必须 `export DSH_PERMISSION_MODE=danger-full-access`，否则只能写文件不能跑命令
  （10-27 夜因此空转整夜）
- **坑 2：密钥**在 `/root/ldm/.dsh_env`（600），由 `_run_loop.sh` source
- npm 上的 `dsh` / `pi` / `deepseek-harness` 都是无关包或占位包，别装

### 迭代链（闭环工作流）
- 编排器 `s89_auto_loop.py`：`spec.json → scan → evaluate → decide → act → record`
- 决策后端：`--agent rules`（规则）/ `--agent dsh`（智能体，失败回退规则）
- 契约写在 `docs/plans/AGENT_TASK_BRIEF.md` 第八节：先判读→行不通查文献→
  提假设带预注册判据→再启动实验→单变量对照→归档
- 已修：spec/state 每轮重读、PID 文件（`results/auto_loop.pid`）
- **注意**：`pkill -f` 对 setsid 脱离的进程不可靠，**必须用数字 PID 精确清理**

### 文件传输
**一律走魔搭**（`s85_ms_transfer.py`），不要 SFTP 直传数据/代码。
```bash
export MS_TOKEN='<本地有>'          # 或从服务器 /root/.ms_token 取
python s85_ms_transfer.py up --files <文件>          # 本地 → 魔搭
python s85_ms_transfer.py pull --files <文件>        # 魔搭 → 服务器
```
上传后服务器侧需先 `snapshot_download` 再 `cp`（见历次命令）。

---

## 6. 关键文件

| 文件 | 作用 |
|---|---|
| `s63_ldm_model.py` | 模型/编码（含 `adj_mask`/`adj_bias_list` 接口） |
| `s71_ldm_iter4.py` | **训练主脚本**（`--ouro` / `--adj-mask` / `--curriculum-step` / `--task mvtd`） |
| `s84_dual_task.py` | MVT-D v2 生成器（无捷径任务） |
| `s88_adj_task.py` | 邻接掩码构造（已结案，保留备查） |
| `s86_progress.py` | 进度一览（在服务器上跑） |
| `s89_auto_loop.py` | 迭代链编排器 |
| `_run_loop.sh` | supervisor（含 DSH_PERMISSION_MODE） |
| `docs/plans/auto_loop_spec.json` | 阶段定义与判据 |
| `docs/plans/AGENT_TASK_BRIEF.md` | 给智能体的简报 + 闭环契约 |
| `docs/plans/ASCEND_910B_MIGRATION_CHECKLIST.md` | 昇腾迁移清单（含验证容差 ±0.02） |

---

## 7. 下一步（按优先级）

1. **读 P19 64k 结果**（15:30–16:00）：hop6 是否升到 ≥0.85。
   - 是 → 通过率可提升，改标准配方为 64k
   - 否 → seed 级问题，需查初始化/课程
2. **迁移昇腾 910B**：按 `ASCEND_910B_MIGRATION_CHECKLIST.md`，
   先用 `P04_cs48k_s71` 配置跑 4k 步做数值一致性验证（容差 ±0.02）
3. **论文写作**：按 `PAPER_SKELETON_20261027.md` v3 口径（压缩式推理），
   待回填：阶梯矩阵全表、训练曲线图、参数-成功率散点
4. 迭代链可继续用 `--agent dsh` 跑后续实验

---

## 8. 纪律（用户明确要求）

- 预注册判据跑前写定，跑完不改口径；负结果照实记录
- 每个训练进程唯一 `--out`；并发 ≤6
- 假说三态命名，改状态要重命名 + 更新 INDEX
- 出问题先暂停、整理、查论文，不许跳跃式证伪
- 结论要落到 `results/` JSON 与 `docs/` 档案
- **不许用 MVT-5 数据支持任何能力结论**（捷径污染）
- 不许 `sleep` 空等（用户曾批评）；但监督时可用 sleep 轮询（用户后续要求）
