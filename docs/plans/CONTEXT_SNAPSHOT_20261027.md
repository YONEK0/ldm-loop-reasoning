# LDM 线上下文快照（压缩恢复用，2026-10-27 14:30 更新）

> 读完本文 + `docs/plans/SESSION_FINAL_REPORT_20261027.md` + `docs/hypotheses/INDEX.md`
> 应等价于完整记忆。工作目录：`E:\develop\project_files\llm-test`

---

## 0. 一句话现状

1.47M 参数的单层循环模型（LDM）在 MVT-5 合成图任务上：**五级全解成功率 82%**（14/17），
**参数效率碾压稠密**（稠密 10 run 无一 5/5），**深度外推是硬边界**（四种干预全败 +
**深度不可插值**，只有训练覆盖目标深度才有效）。**架构定论 L=1**。
`R_INT_even` 三 run 已完成，H-022 判为**证伪**。

---

## 1. 刚完成：H-022 深度插值（已证伪）

**三个 run**：`R_INT_even_s53` / `s17` / `s71`，2026-10-27 13:51–14:22 全部完成。

- 设计：训练 hop ∈ **{2,4,6,8}**（`--hop-list` 固定采样，跳过奇数），32k 步，
  L=1，persist+step，r_max=8
- 未训练深度：hop3、hop5（dev）、hop7（confirm）

**结果（32k 终评）**：

| seed | hop3 | hop5 | hop7 | 训练 hop6 | 训练 hop8 |
|---|---|---|---|---|---|
| 53 | 0.590 | 0.525 | 0.493 | 0.495 | 0.525 |
| **17** | 0.290 | **0.205** | **0.091** | **0.995** | **0.984** |
| 71 | 0.240 | 0.520 | 0.467 | 0.470 | 0.501 |
| 均值 | 0.373 | 0.417 | 0.350 | — | — |

**判决：不成立**（预注册需 ≥0.9）。未训练深度全部 ≈ 或低于随机。

**s17 是最干净对照**：训练深度 hop6=0.995、hop8=0.984 全学会，
夹中间的 hop5=0.205、hop7=0.091 远低于随机 → 逐点习得，非容量问题。

**机制证据**：
1. 排除位置偏置——三 seed `content_follow=1.000`（`results/bias_INT_*.json`）
2. 失败在第一跳——step_head 命中率：训练 hop2=1.000/hop4=0.99/hop6=0.82–0.85，
   未训练 hop3=0.198–0.667/hop5=0.59–0.61

**附带发现**：稀疏课程拖慢训练深度（hop6 稠密 20–22k 解出，
稀疏 s53/s71 到 32k 未解）。

**产物**：`s82_parse_interp.py`、`results/s82_interp_verdict.json`、
`docs/hypotheses/H-022-REFUTED-depth-interpolation.md`、
`docs/notes/H022_INTERP_OBSERVATIONS_20261027.md`

**服务器**：队列已清空，16:30 自动打包上传魔搭，**17:20 实例自动释放**（无需手动关机）。

---

## 2. 任务与模型

**任务 MVT-5**：符号图路径追踪。给事实三元组 + 起点 + 目标关系序列 + 2 个候选
（1 正确 + 1 深断链干扰，干扰链长 = hop−2），问走 hop 跳后到哪个候选。
词表 320；融合编码（每事实一位，subj+rel+obj 相加）；难度轴 hop=2..6（训练）、7/8（外推）。

**模型**：`Prelude(1层) → [共享 Core(1层)] × R → Coda(1层) → 候选打分`
- d=192, heads=4, ffn=768, round_in_core=True, **pos='none'（NoPE，必要）**
- 参数 1,474,307（含 step_head 61,760）；展开等价 4.59M
- **L（n_core）定论 = 1**（见 §4 H-021）

---

## 3. 已验证的正面结论

| 编号 | 结论 | 关键证据 |
|---|---|---|
| H-001 | 循环轮次承载串行深度（阶梯形） | hop5@r1=0.53、r≥4 达 0.99；hop2–4@r1=1.00（作答压缩） |
| H-002 | NoPE 必要 | NoPE 双 seed 4/5+5/5；正弦双 seed 0/5 |
| H-003 | 参数效率优于稠密 | loop 1.47M：14/17 五级全解；dense 1.4M/2.3M/3.2M 共 10 run，**0 次 5/5**（最好 3/5, peak 0.879） |
| H-011 | persist-sup 解决过冲遗忘 | 逐 hop"归零事件" 5→0；hop6 首次稳定 0.96–1.00 |
| H-013 | step-sup（路径监督）提升成功率 | persist-only 8/14=57% → persist+step 14/17=82%；+4.4% 参数 |

**训练到 hop8 的学习能力（关键）**：L=1、课程到 8、32k →
hop7=**1.000**、hop8=**0.992**；阶梯验证 r=1 时 0.456/0.525、r=3 时 0.984/0.856
→ **真实串行计算，非捷径**。

---

## 4. 已证伪 / 已排除（不要再试）

| 编号 | 方向 | 证据 |
|---|---|---|
| H-004 | 残差爆炸 | 成功/失败残差增长 6.78× vs 6.63× |
| H-005 | 学习率过高 | 1e-3/5e-4/3e-4 全失败，3e-4 最差 0.176 |
| H-006 | EMA | 救 s53 毁 s17，净零 |
| H-008 | 加宽 d（×3.9 参数） | 0.669 vs 基线 0.671 |
| H-010 | 课程硬切换 | mix 0/0.3/0.6 全 0.664–0.671 |
| H-012 | 课程回采 cap-reach | 4 个失败种子无改善 |
| H-015 | confirm 集偏置 | 位置偏好 0.496、内容跟随 1.000（无偏置） |
| H-021 | 增加每轮层数 L | L=1/2/4 在课程到 8 时 hop7 均 0.97–1.00；L=4 参数 ×1.9 无优势 |
| **H-022** | **深度插值（稀疏课程）** | **训练 {2,4,6,8}，未训练的 hop3/5/7 均值 0.35–0.42（≈/低于随机）；s17 训练 hop6/8=0.995/0.984 而 hop5/7=0.205/0.091** |
| — | per-hop 读出头 / FiLM | +0.089 / +0.03，破坏参数效率，组合更差 |
| — | dyn-r（动态轮数，Kohli） | hop7 0.43–0.48 |
| — | warm-noise（状态唤醒，Gu） | hop7 0.42–0.50 |
| — | sup-cap（保持监督截断+16 轮） | hop7 0.32–0.43 |
| — | 推理轮数 6→16 | hop7 0.32→0.40 平坦 |

**外推失败的直接机制（H-022 后修正）**：模型学的是**"离散深度 → 每轮压缩比"的映射表**，
不是连续策略函数。未见深度没有对应表项，输出退回训练分布的混合，
常表现为**系统性选错**（低于随机，s17 hop7=0.091）而非均匀猜测。
**唯一有效手段 = 训练覆盖目标深度，且必须逐级覆盖。**

---

## 5. 关键文件

| 文件 | 作用 |
|---|---|
| `s63_ldm_model.py` | 模型/编码/数据集（含 `step_ent` 路径目标、`warm_noise`、`n_heads_read`、`film_rounds`） |
| `s66_ldm_mvt_gen.py` | MVT-5 生成器 `gen_example(rng, split, hop, comp_mode='deep')` |
| `s71_ldm_iter4.py` | **训练主脚本**（exact-T / ponder / persist-sup / step-sup / dyn-r / sup-cap / hop-list） |
| `s78_agent_loop.py` | 自主编排器（队列+白名单动作+dsh 判读+到期打包上传） |
| `s79_confirm_bias.py` | 评测集偏置诊断（位置/内容跟随） |
| `s81_eval_rounds.py` | 任意推理轮数评测（阶梯验证） |
| `s77_watchdog.py` | 灾变监控守护 |
| `s76_remote.py` | 服务器部署/上传/launch/poll/fetch |
| `docs/plans/SESSION_FINAL_REPORT_20261027.md` | **最终判读报告** |
| `docs/plans/LDM_BACKGROUND_FOR_AGENT.md` | 给 dsh 的完整背景 |
| `docs/plans/AGENT_PROTOCOL_DHES.md` | dsh 执行手册 |
| `docs/hypotheses/INDEX.md` | 假说表（三态命名 PENDING/CONFIRMED/REFUTED） |
| `docs/errors/INDEX.md` | 错误档案（按症状路由） |

**关键命令**：
```bash
# 主配方
python s71_ldm_iter4.py --pos none --persist-sup --step-sup \
  --curriculum 2,3,4,5,6 --steps 24000 --bs 64 --lr 1e-3 --r-max 8 \
  --eval-every 2000 --early-stop --seed S --out results/X.json
# 阶梯验证（任意轮数）
python s81_eval_rounds.py --ckpt results/X_ckpt.pt --rounds 1,2,3,4,6,8 --out Y.json
```

**评测字段**：`dev_r{r}`（固定跑 r 轮精度）、`dev_r{r}_by_hop`（按难度分组，判读主依据）、
`confirm_r6_by_hop`（hop7/8）、`early_stop`（注意：课程 >6 时早停自动禁用，
因为 dev 只覆盖 hop2–6）。

---

## 6. 服务器与基础设施

- **SSH**：历史服务器已回收，连接信息不再记录（凭据改由环境变量注入，
  不入库/不上传）。`/root/ldm` 为工作目录，`/root/miniconda3/bin/python`（torch 2.5.1+cu124）
- **GPU**：2080Ti 11G；实测并发：1 路 51.8ms/step、3 路 52.7ms（1.02×）、
  **6 路 73.1ms（1.41×）但总吞吐最高**（4.26×）→ 并行度设 6
- **到期 2026-10-27 17:20**；编排器 16:30 自动停、打包 `package_done_*.tgz`、上传魔搭
  `Yoneko/ldm-feasibility-testkit/autopilot/`（已验证可上传）
- **本地已有**：`results/final_package_1027.tgz`（46 run 的完整结果包）
- **dsh**：`/root/dsh-test`，headless 模式 `node _dsh.mjs headless "<任务>"`
  （必须用 shim，直接 npm bin 因 `import.meta.main` 静默退出）；key 在
  `/root/dsh-test/.env`（`DEEPSEEK_API_KEY`）；Node 需 ≥22.15（zstd API）
- **编排器**：`launch_agent.sh`（supervisor 循环）+ `s78_agent_loop.py`；
  队列状态 `results/agent_queue_state.json`；报告 `results/agent_report.md`

---

## 7. 已修的关键 bug（重犯会浪费大量时间）

1. **早停误杀深课程**：dev 只到 hop6，课程到 8 时会 20k 步就早停 → 已加
   `curriculum>6 禁用早停` 守卫
2. **tag 前缀**：dsh 加的 tag 自带 `R_`，代码又拼 → 续跑/状态错位 → 统一从 `--out` 取 stem
3. **并发计数翻倍**：进程表每 run 记两个 key → 并行度永远到不了设定值 → 每 run 只记一个
4. **进程消失≠完成**：被杀掉的 run 被标 done → 现在"未达目标步数且无早停"回到 pending
5. **重复 supervisor**：多次重启会叠加，导致多编排器互相覆盖状态 → 只用数字 PID 精确清理
6. **`json.dumps` 放 flush 参数** → TypeError；**CUDA RNG state 需 `.cpu().to(uint8)`**

---

## 8. 方法学纪律（用户明确要求，必须遵守）

- **预注册判据**：跑前写定，不许事后改口径
- **假说三态**：文件名带 `PENDING` / `CONFIRMED` / `REFUTED`，改状态要重命名并更新 INDEX
- **错误档案**：按症状建路由表，新问题先查索引
- **禁止跳跃式证伪**：下结论要有强关联证据；不确定就标"未证实"
- **负面结果照实记录**，不淡化
- **一条新结论要有对照**：同环境、同配方、单变量

---

## 9. 下一步候选（按价值排序）

1. **论文骨架已落盘**：`docs/plans/PAPER_SKELETON_20261027.md`
   （机制：阶梯+NoPE+参数效率 / 方法：persist-sup + step-sup /
   边界：外推失败四干预 + H-022 不可插值 / 限制：1.47M、合成任务）
   —— 待回填：阶梯矩阵全表、训练曲线图、参数-成功率散点
2. **2A 放大重训立项**：若要把配方推到更大规模，需先定：目标规模、
   数据（真实文本 vs 合成）、预算
3. 已完成：深度插值（H-022）→ 判为**证伪**，记录为"必须逐级覆盖"
4. 可选后续实验：把 H-022 反转为正向测试 —— 用"训练覆盖"策略验证
   hop8 可达 0.99（已由 R_H_hop8_s53 证实），作为论文的"正面边界"证据
