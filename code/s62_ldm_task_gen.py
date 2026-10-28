"""s62: 循环决策模型（LDM）任务生成——图上"选择正确的下一步边动作"。

任务（LDM 首轮可行性，规格见
docs/plans/LOOPED_DECISION_MODEL_FEASIBILITY_SPEC_DRAFT_V1.md V1.1）：
  - 世界模型与 T-13 同族：匿名实体池 + 三种关系
    P=parent_of（有向）、M=spouse_of（对称）、S=sibling_of（对称）。
  - 每条样例：一组事实（含背景干扰）、当前实体 e0、目标关系路径 goal
    （长度 hop>=2）、以及从 e0 出发与 goal[0] 同关系的全部候选边。
  - 标签 = 唯一能沿 goal 完整走通的候选边（其余候选在剩余路径的不同
    深度断链）。输出契约为候选逐项 logit + 交叉熵监督唯一正确动作。

关键设计（V1.1 冻结）：图必须非函数式——e0 有多条同关系出边，否则
正确动作只由 goal[0] 的关系名决定（函数式图上每实体每关系至多一条
出边），任务退化为无需多跳追踪。动作唯一性与答案唯一性均由确定性
solver 在完整事实集上重推校验。

难度轴：
  - hop = len(goal)：正确动作的判定需要追踪剩余路径能否走通；
    竞争边可在深度 0..hop-2 处断链（断链越深越难）。
  - n_candidates：同关系竞争边数量。
  - n_distractors：背景干扰事实数量（与决策结构实体不相交）。

切分：
  - train / dev_random：hop 2..3 的训练路径。
  - confirm_depth：hop 4..6 的深度外推（训练未见 hop）。
  - confirm_combo：hop 2..3 的留出关系组合。
  - confirm_distractor：训练路径 + 更多候选与背景干扰（广度/长度轴）。
  - pilot：小规模标定集，不进正式对比。

输出：
  data/ldm_edge_action/{split}.jsonl + manifest.json
  results/ldm_task_gen.json
  runs/ldm_task_gen/history.jsonl

纯 CPU；不加载模型、不使用 GPU。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data' / 'ldm_edge_action'
RUN_DIR = ROOT / 'runs' / 'ldm_task_gen'
OUT = ROOT / 'results' / 'ldm_task_gen.json'

SEED = 2033
ENTITY_POOL = [f'E{i:03d}' for i in range(300)]
SYM_RELS = ('M', 'S')            # 对称关系
ATOM_RELS = ('P', 'M', 'S')

# 路径表（goal 即原始关系序列；命名仅便于人读）。
PATHS = {
    'cousin3': ['P', 'P', 'S'],            # 3 训练
    'gu4': ['P', 'P', 'P', 'S'],           # 4 训练
    'uncle3': ['P', 'S', 'P'],             # 3 组合留出
    'inlaw3': ['M', 'P', 'S'],             # 3 组合留出
    'mix4': ['M', 'S', 'P', 'P'],          # 4 组合留出
    'sc5': ['P', 'P', 'P', 'P', 'S'],      # 5 深度外推
    'kin6': ['P', 'P', 'P', 'P', 'P', 'S'],   # 6 深度外推
    'kin7': ['P', 'P', 'P', 'P', 'P', 'P', 'S'],  # 7 深度外推
}
HOP_OF = {k: len(v) for k, v in PATHS.items()}

# hop1/hop2 结构性排除（v2）：hop2 的剩余路径只有 1 步，任何竞争边要么
# 在第 1 步就断（单步即可识破，66.7% 样本退化），要么走通（不唯一）。
# 因此训练最低 hop 必须 >=3，才能构造"需追踪到最后才排除"的竞争边。
# 训练 hop 3..4，深度外推 5..7。
TRAIN_PATHS = ['cousin3', 'gu4']                    # hop 3..4
COMBO_HOLDOUT = ['uncle3', 'inlaw3', 'mix4']        # hop 3..4 留出组合
DEPTH_HOLDOUT = ['sc5', 'kin6', 'kin7']             # hop 5..7
PILOT_PATHS = TRAIN_PATHS + ['uncle3', 'sc5', 'kin7']

BG_ENTS = 8  # 背景干扰事实专用实体数（与决策结构不相交）


@dataclass(frozen=True)
class Fact:
    subj: str
    rel: str
    obj: str


@dataclass(frozen=True)
class Example:
    ex_id: str
    split: str
    goal_name: str
    goal: tuple[str, ...]
    hop: int
    e0: str
    e_answer: str
    facts: tuple[Fact, ...]
    candidates: tuple[tuple[str, str], ...]   # (rel, target) 展示顺序
    answer_index: int
    n_candidates: int
    n_distractors: int
    comp_depth_by_index: tuple[tuple[int, int], ...]  # 非答案候选 -> 断链深度
    proof: tuple[Fact, ...]
    question: str


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'))


def all_reachable(facts: set[Fact], start: str,
                  path: list[str]) -> set[str]:
    """沿关系路径的可达集；M/S 读作无向。"""
    frontier = {start}
    for rel in path:
        nxt = set()
        for f in facts:
            if f.rel != rel:
                continue
            pairs = [(f.subj, f.obj)]
            if rel in SYM_RELS:
                pairs.append((f.obj, f.subj))
            for a, b in pairs:
                if a in frontier:
                    nxt.add(b)
        frontier = nxt
        if not frontier:
            break
    return frontier


def edges_from(facts: Iterable[Fact], x: str, rel: str) -> set[str]:
    """x 经关系 rel 的全部出边目标（M/S 无向）。"""
    out = set()
    for f in facts:
        if f.rel != rel:
            continue
        if f.subj == x:
            out.add(f.obj)
        if rel in SYM_RELS and f.obj == x:
            out.add(f.subj)
    return out


def random_bg_facts(rng: random.Random, ents: list[str], n: int) -> list[Fact]:
    """背景干扰：限定在 bg 实体内的小型亲属图（P 出度<=1，S/M 一对一）。"""
    facts: set[Fact] = set()
    p_out: dict[str, str] = {}
    paired: dict[str, str] = {}
    guard = 0
    while len(facts) < n and guard < 300:
        guard += 1
        rel = rng.choice(('P', 'P', 'S', 'M'))
        a, b = rng.sample(ents, 2)
        if rel == 'P':
            if p_out.get(a) is None and p_out.get(b) != a:
                facts.add(Fact(a, 'P', b))
                p_out[a] = b
        else:
            if a not in paired and b not in paired:
                facts.add(Fact(a, rel, b))
                paired[a] = b
                paired[b] = a
    return sorted(facts, key=lambda f: (f.subj, f.rel, f.obj))


def make_example(rng: random.Random, split: str, goal_name: str,
                 n_cand: int, n_bg: int) -> Example | None:
    goal = PATHS[goal_name]
    k = len(goal)
    if k < 2:
        return None                      # hop1 结构性排除
    n_comp = n_cand - 1
    # 竞争边断链深度（在剩余路径 goal[1:] 上断裂的位置，0-based）：
    #   d=0   -> 该候选第一步就无后继（浅层区分，可被单步规则识破）
    #   d=k-2 -> 该候选走满 k-2 步后才断（必须追踪到最后才能排除）
    # v1 曾均匀采样 0..k-2，实测 68.9% 样本仅用 goal[1] 单步即可区分正确
    # 边，任务退化为"看下一步有没有出边"，无法作为循环深度载体。
    # v2 起：强制至少一条"深断链"竞争边（d>=k-2），使每个样本都要求
    # 接近全深度的追踪；其余竞争边在 [k-2, ...] 内采样。
    deep = k - 2
    depths = [deep]
    if n_comp > 1:
        pool = list(range(max(0, deep - 1), deep + 1))
        depths.extend(rng.choice(pool) for _ in range(n_comp - 1))
    rng.shuffle(depths)

    need_chain = k + 1                   # e0 + k 个链上实体
    need_comp = sum(1 + d for d in depths)
    total_need = need_chain + need_comp + BG_ENTS
    if total_need > len(ENTITY_POOL):
        return None
    ents = rng.sample(ENTITY_POOL, total_need)
    it = iter(ents)
    chain = [next(it) for _ in range(need_chain)]
    e0, e_answer = chain[0], chain[-1]
    chain_facts = [Fact(chain[i], goal[i], chain[i + 1]) for i in range(k)]

    comp_facts: list[Fact] = []
    comp_targets: list[str] = []
    for d in depths:
        a = next(it)
        comp_targets.append(a)
        comp_facts.append(Fact(e0, goal[0], a))
        prev = a
        for j in range(d):
            nxt = next(it)
            comp_facts.append(Fact(prev, goal[1 + j], nxt))
            prev = nxt

    bg_ents = [next(it) for _ in range(BG_ENTS)]
    bg_facts = random_bg_facts(rng, bg_ents, n_bg)

    fact_list = list(dict.fromkeys(chain_facts + comp_facts + bg_facts))
    fact_set = set(fact_list)
    rng.shuffle(fact_list)

    # 候选 = e0 的全部 goal[0] 出边；正确目标 = 链上第一个实体。
    depth_of = {t: d for t, d in zip(comp_targets, depths)}
    cand_targets = sorted(edges_from(fact_set, e0, goal[0]))
    if len(cand_targets) != n_cand or chain[1] not in cand_targets:
        return None
    correct_target = chain[1]

    # solver 校验（在完整事实集上重推，不信任构造过程）：
    # 1) 答案唯一：reachable(e0, goal) == {e_answer}
    if all_reachable(fact_set, e0, goal) != {e_answer}:
        return None
    # 2) 动作唯一：恰有一个候选能从自身走通剩余路径到达 e_answer。
    reach_flags = [e_answer in all_reachable(fact_set, t, list(goal[1:]))
                   for t in cand_targets]
    if sum(reach_flags) != 1:
        return None
    # 3) 候选闭包：e0 的 goal[0] 出边集与候选一致（edges_from 已保证）。
    order = list(range(n_cand))
    rng.shuffle(order)
    candidates = tuple((goal[0], cand_targets[i]) for i in order)
    answer_index = next(i for i, (_, t) in enumerate(candidates)
                        if t == correct_target)
    comp_depth_by_idx = tuple(
        (i, depth_of[t])
        for i, (_, t) in enumerate(candidates) if t != correct_target)

    goal_str = ' '.join(goal)
    fact_str = ' ; '.join(f'{f.subj} {f.rel} {f.obj}' for f in fact_list)
    cand_str = ' , '.join(f'{r} {t}' for r, t in candidates)
    question = (f'[STATE] {fact_str} [CURRENT] {e0} [GOAL] {goal_str} '
                f'[ACTIONS] {cand_str} [ANSWER]')

    content = {
        'goal': list(goal), 'e0': e0,
        'facts': sorted([f.__dict__ for f in fact_list],
                        key=lambda d: (d['subj'], d['rel'], d['obj'])),
    }
    ex_id = sha256_text(canonical_json(content))[:16]
    return Example(
        ex_id=ex_id, split=split, goal_name=goal_name, goal=tuple(goal),
        hop=k, e0=e0, e_answer=e_answer, facts=tuple(fact_list),
        candidates=candidates, answer_index=answer_index,
        n_candidates=n_cand, n_distractors=len(bg_facts),
        comp_depth_by_index=comp_depth_by_idx, proof=tuple(chain_facts),
        question=question,
    )


def generate_split(split: str, goal_name: str, n: int, n_cand_range,
                   n_bg_range, seed: int) -> list[Example]:
    rng = random.Random(seed)
    out: dict[str, Example] = {}
    attempts = 0
    max_attempts = max(2000, n * 100)
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        n_cand = rng.choice(n_cand_range)
        n_bg = rng.choice(n_bg_range)
        ex = make_example(rng, split, goal_name, n_cand, n_bg)
        if ex is None:
            continue
        out[ex.ex_id] = ex
    if len(out) < n:
        raise RuntimeError(f'生成不足：{split}/{goal_name} {len(out)}/{n}')
    return list(out.values())


def example_to_json(ex: Example) -> dict:
    return {
        'id': ex.ex_id,
        'split': ex.split,
        'goal_name': ex.goal_name,
        'goal': list(ex.goal),
        'hop': ex.hop,
        'e0': ex.e0,
        'e_answer': ex.e_answer,
        'candidates': [{'rel': r, 'obj': t} for r, t in ex.candidates],
        'answer_index': ex.answer_index,
        'n_candidates': ex.n_candidates,
        'n_distractors': ex.n_distractors,
        'comp_depth_by_index': {str(i): d for i, d in ex.comp_depth_by_index},
        'proof': [f.__dict__ for f in ex.proof],
        'facts': [f.__dict__ for f in ex.facts],
        'question': ex.question,
    }


def verify_examples(rows: Iterable[Example]) -> dict:
    """独立复检：从存储字段重推答案唯一性与动作唯一性。"""
    total = bad_answer = bad_action = bad_closure = 0
    bad_samples: list[dict] = []
    for ex in rows:
        total += 1
        facts = set(ex.facts)
        goal = list(ex.goal)
        reach = all_reachable(facts, ex.e0, goal)
        cand_targets = sorted(edges_from(facts, ex.e0, goal[0]))
        shown = sorted(t for _, t in ex.candidates)
        ok = True
        if reach != {ex.e_answer}:
            bad_answer += 1
            ok = False
        flags = [ex.e_answer in all_reachable(facts, t, goal[1:])
                 for t in shown]
        correct_shown = shown[flags.index(True)] if any(flags) else None
        if (sum(flags) != 1 or ex.answer_index >= ex.n_candidates
                or ex.candidates[ex.answer_index][1] != correct_shown):
            bad_action += 1
            ok = False
        if cand_targets != shown:
            bad_closure += 1
            ok = False
        if not ok and len(bad_samples) < 5:
            bad_samples.append({'id': ex.ex_id, 'goal_name': ex.goal_name})
    return {'checked': total, 'bad_answer': bad_answer,
            'bad_action': bad_action, 'bad_closure': bad_closure,
            'examples': bad_samples}


def canonical_key(ex: Example) -> str:
    return sha256_text(canonical_json({
        'goal': list(ex.goal), 'e0': ex.e0,
        'facts': sorted([f.__dict__ for f in ex.facts],
                        key=lambda d: (d['subj'], d['rel'], d['obj'])),
    }))


def structure_proxy(e: Example) -> tuple:
    """结构同构的粗代理：goal + 事实关系多重集 + 候选深度轮廓 + 候选数。
    非完整图同构检查；仅用于报告潜在结构泄漏的下界。"""
    return (tuple(e.goal),
            tuple(sorted(f.rel for f in e.facts)),
            tuple(sorted(d for _, d in e.comp_depth_by_index)),
            e.n_candidates)


def cross_split_audit(train: list[Example],
                      evals: dict[str, list[Example]]) -> dict:
    train_keys = {canonical_key(e) for e in train}
    train_fact_sets = {frozenset(e.facts) for e in train}
    train_struct = Counter(structure_proxy(e) for e in train)
    train_entities = set().union(*(set(f.subj for f in e.facts)
                                   | set(f.obj for f in e.facts)
                                   for e in train))
    report = {}
    for name, rows in evals.items():
        keys = {canonical_key(e) for e in rows}
        same_id = sum(1 for e in rows if e.ex_id in {t.ex_id for t in train})
        fact_overlap = sum(1 for e in rows
                           if frozenset(e.facts) in train_fact_sets)
        struct_overlap = sum(1 for e in rows
                             if structure_proxy(e) in train_struct)
        ents = set().union(*(set(f.subj for f in e.facts)
                             | set(f.obj for f in e.facts) for e in rows))
        report[name] = {
            'n': len(rows),
            'same_logical_id_overlap': same_id,
            'canonical_key_overlap': len(keys & train_keys),
            'identical_fact_set_overlap': fact_overlap,
            'structure_proxy_overlap': struct_overlap,
            'entity_pool_overlap_count': len(train_entities & ents),
        }
    return report


def difficulty_stats(rows: Iterable[Example]) -> dict:
    rows = list(rows)
    ans_pos = Counter(e.answer_index for e in rows)
    comp_depths = Counter(d for e in rows for _, d in e.comp_depth_by_index)
    sym_len = [len(e.question.split()) for e in rows]
    n_cand = [e.n_candidates for e in rows]
    return {
        'n': len(rows),
        'goal_counts': dict(sorted(Counter(e.goal_name for e in rows).items())),
        'hop_counts': dict(sorted(Counter(e.hop for e in rows).items())),
        'n_candidates_hist': dict(sorted(Counter(n_cand).items())),
        'mean_candidates': round(statistics.mean(n_cand), 2),
        'majority_baseline': round(1.0 / statistics.mean(n_cand), 3),
        'answer_position_hist': dict(sorted(ans_pos.items())),
        'comp_depth_hist': dict(sorted(comp_depths.items())),
        'bg_distractors_mean': round(statistics.mean(
            e.n_distractors for e in rows), 2),
        'n_facts_mean': round(statistics.mean(len(e.facts) for e in rows), 2),
        'question_symbols_p50': int(statistics.median(sym_len)),
        'question_symbols_p95': sorted(sym_len)[int(0.95 * len(sym_len)) - 1],
    }


def write_jsonl(path: Path, rows: Iterable[Example]):
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for e in rows:
            f.write(json.dumps(example_to_json(e), ensure_ascii=False) + '\n')


def append_history(rec: dict):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with (RUN_DIR / 'history.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def build_dataset(args) -> dict[str, list[Example]]:
    splits: dict[str, list[Example]] = {}

    def gen(split: str, goals: list[str], per: int, cand_rng, bg_rng,
            seed_off: int):
        rows: list[Example] = []
        for j, g in enumerate(goals):
            rows.extend(generate_split(
                split, g, per, cand_rng, bg_rng,
                seed=SEED + seed_off + j))
        return rows

    splits['train'] = gen('train', TRAIN_PATHS, args.train_per_path,
                          (3, 4, 5), (2, 3, 4), 1000)
    splits['dev_random'] = gen('dev_random', TRAIN_PATHS, args.dev_per_path,
                               (3, 4, 5), (2, 3, 4), 2000)
    splits['confirm_depth'] = gen('confirm_depth', DEPTH_HOLDOUT,
                                  args.depth_per_path,
                                  (3, 4, 5), (2, 3, 4), 3000)
    splits['confirm_combo'] = gen('confirm_combo', COMBO_HOLDOUT,
                                  args.combo_per_path,
                                  (3, 4, 5), (2, 3, 4), 4000)
    splits['confirm_distractor'] = gen('confirm_distractor', TRAIN_PATHS,
                                       args.distractor_per_path,
                                       (6, 7, 8), (8, 9, 10), 5000)
    splits['pilot'] = gen('pilot', PILOT_PATHS, args.pilot_per_path,
                          (3, 4, 5), (2, 3, 4), 6000)

    # 一致性门禁。
    train_hop_max = max(HOP_OF[g] for g in TRAIN_PATHS)
    depth_hop_min = min(HOP_OF[g] for g in DEPTH_HOLDOUT)
    assert train_hop_max < depth_hop_min, (
        f'深度外推不严格：train hop max={train_hop_max} '
        f'>= confirm_depth hop min={depth_hop_min}')
    combo_hops = {HOP_OF[g] for g in COMBO_HOLDOUT}
    assert combo_hops <= {HOP_OF[g] for g in TRAIN_PATHS}, (
        '组合留出 hop 超出训练 hop 范围')
    train_goals = {tuple(PATHS[g]) for g in TRAIN_PATHS}
    for g in COMBO_HOLDOUT + DEPTH_HOLDOUT:
        assert tuple(PATHS[g]) not in train_goals, f'{g} 与训练路径重复'
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-per-path', type=int, default=1200)
    ap.add_argument('--dev-per-path', type=int, default=200)
    ap.add_argument('--depth-per-path', type=int, default=125)
    ap.add_argument('--combo-per-path', type=int, default=80)
    ap.add_argument('--distractor-per-path', type=int, default=80)
    ap.add_argument('--pilot-per-path', type=int, default=20)
    ap.add_argument('--data-dir', default='',
                    help='输出目录（默认 data/ldm_edge_action）；'
                         '版本化数据集用不同目录，保留旧版')
    ap.add_argument('--entity-pool', type=int, default=300,
                    help='实体池大小（诊断用；缩小=降低绑定难度）')
    ap.add_argument('--write', action='store_true',
                    help='写出正式数据文件；否则只标定统计')
    args = ap.parse_args()

    if args.data_dir:
        globals()['DATA_DIR'] = Path(args.data_dir)
    if args.entity_pool != 300:
        globals()['ENTITY_POOL'] = [f'E{i:03d}'
                                    for i in range(args.entity_pool)]

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    append_history({'event': 'start', 'phase': 'LDM task generation',
                    'write': bool(args.write), 'seed': SEED})

    splits = build_dataset(args)
    uniq = {name: verify_examples(rows) for name, rows in splits.items()}
    audit = cross_split_audit(splits['train'],
                              {k: v for k, v in splits.items()
                               if k != 'train'})
    stats = {name: difficulty_stats(rows) for name, rows in splits.items()}

    file_hashes = {}
    if args.write:
        for name, rows in splits.items():
            p = DATA_DIR / f'{name}.jsonl'
            write_jsonl(p, rows)
            file_hashes[name] = sha256_text(p.read_text(encoding='utf-8'))

    manifest = {
        'seed': SEED,
        'script': Path(__file__).name,
        'script_sha256': sha256_text(
            Path(__file__).read_text(encoding='utf-8')),
        'task': 'edge_choice_decision (non-functional graph)',
        'entity_pool_size': len(ENTITY_POOL),
        'paths': {k: list(v) for k, v in PATHS.items()},
        'train_paths': TRAIN_PATHS,
        'combo_holdout': COMBO_HOLDOUT,
        'depth_holdout': DEPTH_HOLDOUT,
        'candidate_policy': {'train/dev/depth/combo': [3, 4, 5],
                             'confirm_distractor': [6, 7, 8]},
        'bg_distractor_policy': {'train/dev/depth/combo': [2, 3, 4],
                                 'confirm_distractor': [8, 9, 10]},
        'splits': {k: {'n': len(v), 'sha256': file_hashes.get(k)}
                   for k, v in splits.items()},
        'uniqueness_check': uniq,
        'leakage_audit': audit,
        'note': '纯 CPU 合成决策任务；答案唯一 + 动作唯一由确定性 solver '
                '在完整事实集上复检；竞争边断链深度为第二难度轴。',
    }
    if args.write:
        (DATA_DIR / 'manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8')

    report = {
        'args': vars(args), 'stats': stats,
        'uniqueness_check': uniq, 'leakage_audit': audit,
        'files': {k: str(DATA_DIR / f'{k}.jsonl') for k in splits},
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                   encoding='utf-8')
    append_history({'event': 'final', 'write': bool(args.write),
                    'stats': {k: v['n'] for k, v in stats.items()},
                    'bad': {k: (v['bad_answer'], v['bad_action'],
                                v['bad_closure']) for k, v in uniq.items()},
                    'canonical_overlap': {k: v['canonical_key_overlap']
                                          for k, v in audit.items()}})
    print(json.dumps({'stats': stats,
                      'uniq_bad': {k: (v['bad_answer'], v['bad_action'],
                                       v['bad_closure'])
                                   for k, v in uniq.items()},
                      'audit': audit}, ensure_ascii=False, indent=2))
    print('LDM task generation done ->', OUT, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
