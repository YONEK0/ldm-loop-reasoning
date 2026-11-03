"""s70: 文本通道自适应长度的省算力验证（离线分析，无需 GPU/模型）。

输入：4B 臂 A n=150 逐题 sidecar（strict_r{1,2,4,8} + layers_r{1,2,4,8}）。
问题：按难度自适应分配推理长度（简单题 1 轮、难题 8 轮），能省多少
算力、损多少质量——文本通道的"自适应退出"账。

预注册读数（跑前写定，LDM 日志 §16）：
- oracle 上限：逐题取"最早答对的轮数"，报告其精度与算力 vs 固定 r8；
- 保守策略：r1 对即停、否则跑满 r8（无置信度信号的极端保守版）；
- 结论表述：节省 = 1 − mean_layers(policy)/mean_layers(r8)；
  质量差 = acc(policy) − acc(r8)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent
DEFAULT = ROOT / 'pkg_hybridloop4b' / 'server_results_run3' / 'armA_items.jsonl'
OUT = ROOT / 'results' / 'text_adaptive_analysis.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', default=str(DEFAULT))
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()

    items = [json.loads(l) for l in
             Path(args.items).read_text(encoding='utf-8').splitlines() if l.strip()]
    rounds = [1, 2, 4, 8]
    n = len(items)

    def policy(fn):
        acc = lay = 0
        for it in items:
            r_use, ok = fn(it)
            acc += ok
            lay += it[f'layers_r{r_use}']
        return {'acc': round(acc / n, 4),
                'layers_mean': round(lay / n, 1)}

    fixed = {r: policy(lambda it, r=r: (r, it[f'strict_r{r}'])) for r in rounds}

    def oracle(it):
        for r in rounds:
            if it[f'strict_r{r}']:
                return r, True
        return 8, it['strict_r8']

    def conservative(it):
        return (1, True) if it['strict_r1'] else (8, it['strict_r8'])

    def mid(it):   # r1 对即停；否则 r4 对即停；否则 r8
        for r in (1, 4):
            if it[f'strict_r{r}']:
                return r, True
        return 8, it['strict_r8']

    def stability(it):
        """可实现策略（无真值信号）：逐轮取解析出的答案 pred，
        与上一个非空 pred 一致即停；从未稳定则跑满取 r8。
        停在哪轮用哪轮的 correct 记账。"""
        last = None
        for r in rounds:
            pr = it['rounds'][str(r)]['pred']
            if pr is None:
                continue
            if last is not None and pr == last:
                return r, it['rounds'][str(r)]['correct']
            last = pr
        return 8, it['strict_r8']

    pol = {'oracle_min_round': policy(oracle),
           'conservative_1_or_8': policy(conservative),
           'mid_1_4_8': policy(mid),
           'stability_implementable': policy(stability)}

    base_lay = fixed[8]['layers_mean']
    base_acc = fixed[8]['acc']
    savings = {k: round(1 - v['layers_mean'] / base_lay, 4) for k, v in pol.items()}
    dq = {k: round(v['acc'] - base_acc, 4) for k, v in pol.items()}

    rep = {'n': n, 'fixed': fixed, 'policies': pol,
           'saving_vs_r8': savings, 'quality_delta_vs_r8': dq,
           'note': 'oracle=每题取最早答对轮数；文本通道逐题算力口径 '
                   'layers_mean (token-layer)。质量差 0 表示该策略精度 '
                   '不低于固定 r8。'}
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2)
                              + '\n', encoding='utf-8')
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
