"""s61: T-14 确认集评测（深度/组合/长度三轴，纯推理）。

用 T-14 已训练的 4 个 checkpoint 在三个确认 split 上评测：
  confirm_depth  : hop 3–6（训练只见 hop2）→ 严格深度外推
  confirm_combo  : hop2 但 P,S 组合留出 → 组合泛化
  confirm_length : hop2 但干扰事实加倍 → 长度鲁棒性
  dev_random     : 同训练分布（参照）

输出：results/t14_confirm_sets.json
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from s60_t14_param_eff_grid import (build_arm, evaluate_arm)  # noqa: E402
from s48_t5_eos_patch_train import append_jsonl  # noqa: E402
from s59_t13_gpu_pilot import (CTX, N_LABELS, load_rows,  # noqa: E402
                               load_tok)

RUN = ROOT / 'runs' / 't14_param_eff'
OUT = ROOT / 'results' / 't14_confirm_sets.json'
CELLS = [('dense', 1), ('dense', 2), ('loop', 1), ('loop', 2)]
SPLITS = ['dev_random', 'confirm_combo', 'confirm_length', 'confirm_depth']


@torch.no_grad()
def main():
    tok = load_tok()
    digit_ids = [tok.encode(' ' + str(i + 1))[-1] for i in range(N_LABELS)]
    rows = {s: load_rows(s, 200 if s == 'dev_random' else 120)
            for s in SPLITS}
    append_jsonl(RUN / 'history.jsonl',
                 {'event': 'confirm_start', 'cells': CELLS,
                  'splits': {k: len(v) for k, v in rows.items()},
                  'ts': time.strftime('%F %T')})

    results = []
    for arm, r in CELLS:
        ck_path = RUN / f'ckpt_{arm}_r{r}.pt'
        if not ck_path.exists():
            print(f'skip {arm} r={r}: no ckpt', flush=True)
            continue
        model, fresh, depth, bypass, n_train, span = build_arm(arm, r)
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        with torch.no_grad():
            for p, saved in zip(fresh, ck['fresh']):
                p.data.copy_(saved.to(p.device))
        model.eval()
        rec = {'arm': arm, 'r': r, 'n_trainable': n_train,
               'span': span, 'depth': depth,
               'trained_step': ck['step']}
        for s, rr in rows.items():
            acc, _ = evaluate_arm(model, tok, rr, depth, bypass, digit_ids)
            rec[s] = round(acc, 4)
            print(f'{arm} r={r} {s}: {acc:.4f}', flush=True)
        results.append(rec)
        append_jsonl(RUN / 'history.jsonl',
                     {'event': 'confirm_cell', **{k: v for k, v in rec.items()
                                                  if k != 'span'},
                      'ts': time.strftime('%F %T')})
        del model, fresh
        torch.cuda.empty_cache()

    report = {'config': {'ctx': CTX, 'splits': {k: len(v)
                                                for k, v in rows.items()},
                         'note': '同一 checkpoint 上跨 split 对比；'
                                 'confirm_depth 为严格外推'},
              'cells': results, 'ts': time.strftime('%F %T')}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding='utf-8')
    print('T-14 confirm done ->', OUT, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
