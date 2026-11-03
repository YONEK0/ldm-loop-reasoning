"""s73_ldm_dashboard.py — LDM 系列训练实时面板（只读，独立进程，不碰 CUDA）。

监视 results/ 下的 LDM 训练 json（s71_ldm_iter4.py 输出），按文件 mtime 判定
"运行中 / 已停"，提供：

  - 每 run 状态卡：最新步数、cap、loss、运行时长、是否在推进
  - 曲线：dev_r{1..8} 精度 vs step（对角线出现与否的直接视图）
  - by_hop 热力：hop × r 的精度矩阵（对角线应当高亮）
  - 退出控制器：e_halt / KL / q_mean_first 实时曲线（仅 ponder 路径）
  - 判读区：ponder_eval 的预注册四判据

用法：
  python s73_ldm_dashboard.py --port 8752
  浏览器打开 http://127.0.0.1:8752
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'

# 面板纳管的目标（按优先级展示）
RUNS = [
    ('iter4_v8_s53_lr5e4', 'D2 LR扫描 seed53 lr=5e-4'),
    ('iter4_v8_s53_lr3e4', 'D2 LR扫描 seed53 lr=3e-4'),
    ('iter4_v7_exactT_s53', 'V7 exact-T seed53 lr=1e-3（失败基准）'),
    ('iter4_v7_exactT_s41', 'V7 exact-T seed41（五级全解）'),
    ('iter4_v7_exactT_s31', 'V7 exact-T seed31（只到 hop3）'),
    ('iter4_v1_none_s17', 'V1 exact-T seed17（对角线基准）'),
    ('iter4_v5b_ponder_s17', 'V5b PonderNet seed17'),
    ('iter4_v5b_ponder_s23', 'V5b PonderNet seed23'),
]

STALE_SEC = 180      # 超过这么久没更新 → 判为已停


def load_run(name: str):
    p = RES / f'{name}.json'
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None      # 正在写、暂时不完整
    d['_mtime'] = p.stat().st_mtime
    d['_age'] = time.time() - d['_mtime']
    d['_alive'] = d['_age'] < STALE_SEC
    return d


def summarize(name: str, label: str):
    d = load_run(name)
    if d is None:
        return {'name': name, 'label': label, 'missing': True}
    hist = d.get('hist') or []
    ev = d.get('evals') or d.get('last_evals') or {}
    out = {
        'name': name, 'label': label, 'alive': d['_alive'],
        'age_sec': round(d['_age'], 1),
        'step': hist[-1].get('step') if hist else None,
        'steps_target': (d.get('args') or {}).get('steps'),
        'cap': hist[-1].get('cap') if hist else None,
        'loss': hist[-1].get('loss') if hist else None,
        'sec': hist[-1].get('sec') if hist else None,
        'hist': hist,
        'eval_step': ev.get('step'),
        'rounds': {f'r{r}': ev.get(f'dev_r{r}') for r in (1, 2, 3, 4, 5, 6, 8)
                   if f'dev_r{r}' in ev},
        'by_hop': {f'r{r}': ev.get(f'dev_r{r}_by_hop') for r in (1, 2, 3, 4, 5, 6, 8)
                   if f'dev_r{r}_by_hop' in ev},
        'confirm': {f'r{r}': ev.get(f'confirm_r{r}') for r in (1, 2, 3, 4, 5, 6, 8)
                    if f'confirm_r{r}' in ev},
    }
    pe = d.get('ponder_eval')
    if pe:
        out['ponder'] = {
            'oracle': pe.get('oracle_fixed_r'),
            'fixed_r8': pe.get('fixed_r8'),
            'feasible': pe.get('feasible_round_by_hop'),
            'tau_sweep': [{k: s.get(k) for k in
                           ('tau', 'acc', 'mean_rounds', 'match_feasible_rate',
                            'stop_match_rate', 'collapse', 'exit_hist', 'by_hop')}
                          for s in pe.get('tau_sweep', [])],
            'verdict': pe.get('prereg_verdict'),
        }
    return out


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>LDM Training Dashboard</title>
<style>
 :root{--bg:#0d1117;--fg:#e6edf3;--dim:#7d8590;--card:#161b22;--line:#30363d;
       --ok:#3fb950;--warn:#d29922;--bad:#f85149;--acc:#58a6ff}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:13px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}
 header{padding:14px 20px;border-bottom:1px solid var(--line);
        display:flex;align-items:center;gap:16px;flex-wrap:wrap}
 h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.4px}
 .pill{padding:2px 9px;border-radius:10px;font-size:11px;border:1px solid var(--line)}
 .alive{color:var(--ok);border-color:var(--ok)}
 .dead{color:var(--dim)}
 main{padding:16px 20px;display:grid;gap:16px;
      grid-template-columns:repeat(auto-fit,minmax(430px,1fr))}
 .card{background:var(--card);border:1px solid var(--line);border-radius:8px;
       padding:14px;min-width:0}
 .hd{display:flex;justify-content:space-between;align-items:baseline;
     margin-bottom:10px;gap:8px;flex-wrap:wrap}
 .title{font-weight:600;font-size:13px}
 .meta{color:var(--dim);font-size:11px}
 .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0}
 .stat{background:#0b0f14;border:1px solid var(--line);border-radius:6px;padding:7px 9px}
 .stat .k{color:var(--dim);font-size:10px;text-transform:uppercase;letter-spacing:.5px}
 .stat .v{font-size:15px;font-weight:600;margin-top:2px}
 canvas{width:100%;height:150px;display:block}
 table{width:100%;border-collapse:collapse;font-size:11px}
 th,td{padding:3px 5px;text-align:center;border-bottom:1px solid #21262d}
 th{color:var(--dim);font-weight:500}
 td.hot{font-weight:700}
 .verdict{margin-top:8px;font-size:11px;line-height:1.7}
 .yes{color:var(--ok)}.no{color:var(--bad)}.neu{color:var(--warn)}
 .scroll{max-height:120px;overflow:auto}
</style></head><body>
<header>
  <h1>LDM TRAINING DASHBOARD</h1>
  <span class="pill" id="clock">--:--:--</span>
  <span class="meta" id="legend">对角线视图：dev_r{r} = 固定跑 r 轮的精度；by_hop 热力中 hop=T 行在 r=T 列应显著更高</span>
</header>
<main id="root"></main>
<script>
const ROUNDS=[1,2,3,4,5,6,8];
const COLORS={r1:'#8b949e',r2:'#58a6ff',r3:'#3fb950',r4:'#d29922',r5:'#f85149',r6:'#bc8cff',r8:'#ff7b72'};

function heat(v){
  if(v==null) return '#161b22';
  const t=Math.max(0,Math.min(1,(v-0.5)/0.5));
  const r=Math.round(20+t*30), g=Math.round(60+t*150), b=Math.round(60+t*80);
  return `rgb(${r},${g},${b})`;
}

function chart(run){
  const h=run.hist||[];
  if(h.length<2) return '<div class="meta">尚无足够数据点</div>';
  const W=430,H=150,P=28;
  const steps=h.map(x=>x.step);
  const xmin=Math.min(...steps), xmax=Math.max(...steps);
  const keys=ROUNDS.map(r=>'r'+r);
  let svg=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
  // axes
  svg+=`<line x1="${P}" y1="${H-16}" x2="${W-6}" y2="${H-16}" stroke="#30363d"/>`;
  svg+=`<line x1="${P}" y1="6" x2="${P}" y2="${H-16}" stroke="#30363d"/>`;
  for(const v of [0.5,0.75,1.0]){
    const y=H-16-(v)* (H-26);
    svg+=`<line x1="${P}" y1="${y}" x2="${W-6}" y2="${y}" stroke="#21262d" stroke-dasharray="2,3"/>`;
    svg+=`<text x="2" y="${y+3}" fill="#7d8590" font-size="8">${v.toFixed(2)}</text>`;
  }
  for(const k of keys){
    let pts=[];
    for(const rec of h){
      const v=rec[k];
      if(v==null) continue;
      const x=P+(xmax===xmin?0:(rec.step-xmin)/(xmax-xmin))*(W-P-8);
      const y=H-16-v*(H-26);
      pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }
    if(pts.length>1) svg+=`<polyline fill="none" stroke="${COLORS[k]}" stroke-width="1.4" points="${pts.join(' ')}"/>`;
  }
  svg+='</svg>';
  const lg=keys.map(k=>`<span style="color:${COLORS[k]};margin-right:8px">${k}</span>`).join('');
  return svg+`<div class="meta" style="margin-top:4px">${lg}</div>`;
}

function heatTable(run){
  const bh=run.by_hop||{};
  const rs=Object.keys(bh).filter(k=>bh[k]);
  if(!rs.length) return '<div class="meta">无 by_hop 数据</div>';
  const hops=[...new Set(rs.flatMap(k=>Object.keys(bh[k])))].sort((a,b)=>a-b);
  let t='<table><tr><th>hop\\r</th>';
  for(const k of rs) t+=`<th>${k.slice(1)}</th>`;
  t+='</tr>';
  for(const hop of hops){
    t+=`<tr><th>${hop}</th>`;
    for(const k of rs){
      const v=bh[k][hop];
      t+=`<td style="background:${heat(v)}" class="${v>=0.9?'hot':''}">${v==null?'-':v.toFixed(2)}</td>`;
    }
    t+='</tr>';
  }
  return t+'</table>';
}

function ponderBlock(run){
  if(!run.ponder) return '';
  const pe=run.ponder;
  let t=`<div class="hd"><span class="title">退出控制器（PonderNet）</span>
    <span class="meta">oracle ${pe.oracle} · fixed_r8 ${pe.fixed_r8}</span></div>`;
  t+='<table><tr><th>τ</th><th>acc</th><th>mean_r</th><th>match_feas</th><th>塌缩</th></tr>';
  for(const s of pe.tau_sweep){
    t+=`<tr><td>${s.tau}</td><td>${s.acc}</td><td>${s.mean_rounds}</td>
        <td>${s.match_feasible_rate==null?'-':s.match_feasible_rate}</td>
        <td>${s.collapse?'<span class="no">是</span>':'<span class="yes">否</span>'}</td></tr>`;
  }
  t+='</table>';
  const v=pe.verdict||{};
  const items=[['质量',v.quality_pass],['轮数',v.rounds_pass],['省算力',v.saving_vs_r8_pass],
               ['一致率',v.match_feasible_pass],['塌缩(A4)',v.collapse_A4==null?null:!v.collapse_A4]];
  t+='<div class="verdict">'+items.map(([n,ok])=>{
    if(ok==null) return `<span class="neu">${n}: 无</span>`;
    return ok?`<span class="yes">✓ ${n}</span>`:`<span class="no">✗ ${n}</span>`;
  }).join(' &nbsp; ')+'</div>';
  return `<div class="card" style="grid-column:1/-1">${t}</div>`;
}

function card(run){
  if(run.missing) return `<div class="card"><div class="title">${run.label}</div>
    <div class="meta">未找到 results/${run.name}.json</div></div>`;
  const stats=[
    ['step', run.step==null?'-':`${run.step}/${run.steps_target||'?'}`],
    ['cap', run.cap==null?'-':run.cap],
    ['loss', run.loss==null?'-':run.loss],
    ['eval', run.eval_step==null?'-':`@${run.eval_step}`],
  ];
  const st=stats.map(([k,v])=>`<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
  const rr=Object.entries(run.rounds).map(([k,v])=>`${k} ${v}`).join(' · ');
  return `<div class="card">
    <div class="hd">
      <span class="title">${run.label}</span>
      <span>
        <span class="pill ${run.alive?'alive':'dead'}">${run.alive?'RUNNING':'STOPPED'}</span>
        <span class="meta">mtime ${run.age_sec}s ago</span>
      </span>
    </div>
    <div class="stats">${st}</div>
    <div class="meta" style="margin-bottom:6px">fixed-r 精度：${rr||'-'}</div>
    ${chart(run)}
    <div style="margin-top:10px">
      <div class="meta" style="margin-bottom:4px">hop × r 精度热力（对角=循环承重证据）</div>
      <div class="scroll">${heatTable(run)}</div>
    </div>
  </div>`;
}

async function tick(){
  document.getElementById('clock').textContent=new Date().toLocaleTimeString();
  try{
    const r=await fetch('/api/state',{cache:'no-store'});
    const j=await r.json();
    document.getElementById('root').innerHTML=
      j.runs.map(card).join('') + (j.ponder_cards||[]).join('');
  }catch(e){ console.error(e); }
}
tick(); setInterval(tick, 3000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith('/api/state'):
            runs = [summarize(n, l) for n, l in RUNS]
            payload = {'runs': runs, 'ts': time.time()}
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = PAGE.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8752)
    ap.add_argument('--host', default='127.0.0.1')
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), H)
    print(f'[ldm-dashboard] http://{args.host}:{args.port}', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
