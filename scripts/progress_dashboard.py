#!/usr/bin/env python
"""학습이 나아지는 모습을 브라우저에서 그래프로 본다.

learner 가 남기는 outputs/<run>/logs/learner_metrics.jsonl 을 읽어 작은 웹 서버로 띄운다.
VS Code 에서 포트를 포워딩하면 랩탑 브라우저에서 바로 볼 수 있다. 10초마다 자동 갱신된다.

  python scripts/progress_dashboard.py                       # 기본 run, 8088 포트
  python scripts/progress_dashboard.py --run outputs/train/panda_place_gpu --port 8088

읽는 법
  - 성공률이 오르는데 개입률이 같이 내려가면 정책이 스스로 하게 된 것이다.
  - 성공률만 높고 개입률도 높으면 사람이 대신 해 준 것이라 학습 신호로는 약하다.
"""
import argparse
import json
import os
import shutil
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--run", default="outputs/train/panda_place_gpu", help="learner 의 output_dir")
p.add_argument("--port", type=int, default=8088)
p.add_argument("--window", type=int, default=10, help="이동평균에 쓸 에피소드 수")
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--demo", action="store_true",
               help="학습을 시작하기 전에 화면이 어떻게 보이는지 가짜 데이터로 미리 본다")
p.add_argument("--view", default=os.environ.get("HIL_VIEW_STREAM", "/dev/shm/hil_view.jpg"),
               help="조작자용 화면 JPEG 경로 (ViewStreamWrapper 가 쓰는 파일)")
p.add_argument("--keymap", default=os.environ.get("GYM_HIL_KEYMAP", "configs/keymap.json"),
               help="키 배치 파일. 브라우저에서 보낼 수 있는 키 목록을 여기서 읽는다.")
p.add_argument("--display", default=os.environ.get("DISPLAY", ":1"),
               help="키를 넣을 X 디스플레이")
p.add_argument("--no-control", action="store_true",
               help="브라우저에서 키보드로 조작하는 기능을 끈다")
args = p.parse_args()

RUN = Path(args.run)
METRICS = RUN / "logs" / "learner_metrics.jsonl"
VIEW = Path(args.view)
MAX_POINTS = 400

# 브라우저 키 -> X keysym. 여기 없는 키는 전달하지 않는다.
BROWSER_TO_X = {
    "ArrowUp": "Up", "ArrowDown": "Down", "ArrowLeft": "Left", "ArrowRight": "Right",
    "Space": "space", "Enter": "Return", "Escape": "Escape", "Tab": "Tab",
    "ShiftLeft": "Shift_L", "ShiftRight": "Shift_R",
    "ControlLeft": "Control_L", "ControlRight": "Control_R",
    "AltLeft": "Alt_L", "AltRight": "Alt_R", "Backspace": "BackSpace",
    **{f"Key{c}": c.lower() for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
}
# 키맵 이름 -> X keysym
KEYMAP_TO_X = {
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "space": "space", "enter": "Return", "esc": "Escape", "tab": "Tab",
    "backspace": "BackSpace", "shift_l": "Shift_L", "shift_r": "Shift_R",
    "ctrl_l": "Control_L", "ctrl_r": "Control_R", "alt_l": "Alt_L", "alt_r": "Alt_R",
}


def load_allowed_keys():
    """키맵에 쓰인 키만 브라우저에서 보낼 수 있게 허용 목록을 만든다."""
    labels = {
        "forward_x": "앞으로", "backward_x": "뒤로", "forward_y": "왼쪽", "backward_y": "오른쪽",
        "forward_z": "위로", "backward_z": "아래로", "gripper_close": "잡기",
        "gripper_open": "놓기", "intervention": "조종권", "success": "성공 종료",
        "failure": "실패 종료", "rerecord": "다시 녹화",
    }
    try:
        km = json.loads(Path(args.keymap).read_text())
    except Exception:  # noqa: BLE001
        km = {}
    allowed, legend = set(), []
    for action, label in labels.items():
        name = km.get(action)
        if not isinstance(name, str):
            continue
        x = KEYMAP_TO_X.get(name, name if len(name) == 1 else None)
        if not x:
            continue
        allowed.add(x)
        legend.append({"key": name, "x": x, "label": label, "action": action})
    return allowed, legend


ALLOWED_KEYS, KEY_LEGEND = load_allowed_keys()
CONTROL_ON = not args.no_control and bool(ALLOWED_KEYS) and shutil.which("xdotool") is not None


def downsample(xs, step=1):
    if len(xs) <= MAX_POINTS:
        return xs
    k = len(xs) / MAX_POINTS
    return [xs[int(i * k)] for i in range(MAX_POINTS)]


def demo_payload():
    """--demo: 학습 전에 화면 모양을 보기 위한 가짜 데이터."""
    import math

    rate, interv, online, offline, loss = [], [], [], [], []
    step = 0
    for i in range(60):
        prog = i / 59
        step += 180
        rate.append([step, 100 * min(1.0, 0.05 + 0.9 * prog + 0.06 * math.sin(i))])
        interv.append([step, 100 * max(0.0, 0.95 - 0.88 * prog + 0.04 * math.cos(i))])
        online.append([step * 3, min(20000, step * 2.4)])
        offline.append([step * 3, 3600])
        loss.append([step * 3, 1.6 / (1 + 4 * prog) + 0.05 * math.sin(i * 1.7)])
    return {
        "run": "(미리보기 — 실제 학습 데이터가 아니다)", "exists": True, "demo": True,
        "window": args.window, "episodes": 60, "successes": 25,
        "recent_rate": 85.0, "recent_interv": 11.0,
        "opt_step": 32400, "online_size": 16278, "offline_size": 3600,
        "rate": rate, "interv": interv, "raw": [], "online": online,
        "offline": offline, "loss": loss,
    }


def collect():
    if args.demo:
        return demo_payload()
    episodes, train = [], []
    if METRICS.exists():
        with METRICS.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 기록 중인 마지막 줄
                if rec.get("kind") == "interaction":
                    episodes.append(rec)
                elif rec.get("kind") == "train":
                    train.append(rec)

    w = max(1, args.window)
    rate_pts, interv_pts, raw_pts = [], [], []
    succ_hist, iv_hist = [], []
    # x 축은 에피소드 번호다. actor 를 재시작하면 "Interaction step" 이 0 부터 다시 세기
    # 때문에 그것을 x 축으로 쓰면 그래프가 뒤로 감긴다. 에피소드 번호는 항상 단조 증가한다.
    for _i, e in enumerate(episodes, 1):
        step = float(_i)
        rew = float(e.get("Episodic reward") or 0.0)
        ok = 1.0 if rew > 0 else 0.0
        succ_hist.append(ok)
        raw_pts.append([step, ok])
        rate_pts.append([step, 100.0 * sum(succ_hist[-w:]) / len(succ_hist[-w:])])
        ir = e.get("Intervention rate")
        if isinstance(ir, (int, float)):
            iv_hist.append(float(ir))
            interv_pts.append([step, 100.0 * sum(iv_hist[-w:]) / len(iv_hist[-w:])])

    online, offline, loss = [], [], []
    for t in train:
        os_ = t.get("Optimization step")
        if not isinstance(os_, (int, float)):
            continue
        if isinstance(t.get("replay_buffer_size"), (int, float)):
            online.append([float(os_), float(t["replay_buffer_size"])])
        if isinstance(t.get("offline_replay_buffer_size"), (int, float)):
            offline.append([float(os_), float(t["offline_replay_buffer_size"])])
        for k in ("loss_critic", "critic_loss"):
            if isinstance(t.get(k), (int, float)):
                loss.append([float(os_), float(t[k])])
                break

    last_train = train[-1] if train else {}
    recent = succ_hist[-20:]
    recent_iv = iv_hist[-20:]
    return {
        "run": str(RUN),
        "exists": METRICS.exists(),
        "demo": False,
        "window": w,
        "episodes": len(episodes),
        "successes": int(sum(succ_hist)),
        "recent_rate": (100.0 * sum(recent) / len(recent)) if recent else None,
        "recent_interv": (100.0 * sum(recent_iv) / len(recent_iv)) if recent_iv else None,
        "opt_step": last_train.get("Optimization step"),
        "online_size": last_train.get("replay_buffer_size"),
        "offline_size": last_train.get("offline_replay_buffer_size"),
        "rate": downsample(rate_pts),
        "interv": downsample(interv_pts),
        "raw": downsample(raw_pts),
        "online": downsample(online),
        "offline": downsample(offline),
        "loss": downsample(loss),
    }


PAGE = r"""<!doctype html>
<html lang="ko"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --surface:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --ink3:#8a8985;
  --line:#e4e3df; --grid:#efeeea;
  --s1:#2a78d6; --s2:#eb6834;
  --good:#0ca30c;
  color-scheme:light;
}
@media (prefers-color-scheme:dark){:root{
  --surface:#1a1a19; --panel:#232322; --ink:#ffffff; --ink2:#c3c2b7; --ink3:#8f8e86;
  --line:#34342f; --grid:#2a2a27;
  --s1:#3987e5; --s2:#d95926;
  color-scheme:dark;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
  font:14px/1.6 "IBM Plex Sans KR",system-ui,-apple-system,"Segoe UI",sans-serif;
  padding:20px 16px 48px}
.wrap{max-width:900px;margin:0 auto}
h1{font-size:19px;margin:0 0 2px;font-weight:600;letter-spacing:-.01em}
.sub{color:var(--ink2);font-size:12.5px;margin:0 0 18px;
  font-family:ui-monospace,"SFMono-Regular",Menlo,monospace}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:22px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px 12px}
.tile .lab{font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--ink3);
  font-family:ui-monospace,Menlo,monospace}
.tile .val{font-size:23px;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.25}
.tile .val small{font-size:12px;font-weight:400;color:var(--ink2)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;
  padding:14px 14px 8px;margin-bottom:16px}
.card h2{font-size:14px;margin:0 0 1px;font-weight:600}
.card .hint{font-size:12px;color:var(--ink2);margin:0 0 10px}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:0 0 6px;font-size:12px;color:var(--ink2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.swatch{width:11px;height:11px;border-radius:2px;flex:none}
svg{display:block;width:100%;height:auto;overflow:visible}
.empty{color:var(--ink2);font-size:13px;padding:22px 2px;text-align:center}
.banner{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--s2);
  border-radius:0 5px 5px 0;padding:10px 13px;margin:0 0 16px;font-size:13px;color:var(--ink2)}
.banner.ok{border-left-color:var(--good);display:none}
.banner b{color:var(--ink)}
.ctlbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:12px 0 8px}
.ctlbar button{font:inherit;font-size:13px;cursor:pointer;padding:5px 12px;border-radius:5px;
  border:1px solid var(--line);background:var(--surface);color:var(--ink)}
.ctlbar button.on{background:var(--s1);border-color:var(--s1);color:#fff}
.ctlbar span{font-size:12px;color:var(--ink2)}
.keys{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px}
.keys div{font-size:11.5px;color:var(--ink2);border:1px solid var(--line);border-radius:4px;
  padding:2px 7px;background:var(--surface)}
.keys div.hit{background:var(--s1);border-color:var(--s1);color:#fff}
.keys b{font-family:ui-monospace,Menlo,monospace;color:var(--ink)}
.foot{color:var(--ink3);font-size:11.5px;font-family:ui-monospace,Menlo,monospace;margin-top:18px}
.tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);
  border-radius:5px;padding:6px 9px;font-size:12px;box-shadow:0 3px 12px rgba(0,0,0,.13);
  font-variant-numeric:tabular-nums;z-index:9;display:none;white-space:nowrap}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:4px}
th,td{text-align:right;padding:4px 8px;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
th{color:var(--ink3);font-weight:500;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase}
details{margin-top:6px}summary{cursor:pointer;font-size:12px;color:var(--ink2)}
</style></head><body>
<div class="wrap">
  <h1>학습 진행</h1>
  <p class="sub" id="runline">__RUN__</p>
  <div class="banner" id="banner">데이터를 읽는 중…</div>
  <noscript><p class="banner">이 페이지는 자바스크립트로 그래프를 그린다. 브라우저에서 자바스크립트를 켜야 한다.</p></noscript>
  <div class="card" id="viewcard" hidden>
    <h2>지금 시뮬레이터</h2>
    <p class="hint">이 <strong>화면을 한 번 클릭</strong>하면 여기서 바로 조작할 수 있다(방향키가 페이지를 움직이지 않는다). 먼저 <kbd>Space</kbd>.</p>
    <img id="view" alt="시뮬레이터 화면" style="width:100%;max-width:560px;border-radius:5px;display:block">
    <div id="ctl" hidden>
      <div class="ctlbar">
        <button type="button" id="ctlbtn">이 창에서 조작하기</button>
        <span id="ctlstate">꺼짐 — 켜면 이 브라우저 창의 키 입력이 로봇으로 간다</span>
      </div>
      <div class="keys" id="keys"></div>
    </div>
  </div>

  <div class="tiles" id="tiles"></div>

  <div class="card">
    <h2>성공률과 개입률</h2>
    <p class="hint">최근 <span id="w">10</span>개 에피소드의 이동평균. <strong>개입률이 내려가면서 성공률이 오르는 것</strong>이 정책이 배우고 있다는 신호다.</p>
    <div class="legend">
      <span><i class="swatch" style="background:var(--s1)"></i>성공률</span>
      <span><i class="swatch" style="background:var(--s2)"></i>개입률</span>
    </div>
    <div id="c1"></div>
    <details><summary>숫자로 보기</summary><div id="t1"></div></details>
  </div>

  <div class="card">
    <h2>버퍼 크기</h2>
    <p class="hint">시연 버퍼는 녹화분이라 크기가 고정이다. 온라인 버퍼는 0에서 차오르다 한도에 닿으면 오래된 것부터 밀려난다. 학습 배치는 둘에서 절반씩 뽑는다.</p>
    <div class="legend">
      <span><i class="swatch" style="background:var(--s1)"></i>온라인</span>
      <span><i class="swatch" style="background:var(--s2)"></i>시연</span>
    </div>
    <div id="c2"></div>
  </div>

  <div class="card">
    <h2>critic loss</h2>
    <p class="hint">가치 함수의 학습 오차. 요동치는 건 정상이고, 발산(계속 커짐)만 아니면 된다.</p>
    <div id="c3"></div>
  </div>

  <p class="foot" id="foot"></p>
</div>
<div class="tip" id="tip"></div>

<script>
const TIP = document.getElementById('tip');
const NS = 'http://www.w3.org/2000/svg';
const W = 860, H = 250, M = {t:12, r:16, b:34, l:52};

function el(n, attrs, txt){
  const e = document.createElementNS(NS, n);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (txt != null) e.textContent = txt;
  return e;
}
function nice(v){
  if (v >= 1000000) return (v/1000000).toFixed(1)+'M';
  if (v >= 1000) return (v/1000).toFixed(v>=10000?0:1)+'k';
  if (Math.abs(v) >= 10) return Math.round(v).toString();
  if (Math.abs(v) >= 1) return v.toFixed(1);
  return v.toFixed(3);
}
function ticks(lo, hi, n){
  if (hi <= lo) return [lo];
  const raw = (hi-lo)/n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw/mag, s = norm<1.5?1:norm<3?2:norm<7?5:10, stepv = s*mag;
  const out=[]; for(let v=Math.ceil(lo/stepv)*stepv; v<=hi+1e-9; v+=stepv) out.push(v);
  return out;
}

// series: [{pts:[[x,y],...], color:'var(--s1)', name:'성공률'}]
function chart(mount, series, opts){
  const host = document.getElementById(mount);
  host.textContent = '';
  const live = series.filter(s => s.pts && s.pts.length);
  if (!live.length){
    const d = document.createElement('div');
    d.className = 'empty';
    d.textContent = opts.empty || '아직 데이터가 없다.';
    host.appendChild(d);
    return;
  }
  let xmin=Infinity, xmax=-Infinity, ymin=Infinity, ymax=-Infinity;
  for (const s of live) for (const [x,y] of s.pts){
    if(x<xmin)xmin=x; if(x>xmax)xmax=x; if(y<ymin)ymin=y; if(y>ymax)ymax=y;
  }
  if (opts.y0to100){ ymin=0; ymax=100; }
  else { if (ymin>0) ymin=0; if (ymax===ymin) ymax=ymin+1; ymax += (ymax-ymin)*0.08; }
  if (xmax===xmin) xmax = xmin+1;

  const px = x => M.l + (x-xmin)/(xmax-xmin)*(W-M.l-M.r);
  const py = y => H-M.b - (y-ymin)/(ymax-ymin)*(H-M.t-M.b);

  const svg = el('svg', {viewBox:`0 0 ${W} ${H}`, role:'img',
                         'aria-label':opts.label || mount});
  // y 그리드 + 눈금
  for (const t of ticks(ymin, ymax, 4)){
    svg.appendChild(el('line', {x1:M.l, x2:W-M.r, y1:py(t), y2:py(t),
      stroke:'var(--grid)', 'stroke-width':1}));
    const lab = el('text', {x:M.l-8, y:py(t)+4, 'text-anchor':'end',
      fill:'var(--ink3)', 'font-size':11}, opts.y0to100 ? Math.round(t)+'%' : nice(t));
    svg.appendChild(lab);
  }
  // x 눈금
  for (const t of ticks(xmin, xmax, 5)){
    svg.appendChild(el('text', {x:px(t), y:H-M.b+18, 'text-anchor':'middle',
      fill:'var(--ink3)', 'font-size':11}, nice(t)));
  }
  svg.appendChild(el('text', {x:(M.l+W-M.r)/2, y:H-2, 'text-anchor':'middle',
    fill:'var(--ink3)', 'font-size':11}, opts.xlabel || ''));
  // 축선
  svg.appendChild(el('line', {x1:M.l, x2:W-M.r, y1:H-M.b, y2:H-M.b,
    stroke:'var(--line)', 'stroke-width':1}));

  for (const s of live){
    let d = '';
    s.pts.forEach(([x,y],i) => { d += (i?'L':'M') + px(x).toFixed(1) + ' ' + py(y).toFixed(1); });
    svg.appendChild(el('path', {d, fill:'none', stroke:s.color, 'stroke-width':2,
      'stroke-linejoin':'round', 'stroke-linecap':'round'}));
    const last = s.pts[s.pts.length-1];
    svg.appendChild(el('circle', {cx:px(last[0]), cy:py(last[1]), r:4.5,
      fill:s.color, stroke:'var(--panel)', 'stroke-width':2}));
  }

  // 크로스헤어 + 툴팁
  const cross = el('line', {y1:M.t, y2:H-M.b, stroke:'var(--ink3)',
    'stroke-width':1, 'stroke-dasharray':'3 3', opacity:0});
  svg.appendChild(cross);
  const hit = el('rect', {x:M.l, y:M.t, width:W-M.l-M.r, height:H-M.t-M.b,
    fill:'transparent'});
  svg.appendChild(hit);
  hit.addEventListener('mousemove', ev => {
    const r = svg.getBoundingClientRect();
    const sx = (ev.clientX - r.left) / r.width * W;
    const xv = xmin + (sx - M.l)/(W-M.l-M.r)*(xmax-xmin);
    cross.setAttribute('x1', sx); cross.setAttribute('x2', sx);
    cross.setAttribute('opacity', 1);
    let html = `<b>${opts.xlabel||'x'} ${nice(xv)}</b>`;
    for (const s of live){
      let best = s.pts[0];
      for (const pt of s.pts) if (Math.abs(pt[0]-xv) < Math.abs(best[0]-xv)) best = pt;
      html += `<br>${s.name}: ${opts.y0to100 ? best[1].toFixed(0)+'%' : nice(best[1])}`;
    }
    TIP.innerHTML = html;
    TIP.style.display = 'block';
    TIP.style.left = Math.min(ev.clientX+14, window.innerWidth-170) + 'px';
    TIP.style.top = (ev.clientY+14) + 'px';
  });
  hit.addEventListener('mouseleave', () => {
    TIP.style.display='none'; cross.setAttribute('opacity',0);
  });
  host.appendChild(svg);
}

function tile(lab, val, sub){
  return `<div class="tile"><div class="lab">${lab}</div>
    <div class="val">${val}${sub?` <small>${sub}</small>`:''}</div></div>`;
}

function table(mount, d){
  const host = document.getElementById(mount);
  if (!d.rate.length){ host.textContent=''; return; }
  const n = d.rate.length, pick = [];
  const k = Math.max(1, Math.floor(n/12));
  for (let i=0;i<n;i+=k) pick.push(i);
  if (pick[pick.length-1] !== n-1) pick.push(n-1);
  let h = '<table><thead><tr><th>에피소드</th><th>성공률</th><th>개입률</th></tr></thead><tbody>';
  for (const i of pick){
    const iv = d.interv[i];
    h += `<tr><td>${Math.round(d.rate[i][0]).toLocaleString()}</td>
      <td>${d.rate[i][1].toFixed(0)}%</td>
      <td>${iv ? iv[1].toFixed(0)+'%' : '-'}</td></tr>`;
  }
  host.innerHTML = h + '</tbody></table>';
}

async function refresh(){
  let d;
  try { d = await (await fetch('data.json', {cache:'no-store'})).json(); }
  catch (e) { document.getElementById('foot').textContent = '데이터를 못 읽었다: ' + e; return; }

  document.getElementById('runline').textContent = d.run;
  document.getElementById('w').textContent = d.window;

  const bn = document.getElementById('banner');
  if (d.demo){
    bn.className = 'banner';
    bn.innerHTML = '<b>미리보기 모드</b> — 가짜 데이터다. 실제 학습을 보려면 <code>--demo</code> 없이 다시 띄운다.';
  } else if (!d.exists){
    bn.className = 'banner';
    bn.innerHTML = '<b>아직 학습을 시작하지 않았다.</b> learner 를 띄우면 지표 파일이 생기고 여기에 그래프가 그려진다.'
      + ' 화면 모양을 먼저 보고 싶으면 <code>python scripts/progress_dashboard.py --demo</code> 로 띄운다.';
  } else if (!d.episodes){
    bn.className = 'banner';
    bn.innerHTML = '<b>첫 에피소드를 기다리는 중.</b> actor 는 에피소드가 끝날 때만 전이를 보낸다 — 20~40초쯤 걸린다.';
  } else {
    bn.className = 'banner ok';
  }

  const pct = v => v==null ? '–' : v.toFixed(0)+'%';
  document.getElementById('tiles').innerHTML =
      tile('최근 20 성공률', pct(d.recent_rate))
    + tile('최근 20 개입률', pct(d.recent_interv))
    + tile('에피소드', (d.episodes||0).toLocaleString(), `성공 ${d.successes||0}`)
    + tile('최적화 스텝', d.opt_step==null?'–':Number(d.opt_step).toLocaleString())
    + tile('온라인 버퍼', d.online_size==null?'–':Number(d.online_size).toLocaleString())
    + tile('시연 버퍼', d.offline_size==null?'–':Number(d.offline_size).toLocaleString());

  const empty = d.exists
    ? '첫 에피소드가 끝나면 그려진다. actor 는 에피소드가 끝날 때만 전이를 보낸다.'
    : 'learner 가 아직 지표 파일을 만들지 않았다.';

  chart('c1', [
    {pts:d.rate,   color:'var(--s1)', name:'성공률'},
    {pts:d.interv, color:'var(--s2)', name:'개입률'},
  ], {y0to100:true, xlabel:'에피소드 번호', empty, label:'성공률과 개입률 추이'});

  chart('c2', [
    {pts:d.online,  color:'var(--s1)', name:'온라인'},
    {pts:d.offline, color:'var(--s2)', name:'시연'},
  ], {xlabel:'최적화 스텝', empty, label:'버퍼 크기 추이'});

  chart('c3', [
    {pts:d.loss, color:'var(--s1)', name:'critic loss'},
  ], {xlabel:'최적화 스텝', empty, label:'critic loss 추이'});

  table('t1', d);
  document.getElementById('foot').textContent =
    new Date().toLocaleTimeString('ko-KR') + ' · 10초마다 갱신';
}
let CTL = false, CTLKEYS = {};
async function setupControl(){
  let info;
  try { info = await (await fetch('control-info', {cache:'no-store'})).json(); }
  catch(e){ return; }
  const box = document.getElementById('ctl');
  if (!info.on){ box.hidden = true; return; }
  box.hidden = false;
  const legend = document.getElementById('keys');
  legend.innerHTML = info.keys.map(k =>
    `<div data-x="${k.x}"><b>${k.key}</b> ${k.label}</div>`).join('');
  info.keys.forEach(k => { CTLKEYS[k.x] = true; });

  const btn = document.getElementById('ctlbtn');
  const state = document.getElementById('ctlstate');
  const img = document.getElementById('view');
  function setCtl(on){
    CTL = on;
    btn.classList.toggle('on', CTL);
    btn.textContent = CTL ? '조작 켜짐 (끄려면 클릭)' : '이 창에서 조작하기';
    state.textContent = CTL
      ? '이 창의 키 입력이 로봇으로 간다. 먼저 Space 로 조종권을 가져올 것.'
      : '꺼짐 — 화면을 클릭하거나 이 버튼을 누르면 이 창의 키 입력이 로봇으로 간다';
    if (img) img.style.outline = CTL ? '3px solid #2ea043' : 'none';
  }
  btn.addEventListener('click', () => setCtl(!CTL));
  // 시뮬레이터 화면을 클릭하면 바로 조작이 켜진다. 여기서부터 방향키는 페이지를 스크롤하지 않는다.
  if (img) img.addEventListener('mousedown', ev => { ev.preventDefault(); setCtl(true); });
  setCtl(false);

  const down = new Set();
  async function send(code, action){
    try { await fetch('key', {method:'POST', headers:{'Content-Type':'application/json'},
                              body: JSON.stringify({code, action})}); } catch(e){}
  }
  function mark(code, on){
    const map = {ArrowUp:'Up',ArrowDown:'Down',ArrowLeft:'Left',ArrowRight:'Right',
      Space:'space',Enter:'Return',Escape:'Escape',ShiftLeft:'Shift_L',ShiftRight:'Shift_R',
      ControlLeft:'Control_L',ControlRight:'Control_R'};
    const x = map[code] || (code.startsWith('Key') ? code.slice(3).toLowerCase() : null);
    const el = legend.querySelector(`[data-x="${x}"]`);
    if (el) el.classList.toggle('hit', on);
  }
  // 방향키·스페이스는 브라우저 기본 동작이 스크롤이다. 조작이 켜져 있으면 무조건 막는다.
  const SCROLLERS = new Set(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Space','PageUp','PageDown','Home','End']);
  document.addEventListener('keydown', ev => {
    if (ev.target && /^(INPUT|TEXTAREA)$/.test(ev.target.tagName)) return;
    if (!CTL) return;
    if (SCROLLERS.has(ev.code)) ev.preventDefault();
    if (ev.repeat) return;
    ev.preventDefault(); down.add(ev.code); mark(ev.code, true); send(ev.code, 'down');
  }, {passive: false});
  document.addEventListener('keyup', ev => {
    if (!CTL) return;
    ev.preventDefault(); down.delete(ev.code); mark(ev.code, false); send(ev.code, 'up');
  }, {passive: false});
  window.addEventListener('blur', () => {
    down.forEach(c => { mark(c, false); send(c, 'up'); });
    down.clear();
  });
}

async function checkView(){
  try{
    const v = await (await fetch('view-status', {cache:'no-store'})).json();
    const card = document.getElementById('viewcard');
    const img = document.getElementById('view');
    if (v.exists && card.hidden){
      card.hidden = false;
      img.src = 'view.mjpg?t=' + Date.now();
    } else if (!v.exists && !card.hidden){
      card.hidden = true;
      img.removeAttribute('src');
    }
  }catch(e){/* 무시 */}
}
refresh(); checkView(); setupControl();
setInterval(refresh, 10000);
setInterval(checkView, 5000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream_view(self):
        """조작자용 화면을 MJPEG 로 흘려준다. 파일이 바뀔 때만 새 프레임을 보낸다."""
        import time

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = None
        idle = 0.0
        try:
            while True:
                try:
                    mtime = os.path.getmtime(VIEW)
                except OSError:
                    time.sleep(0.5)
                    idle += 0.5
                    if idle > 120:
                        return
                    continue
                if mtime != last:
                    last = mtime
                    idle = 0.0
                    try:
                        with open(VIEW, "rb") as f:
                            jpg = f.read()
                    except OSError:
                        continue
                    if jpg[:2] == b"\xff\xd8":  # 온전한 JPEG 인지만 확인
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self):
        if self.path.split("?")[0] != "/key" or not CONTROL_ON:
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:  # noqa: BLE001
            self.send_error(400)
            return
        xkey = BROWSER_TO_X.get(str(body.get("code", "")))
        action = "keydown" if body.get("action") == "down" else "keyup"
        if xkey is None or xkey not in ALLOWED_KEYS:
            self._send(json.dumps({"ok": False, "reason": "허용되지 않은 키"}).encode(),
                       "application/json")
            return
        env = dict(os.environ, DISPLAY=args.display)
        try:
            subprocess.run(["xdotool", action, xkey], env=env, timeout=2,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            ok = True
        except Exception:  # noqa: BLE001
            ok = False
        self._send(json.dumps({"ok": ok, "key": xkey, "action": action}).encode(),
                   "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/control-info":
            self._send(json.dumps({"on": CONTROL_ON, "keys": KEY_LEGEND,
                                   "display": args.display}).encode(), "application/json")
            return
        if path == "/view.mjpg":
            self._stream_view()
            return
        if path == "/view-status":
            self._send(json.dumps({"exists": os.path.exists(VIEW), "path": str(VIEW)}).encode(),
                       "application/json")
            return
        if path in ("/", "/index.html"):
            title = f"학습 진행 · {RUN.name}"
            body = PAGE.replace("__TITLE__", title).replace(
                "__RUN__", "(미리보기)" if args.demo else str(RUN)
            )
            self._send(body.encode(), "text/html; charset=utf-8")
        elif path == "/data.json":
            self._send(json.dumps(collect()).encode(), "application/json; charset=utf-8")
        else:
            self.send_error(404)

    def log_message(self, *a):  # 요청 로그로 터미널을 채우지 않는다
        pass


if __name__ == "__main__":
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("=" * 62)
    print(f"  학습 진행 대시보드가 떴다.  포트 {args.port}")
    print("=" * 62)
    print("  1) VS Code 의 PORTS 탭에서 다음 포트를 포워딩한다:")
    print(f"       {args.port}")
    print("  2) 랩탑 브라우저에서 연다:")
    print(f"       http://localhost:{args.port}/")
    print()
    if args.demo:
        print("  [미리보기 모드] 가짜 데이터를 보여준다.")
    else:
        print(f"  읽는 파일: {METRICS}")
        if not METRICS.exists():
            print("   -> 아직 없다. 학습을 시작하면 생긴다. 지금은 빈 화면이 정상이다.")
            print("      화면 모양을 먼저 보려면: --demo 를 붙여 다시 띄운다.")
    print()
    print("  이 터미널은 켜 둔 채로 두면 된다. Ctrl+C 로 종료.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()
