#!/usr/bin/env python3
"""한 실험의 학습 기록을 HTML 리포트 한 장으로 남긴다.

learner_metrics.jsonl 을 읽어 성공률·개입률·버퍼·손실 그래프와 요약을 그린다.
외부 라이브러리 없이 SVG 로 직접 그리므로 파일 하나만 열면 된다(인터넷 불필요).

  python scripts/report.py                          # 기본 런
  python scripts/report.py --run outputs/train/place_v2
  python scripts/report.py --window 20 --open       # 이동평균 구간

결과: <run>/report.html   (학습 중에도 언제든 다시 돌리면 갱신된다)
"""
from __future__ import annotations

import argparse
import html
import json
import os
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--run", default="outputs/train/panda_place_gpu")
p.add_argument("--window", type=int, default=20, help="이동평균에 쓸 에피소드 수")
p.add_argument("--out", default=None, help="결과 경로 (기본: <run>/report.html)")
args = p.parse_args()

RUN = Path(args.run)
OUT = Path(args.out) if args.out else RUN / "report.html"
W = max(1, args.window)

# 검증된 기본 팔레트의 1~3번 슬롯 (dataviz 스킬 references/palette.md).
# 순서를 고정해서 쓰고, 색을 새로 만들지 않는다.
SERIES = [
    ("성공률", "#2a78d6", "#3987e5"),
    ("개입률", "#eb6834", "#d95926"),
    ("시연+개입 버퍼", "#1baf7a", "#199e70"),
]


def load(run: Path):
    m = run / "logs" / "learner_metrics.jsonl"
    eps, train = [], []
    if not m.exists():
        return eps, train
    with m.open(errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            (eps if r.get("kind") == "interaction" else train).append(r)
    return eps, [t for t in train if t.get("kind") == "train"]


def num(v):
    return v if isinstance(v, (int, float)) else None


def build(eps, train):
    """그래프에 그릴 점들을 만든다."""
    # x 축은 에피소드 번호다. actor 를 재시작하면 "Interaction step" 이 0 부터 다시 세기
    # 때문에(측정: 71,869 -> 798) 그걸 x 축으로 쓰면 그래프가 뒤로 감긴다.
    # 에피소드 번호는 기록이 쌓이는 순서라 항상 단조 증가한다. 스텝은 표와 툴팁에만 쓴다.
    succ_pts, iv_pts, hist, ivh, rows = [], [], [], [], []
    for e in eps:
        rew = num(e.get("Episodic reward"))
        if rew is None:
            continue
        step = num(e.get("Interaction step"))
        ok = 1 if rew > 0.5 else 0
        hist.append(ok)
        n = len(hist)
        w = hist[-W:]
        succ_pts.append((n, 100.0 * sum(w) / len(w)))
        ir = num(e.get("Intervention rate"))
        if ir is not None:
            ivh.append(ir)
            wv = ivh[-W:]
            iv_pts.append((n, 100.0 * sum(wv) / len(wv)))
        rows.append((n, int(step) if step is not None else 0, rew, ok, (ir or 0.0) * 100))

    on_pts, off_pts, loss_pts = [], [], []
    for t in train:
        os_ = num(t.get("Optimization step"))
        if os_ is None:
            continue
        v = num(t.get("replay_buffer_size"))
        if v is not None:
            on_pts.append((os_, v))
        v = num(t.get("offline_replay_buffer_size"))
        if v is not None:
            off_pts.append((os_, v))
        for k in ("loss_critic", "critic_loss"):
            v = num(t.get(k))
            if v is not None:
                loss_pts.append((os_, v))
                break
    return succ_pts, iv_pts, on_pts, off_pts, loss_pts, hist, ivh, rows


def thin(pts, n=400):
    """점이 너무 많으면 고르게 솎는다. 마지막 점은 반드시 남긴다."""
    if len(pts) <= n:
        return pts
    k = len(pts) / n
    out = [pts[int(i * k)] for i in range(n)]
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


def fmt(v):
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:.0f}"
    if v >= 1:
        return f"{v:.1f}"
    return f"{v:.4g}"


def chart(cid, title, sub, series, ylabel, xlabel, pct=False, ymin0=True):
    """SVG 선 그래프 하나. series = [(이름, 라이트색, 다크색, [(x,y)...])]"""
    series = [(n, cl, cd, thin(pts)) for n, cl, cd, pts in series if pts]
    if not series:
        return f'<section class="card"><h2>{html.escape(title)}</h2><p class="empty">아직 데이터가 없다.</p></section>'

    Wd, Hd = 720, 260
    ml, mr, mt, mb = 56, 16, 12, 34
    pw, ph = Wd - ml - mr, Hd - mt - mb

    xs = [x for _, _, _, pts in series for x, _ in pts]
    ys = [y for _, _, _, pts in series for _, y in pts]
    x0, x1 = min(xs), max(xs)
    y0 = 0 if ymin0 else min(ys)
    y1 = 100 if pct else max(ys)
    if y1 <= y0:
        y1 = y0 + 1
    if x1 <= x0:
        x1 = x0 + 1

    def px(x):
        return ml + (x - x0) / (x1 - x0) * pw

    def py(y):
        return mt + ph - (y - y0) / (y1 - y0) * ph

    # 격자 + 눈금 (5칸)
    grid, ticks = [], []
    for i in range(6):
        y = y0 + (y1 - y0) * i / 5
        yy = py(y)
        grid.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml + pw}" y2="{yy:.1f}" class="grid"/>')
        lab = f"{y:.0f}%" if pct else fmt(y)
        ticks.append(f'<text x="{ml - 8}" y="{yy + 4:.1f}" class="tick tick-y">{lab}</text>')
    for i in range(5):
        x = x0 + (x1 - x0) * i / 4
        xx = px(x)
        ticks.append(f'<text x="{xx:.1f}" y="{mt + ph + 20}" class="tick tick-x">{fmt(x)}</text>')

    paths, dots, legend = [], [], []
    for n, cl, cd, pts in series:
        d = " ".join(("M" if i == 0 else "L") + f"{px(x):.1f} {py(y):.1f}" for i, (x, y) in enumerate(pts))
        sid = n.replace(" ", "-")
        paths.append(f'<path d="{d}" class="line" style="stroke:var(--s-{sid})"/>')
        lx, ly = px(pts[-1][0]), py(pts[-1][1])
        # 마지막 값에만 직접 라벨 (모든 점에 숫자를 달지 않는다)
        dots.append(
            f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="4.5" class="end" style="fill:var(--s-{sid})"/>'
            f'<text x="{lx - 8:.1f}" y="{ly - 10:.1f}" class="endlab">'
            f'{(str(round(pts[-1][1])) + "%") if pct else fmt(pts[-1][1])}</text>'
        )
        legend.append(
            f'<span class="lg"><i style="background:var(--s-{sid})"></i>{html.escape(n)}</span>'
        )

    data = json.dumps([{"n": n, "p": pts} for n, _, _, pts in series])
    varcss = "".join(
        f"#{cid}{{--s-{n.replace(' ', '-')}:{cl};}}" for n, cl, _, _ in series
    ) + "".join(
        f'@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) #{cid}'
        f"{{--s-{n.replace(' ', '-')}:{cd};}}}}"
        f':root[data-theme="dark"] #{cid}{{--s-{n.replace(" ", "-")}:{cd};}}'
        for n, _, cd, _ in series
    )
    return f"""<section class="card" id="{cid}">
  <style>{varcss}</style>
  <h2>{html.escape(title)}</h2>
  <p class="sub">{sub}</p>
  <div class="legend">{''.join(legend)}</div>
  <div class="plotwrap">
    <svg viewBox="0 0 {Wd} {Hd}" class="plot" data-series='{html.escape(data)}'
         data-geo='{{"ml":{ml},"mt":{mt},"pw":{pw},"ph":{ph},"x0":{x0},"x1":{x1},"y0":{y0},"y1":{y1},"pct":{str(pct).lower()}}}'>
      {''.join(grid)}
      <line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" class="axis"/>
      {''.join(ticks)}
      {''.join(paths)}
      {''.join(dots)}
      <line class="cross" x1="0" y1="{mt}" x2="0" y2="{mt + ph}" style="opacity:0"/>
      <rect class="hit" x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="transparent"/>
      <text x="{ml + pw / 2}" y="{Hd - 4}" class="axlab">{html.escape(xlabel)}</text>
      <text x="14" y="{mt + ph / 2}" class="axlab" transform="rotate(-90 14 {mt + ph / 2})">{html.escape(ylabel)}</text>
    </svg>
    <div class="tip" hidden></div>
  </div>
</section>"""


def main():
    eps, train = load(RUN)
    succ, iv, on, off, loss, hist, ivh, rows = build(eps, train)

    n_ep = len(hist)
    n_ok = sum(hist)
    rec = hist[-W:]
    rec_rate = 100 * sum(rec) / len(rec) if rec else 0
    reciv = ivh[-W:]
    rec_iv = 100 * sum(reciv) / len(reciv) if reciv else 0
    opt = int(train[-1].get("Optimization step", 0)) if train else 0
    # actor 를 재시작하면 Interaction step 이 0 부터 다시 센다. 되돌아간 지점마다
    # 그 전까지의 최대값을 더해 누적으로 만든다.
    inter, base, prev, restarts = 0, 0, 0, 0
    for e in eps:
        v = num(e.get("Interaction step"))
        if v is None:
            continue
        v = int(v)
        if v < prev:
            base += prev
            restarts += 1
        prev = v
        inter = base + v
    cfg_path = RUN / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    total = None
    try:
        total = json.loads(cfg_path.read_text())["policy"]["online_steps"]
    except Exception:  # noqa: BLE001
        pass

    # 누적 학습 시간. actor 래퍼가 <run>/elapsed.json 에 실제로 돈 시간만 쌓는다.
    el = 0.0
    try:
        el = float(json.loads((RUN / "elapsed.json").read_text()).get("seconds", 0))
    except Exception:  # noqa: BLE001
        pass
    el_txt = f"{int(el)//3600}시간 {int(el)%3600//60}분" if el >= 3600 else f"{int(el)//60}분"

    tiles = [
        ("에피소드", f"{n_ep:,}", "지금까지 끝난 판"),
        ("성공", f"{n_ok:,}", f"전체 성공률 {100 * n_ok // max(n_ep, 1)}%"),
        (f"최근 {len(rec)}판 성공률", f"{rec_rate:.0f}%", "이 값이 올라가야 한다"),
        (f"최근 {len(reciv)}판 개입률", f"{rec_iv:.0f}%", "이 값은 내려가야 한다"),
        ("최적화 스텝", f"{opt:,}", "가중치를 갱신한 횟수"),
        ("상호작용 스텝", f"{inter:,}",
         "로봇이 움직인 스텝 (누적)" + (f" · actor 재시작 {restarts}회" if restarts else "")),
        ("학습 시간", el_txt, "실제로 돌아간 시간 (누적)"),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="tl">{html.escape(t)}</div>'
        f'<div class="tv">{html.escape(v)}</div><div class="ts">{html.escape(s)}</div></div>'
        for t, v, s in tiles
    )

    body_rows = "".join(
        f"<tr><td>{i}</td><td>{s:,}</td><td>{r:+.2f}</td>"
        f'<td class="{"ok" if k else "no"}">{"성공" if k else "실패"}</td><td>{ivr:.0f}%</td></tr>'
        for i, s, r, k, ivr in rows[-200:]
    )

    c1 = chart("c1", "성공률과 개입률", f"최근 {W}판 이동평균. <b>파란선이 올라가면서 주황선이 내려가는 모양</b>이 배우고 있다는 신호다. x축은 에피소드 번호다.",
               [(SERIES[0][0], SERIES[0][1], SERIES[0][2], succ),
                (SERIES[1][0], SERIES[1][1], SERIES[1][2], iv)],
               "비율 (%)", "에피소드 번호", pct=True)
    c2 = chart("c2", "버퍼 크기", "매 학습 배치의 절반씩을 이 둘에서 뽑는다. 온라인은 상한까지 차면 오래된 것부터 밀려난다.",
               [("온라인 버퍼", SERIES[0][1], SERIES[0][2], on),
                (SERIES[2][0], SERIES[2][1], SERIES[2][2], off)],
               "전이 개수", "최적화 스텝")
    c3 = chart("c3", "크리틱 손실", "요동쳐도 정상이다. 계속 커지기만 하면 학습이 발산하는 것.",
               [("크리틱 손실", SERIES[0][1], SERIES[0][2], loss)],
               "손실", "최적화 스텝", ymin0=True)

    doc = TEMPLATE.format(
        run=html.escape(str(RUN)),
        when=time.strftime("%Y-%m-%d %H:%M"),
        tiles=tile_html, c1=c1, c2=c2, c3=c3, rows=body_rows,
        nrows=min(len(rows), 200),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc, encoding="utf-8")
    print(f"리포트를 남겼다: {OUT}")
    print(f"  에피소드 {n_ep}개 · 성공 {n_ok}개 · 최근 {len(rec)}판 {rec_rate:.0f}% · 개입 {rec_iv:.0f}%")
    print(f"  최적화 스텝 {opt:,} · 상호작용 스텝 {inter:,}")
    if not eps:
        print("  (아직 끝난 에피소드가 없어 그래프가 비어 있다)")


TEMPLATE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>학습 리포트</title>
<style>
:root{{color-scheme:light;--bg:#f7f7f5;--surface:#fcfcfb;--line:#e4e3de;
--ink:#0b0b0b;--ink2:#52514e;--ink3:#84837c;--grid:#ebeae5;--ok:#008300;--no:#a3a29a}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{color-scheme:dark;
--bg:#131312;--surface:#1a1a19;--line:#33322f;--ink:#fff;--ink2:#c3c2b7;--ink3:#8d8c84;
--grid:#2a2a28;--ok:#3fbf6a;--no:#6e6d67}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#131312;--surface:#1a1a19;--line:#33322f;
--ink:#fff;--ink2:#c3c2b7;--ink3:#8d8c84;--grid:#2a2a28;--ok:#3fbf6a;--no:#6e6d67}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR",sans-serif}}
.wrap{{max-width:800px;margin:0 auto;padding:28px 16px 60px}}
header{{margin-bottom:22px}}
h1{{font-size:21px;margin:0 0 4px}}
.meta{{color:var(--ink3);font-size:13px}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:10px;
padding:18px;margin-bottom:16px}}
h2{{font-size:16px;margin:0 0 3px}}
.sub{{color:var(--ink2);font-size:13px;margin:0 0 12px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:16px}}
.tile{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px}}
.tl{{font-size:12px;color:var(--ink3)}}
.tv{{font-size:26px;font-weight:650;letter-spacing:-.02em;margin:2px 0}}
.ts{{font-size:12px;color:var(--ink2)}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:6px;font-size:13px;color:var(--ink2)}}
.lg{{display:inline-flex;align-items:center;gap:6px}}
.lg i{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.plotwrap{{position:relative}}
.plot{{width:100%;height:auto;display:block;overflow:visible}}
.grid{{stroke:var(--grid);stroke-width:1}}
.axis{{stroke:var(--line);stroke-width:1}}
.line{{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}
.end{{stroke:var(--surface);stroke-width:2}}
.endlab{{fill:var(--ink2);font-size:11px;text-anchor:end}}
.tick{{fill:var(--ink3);font-size:11px}}
.tick-y{{text-anchor:end}} .tick-x{{text-anchor:middle}}
.axlab{{fill:var(--ink3);font-size:11px;text-anchor:middle}}
.cross{{stroke:var(--ink3);stroke-width:1;stroke-dasharray:3 3;pointer-events:none}}
.tip{{position:absolute;pointer-events:none;background:var(--surface);
border:1px solid var(--line);border-radius:7px;padding:7px 10px;font-size:12px;
box-shadow:0 3px 12px rgba(0,0,0,.14);white-space:nowrap;z-index:5}}
.tip b{{font-weight:600}}
details{{margin-top:6px}} summary{{cursor:pointer;color:var(--ink2);font-size:13px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:10px}}
th,td{{text-align:right;padding:5px 8px;border-bottom:1px solid var(--line)}}
th{{color:var(--ink3);font-weight:500}}
td.ok{{color:var(--ok)}} td.no{{color:var(--no)}}
.empty{{color:var(--ink3);font-size:13px;margin:8px 0 0}}
</style></head><body><div class="wrap">
<header><h1>학습 리포트</h1>
<div class="meta">{run} · {when} 기준</div></header>
<div class="tiles">{tiles}</div>
{c1}
{c2}
{c3}
<section class="card"><h2>숫자로 보기</h2>
<p class="sub">가장 최근 {nrows}판.</p>
<details><summary>표 펼치기</summary>
<table><thead><tr><th>판</th><th>상호작용 스텝</th><th>보상</th><th>결과</th><th>개입률</th></tr></thead>
<tbody>{rows}</tbody></table></details></section>
</div>
<script>
document.querySelectorAll('.plotwrap').forEach(function(wrap){{
  var svg=wrap.querySelector('.plot'); if(!svg) return;
  var S=JSON.parse(svg.dataset.series), G=JSON.parse(svg.dataset.geo);
  var cross=svg.querySelector('.cross'), hit=svg.querySelector('.hit'), tip=wrap.querySelector('.tip');
  function toData(cx){{ return G.x0 + (cx-G.ml)/G.pw*(G.x1-G.x0); }}
  function fmt(v){{ return v>=1000? v.toLocaleString(undefined,{{maximumFractionDigits:0}})
                   : (v>=10? v.toFixed(0) : (v>=1? v.toFixed(1) : v.toPrecision(3))); }}
  hit.addEventListener('pointermove', function(ev){{
    var r=svg.getBoundingClientRect(), vb=svg.viewBox.baseVal;
    var cx=(ev.clientX-r.left)/r.width*vb.width;
    if(cx<G.ml||cx>G.ml+G.pw) return;
    var xv=toData(cx);
    cross.setAttribute('x1',cx); cross.setAttribute('x2',cx); cross.style.opacity=.7;
    var lines=[], xs=null;
    S.forEach(function(s){{
      var best=null,bd=Infinity;
      s.p.forEach(function(pt){{ var d=Math.abs(pt[0]-xv); if(d<bd){{bd=d;best=pt;}} }});
      if(best){{ xs=best[0]; lines.push(s.n+' <b>'+(G.pct? Math.round(best[1])+'%' : fmt(best[1]))+'</b>'); }}
    }});
    tip.innerHTML=(xs!==null? fmt(xs)+(G.pct?'번째 판':' 스텝')+'<br>':'')+lines.join('<br>');
    tip.hidden=false;
    var px=(cx/vb.width)*r.width;
    tip.style.left=Math.min(Math.max(px+12,0), r.width-tip.offsetWidth-4)+'px';
    tip.style.top='8px';
  }});
  hit.addEventListener('pointerleave', function(){{ cross.style.opacity=0; tip.hidden=true; }});
}});
</script></body></html>"""


if __name__ == "__main__":
    main()
