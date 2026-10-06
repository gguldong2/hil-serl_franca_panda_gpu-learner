#!/usr/bin/env python3
"""돌아가는 시뮬레이션을 별도 창으로 보는 전용 뷰어.

대시보드(8088)와 독립된 포트에서 돈다. 학습에는 전혀 손대지 않는다 —
actor 가 /dev/shm/hil_view.jpg 로 내보내는 그림을 읽기만 한다.

actor 는 HIL_VIEW_CAMERAS 에 적힌 카메라를 가로로 이어붙여 한 장으로 쓴다.
예: front,handcam_rgb -> 1024x512 (512 짜리 두 장). 뷰어는 그걸 다시 갈라서
나란히 / 위아래 / 한 대만 크게 중에 골라 보여준다. 자르기는 브라우저에서 하므로
서버 쪽 추가 비용이 없다.

    python3 scripts/view.py                 # 8089 에서 뜬다
    python3 scripts/view.py --port 9000
    ffplay http://127.0.0.1:8089/view.mjpg  # VNC 안에서 네이티브 창으로
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

p = argparse.ArgumentParser(description="시뮬레이션 화면 전용 뷰어")
p.add_argument("--port", type=int, default=8089)
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--view", default=os.environ.get("HIL_VIEW_STREAM", "/dev/shm/hil_view.jpg"),
               help="actor 가 쓰는 그림 파일")
p.add_argument("--labels", default=os.environ.get("HIL_VIEW_CAMERAS", ""),
               help="쉼표로 구분한 카메라 이름. 비우면 자동으로 잡는다.")
p.add_argument("--hud-height", type=int, default=56,
               help="HUD 띠 높이(px). 한 대만 볼 때 위에 겹쳐 보여준다.")
args = p.parse_args()

VIEW = args.view
# 스트리머가 실제로 흘려보낸 프레임 시각. fps 를 정직하게 재려고 둔다.
_FRAME_TIMES: deque[float] = deque(maxlen=120)

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --bg:#0f1114; --panel:#181b1f; --line:#2b3036; --ink:#e8eaec; --ink-2:#9aa2aa;
  --accent:#5aa8c4; --good:#5fb98c; --warn:#d7a24a;
  color-scheme:dark;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Malgun Gothic",sans-serif;
  display:flex;flex-direction:column}
header{display:flex;flex-wrap:wrap;gap:8px;align-items:center;
  padding:9px 14px;background:var(--panel);border-bottom:1px solid var(--line)}
.grp{display:flex;gap:4px;background:var(--bg);border:1px solid var(--line);
  border-radius:4px;padding:3px}
button{font:inherit;font-size:13px;color:var(--ink-2);background:none;border:0;
  padding:5px 11px;border-radius:3px;cursor:pointer;white-space:nowrap}
button:hover{color:var(--ink);background:#22262b}
button.on{background:var(--accent);color:#0f1114;font-weight:600}
.spacer{flex:1}
.stat{font-family:var(--mono);font-size:12px;color:var(--ink-2);
  font-variant-numeric:tabular-nums;white-space:nowrap}
.stat b{color:var(--ink);font-weight:500}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;
  background:var(--good);margin-right:5px;vertical-align:1px}
.dot.stale{background:var(--warn)}
.dot.dead{background:#c0504a}
main{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;
  padding:12px;overflow:auto}
canvas{max-width:100%;max-height:100%;border:1px solid var(--line);border-radius:3px;
  background:#000;display:block}
#msg{font-family:var(--mono);font-size:13px;color:var(--ink-2);text-align:center;
  line-height:1.9}
#msg code{background:#22262b;padding:.15em .4em;border-radius:2px}
kbd{font-family:var(--mono);font-size:11px;border:1px solid var(--line);
  border-radius:3px;padding:1px 5px;color:var(--ink-2)}
@media(max-width:560px){header{padding:8px 10px}.stat{font-size:11px}}
</style>

<header>
  <div class="grp" id="modes">
    <button data-m="side"  class="on">나란히</button>
    <button data-m="stack">위아래</button>
    <button data-m="c0">__L0__</button>
    <button data-m="c1">__L1__</button>
  </div>
  <div class="grp">
    <button id="hud" class="on" title="한 대만 볼 때 위쪽 정보 띠를 겹쳐 보여준다">HUD</button>
    <button id="fs" title="전체화면">⛶</button>
  </div>
  <div class="spacer"></div>
  <div class="stat"><span class="dot" id="dot"></span><b id="fps">–</b> fps
    · <b id="res">–</b> · 마지막 <b id="age">–</b></div>
</header>

<main><canvas id="cv"></canvas><div id="msg" hidden></div></main>

<script>
const CAM_N = __CAM_N__, HUD_H = __HUD_H__;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const msg = document.getElementById('msg');
let mode = 'side';

// MJPEG 연결은 하나만 쓴다. 화면 배치는 캔버스에서 잘라 그린다.
const img = new Image();
img.src = 'view.mjpg?' + Date.now();

function draw(){
  const W = img.naturalWidth, H = img.naturalHeight;
  if (!W || !H) return;
  const cw = Math.round(W / CAM_N);          // 카메라 한 대 폭
  const gap = 6;
  const hudOn = document.getElementById('hud').classList.contains('on');

  if (mode === 'side'){
    cv.width = W; cv.height = H;
    ctx.drawImage(img, 0, 0);
  } else if (mode === 'stack'){
    cv.width = cw; cv.height = H * CAM_N + gap * (CAM_N - 1);
    ctx.fillStyle = '#0f1114'; ctx.fillRect(0, 0, cv.width, cv.height);
    for (let i = 0; i < CAM_N; i++)
      ctx.drawImage(img, i * cw, 0, cw, H, 0, i * (H + gap), cw, H);
  } else {
    const i = +mode.slice(1);
    cv.width = cw; cv.height = H;
    ctx.drawImage(img, i * cw, 0, cw, H, 0, 0, cw, H);
    // HUD 글자는 합쳐진 그림의 맨 왼쪽 위에만 찍힌다. 오른쪽 카메라를 볼 때는
    // 그 띠만 따로 떠서 위에 겹쳐야 회차·시간이 보인다.
    if (hudOn && i > 0)
      ctx.drawImage(img, 0, 0, cw, HUD_H, 0, 0, cw, HUD_H);
  }
}
setInterval(draw, 50);

document.querySelectorAll('#modes button').forEach(b => b.onclick = () => {
  document.querySelectorAll('#modes button').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); mode = b.dataset.m; draw();
});
document.getElementById('hud').onclick = e => { e.target.classList.toggle('on'); draw(); };
document.getElementById('fs').onclick = () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.().catch(()=>{});
};
addEventListener('keydown', e => {
  const k = e.key;
  const b = ({'1':'side','2':'stack','3':'c0','4':'c1'})[k];
  if (b) document.querySelector(`#modes button[data-m="${b}"]`)?.click();
  else if (k === 'f') document.getElementById('fs').click();
  else if (k === 'h') document.getElementById('hud').click();
});

async function poll(){
  try {
    const r = await fetch('meta', {cache:'no-store'});
    const m = await r.json();
    const dot = document.getElementById('dot');
    document.getElementById('fps').textContent = m.fps == null ? '–' : m.fps.toFixed(1);
    document.getElementById('res').textContent =
      img.naturalWidth ? `${img.naturalWidth}×${img.naturalHeight}` : '–';
    if (!m.exists){
      dot.className = 'dot dead';
      document.getElementById('age').textContent = '없음';
      msg.hidden = false; cv.hidden = true;
      msg.innerHTML = `<code>${m.path}</code> 가 없다.<br>` +
        `actor 가 돌고 있는지, <code>HIL_VIEW_STREAM</code> 이 설정됐는지 확인.`;
    } else {
      const a = m.age;
      dot.className = 'dot' + (a > 5 ? ' dead' : a > 1.5 ? ' stale' : '');
      document.getElementById('age').textContent = a < 1 ? '방금' : a.toFixed(1) + '초 전';
      msg.hidden = true; cv.hidden = false;
    }
  } catch(e){ /* 서버가 잠깐 안 받을 수 있다 */ }
}
poll(); setInterval(poll, 1000);
</script>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # 콘솔을 조용히
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self):
        """파일이 바뀔 때만 새 프레임을 내보낸다."""
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
                    if idle > 300:
                        return
                    continue
                if mtime != last:
                    last, idle = mtime, 0.0
                    try:
                        with open(VIEW, "rb") as f:
                            jpg = f.read()
                    except OSError:
                        continue
                    if jpg[:2] != b"\xff\xd8":   # 반쯤 쓰인 파일은 건너뛴다
                        continue
                    _FRAME_TIMES.append(time.time())
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _meta(self):
        exists = os.path.exists(VIEW)
        age = None
        if exists:
            try:
                age = max(0.0, time.time() - os.path.getmtime(VIEW))
            except OSError:
                exists = False
        fps = None
        if len(_FRAME_TIMES) >= 2:
            span = _FRAME_TIMES[-1] - _FRAME_TIMES[0]
            # 스트림이 멈춘 뒤 옛날 값으로 fps 를 부풀리지 않도록
            if span > 0 and time.time() - _FRAME_TIMES[-1] < 3.0:
                fps = (len(_FRAME_TIMES) - 1) / span
        self._send(json.dumps({"exists": exists, "age": age, "fps": fps,
                               "path": VIEW}).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0].lstrip("/")
        if path == "view.mjpg":
            self._stream()
        elif path == "meta":
            self._meta()
        elif path in ("", "index.html"):
            self._send(page().encode(), "text/html; charset=utf-8")
        else:
            self.send_error(404)


def labels_from_running_actor():
    """돌고 있는 actor 의 HIL_VIEW_CAMERAS 를 그대로 가져온다.

    뷰어를 따로 띄우면 actor 의 환경변수가 없으므로, /proc 에서 직접 읽어
    카메라 이름을 맞춘다. 못 찾으면 조용히 포기한다.
    """
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                cmd = f.read().decode("utf-8", "replace")
            if "lerobot.rl.actor" not in cmd:
                continue
            with open(f"/proc/{entry}/environ", "rb") as f:
                env = f.read().decode("utf-8", "replace")
        except OSError:
            continue
        for kv in env.split("\0"):
            if kv.startswith("HIL_VIEW_CAMERAS="):
                names = [x.strip() for x in kv.split("=", 1)[1].split(",") if x.strip()]
                if names:
                    return names
    return []


def detect_layout():
    """그림 크기로 카메라 몇 대가 이어붙어 있는지 짐작한다."""
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if not labels:
        labels = labels_from_running_actor()
        if labels:
            print(f"  돌고 있는 actor 에서 카메라 이름을 읽었다: {', '.join(labels)}")
    n = len(labels)
    try:
        from PIL import Image
        with Image.open(VIEW) as im:
            w, h = im.size
        guess = max(1, round(w / h)) if h else 1
        if n == 0:
            n = guess
        elif n != guess:
            print(f"  주의: --labels 는 {n}대인데 그림은 {w}x{h} ({guess}대로 보인다). "
                  f"{guess}대로 간다.")
            n = guess
            labels = []
    except Exception:  # noqa: BLE001
        if n == 0:
            n = 2
    while len(labels) < n:
        labels.append(f"카메라 {len(labels) + 1}")
    return n, labels[:n]


CAM_N, LABELS = detect_layout()


def page() -> str:
    s = PAGE.replace("__TITLE__", "시뮬레이션 화면")
    s = s.replace("__CAM_N__", str(CAM_N)).replace("__HUD_H__", str(args.hud_height))
    for i in range(4):
        s = s.replace(f"__L{i}__", LABELS[i] if i < len(LABELS) else f"카메라 {i + 1}")
    return s


if __name__ == "__main__":
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"  시뮬레이션 뷰어가 떴다.  포트 {args.port}")
    print(f"     그림  : {VIEW}")
    print(f"     카메라: {CAM_N}대 — {', '.join(LABELS)}")
    print(f"     브라우저: http://127.0.0.1:{args.port}/")
    print(f"     네이티브: ffplay http://127.0.0.1:{args.port}/view.mjpg")
    print("     단축키: 1 나란히 · 2 위아래 · 3/4 한 대만 · h HUD · f 전체화면")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  뷰어를 닫는다.")
