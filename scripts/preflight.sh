#!/usr/bin/env bash
# 학습을 띄우기 전 점검. 컨테이너가 다시 뜬 뒤, 다른 서버로 옮긴 뒤, 오랜만에 돌릴 때 먼저 친다.
#
#   bash scripts/preflight.sh          # 점검만 (아무것도 바꾸지 않는다)
#   bash scripts/preflight.sh --fix    # 고칠 수 있는 것(시스템 패키지, EGL 라이브러리, X :1)은 고친 뒤 다시 점검
#   bash scripts/preflight.sh --quick  # 빠른 점검 (EGL 실렌더 생략, 0.1초). 다른 스크립트가 자동으로 부른다
#
# train.sh actor / record.sh / teleop.sh 는 시작할 때 --quick 을 자동으로 돌린다.
# 통과하면 아무것도 찍지 않고, 막히면 이유를 보여주고 멈춘다.
# 건너뛰려면 앞에 HIL_SKIP_PREFLIGHT=1 (예: xvfb-run 으로 VNC 없이 돌릴 때).
#
# 보는 것
#   0. 시스템 패키지 · venv                        (RunPod 처럼 pod 를 멈췄다 켜면 apt 로 깐 것과 Python 3.12 가 지워진다)
#   1. GPU 드라이버와 nvidia-gl/ 의 버전이 같은가   (드라이버가 바뀌면 EGL 이 죽는다)
#   2. EGL 로 실제로 GPU 렌더링이 되는가            (actor 카메라. 안 되면 actor 가 죽거나 15배 느려진다)
#   3. X 디스플레이 :1 (VNC 또는 Xvfb) 이 떠 있는가 (키보드 개입이 여기 붙는다)
#   4. 디스크 여유                                  (체크포인트 저장 중 꽉 차면 버퍼 스냅샷이 깨진다)
#   5. 이미 떠 있는 learner/actor/대시보드          (중복 실행 방지)
# 마지막 줄에 "준비됨" 또는 "막힘: ..." 을 찍는다. 막힘이면 종료코드 1.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
FIX=0; QUICK=0
case "${1:-}" in
  --fix)   FIX=1 ;;
  --quick) QUICK=1 ;;
  "")      ;;
  *)       echo "사용법: bash scripts/preflight.sh [--fix|--quick]"; exit 2 ;;
esac

ok()   { printf '  \033[32m[OK]\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33m[주의]\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m[막힘]\033[0m %s\n' "$*"; BLOCK+=("$1"); }
BLOCK=()
IS_WSL=0; grep -qi microsoft /proc/version 2>/dev/null && IS_WSL=1
x1_up() { pgrep -x Xtigervnc >/dev/null || [ -S /tmp/.X11-unix/X1 ]; }

check() {
BLOCK=()
echo "== 0. 시스템 패키지 · venv ================================================="
# 컨테이너가 새로 뜨면(RunPod 의 pod 멈춤 → 다시 켜기) /workspace 밖은 이미지 상태로 돌아간다.
# venv(.venv)는 남지만 그 안의 python 은 시스템 python3.12 를 가리키는 링크라 같이 깨진다.
NEED_SYS_FIX=0
if [ ! -e .venv/bin/python ] && [ ! -L .venv/bin/python ]; then
  bad "venv 없음" "프로젝트 설치가 안 됐다 → bash scripts/server22/02_setup_project.sh"
elif ! .venv/bin/python -c '' 2>/dev/null; then
  bad "venv Python 안 돎" "시스템 Python 3.12 가 없다 — 컨테이너(pod)가 새로 떠 지워졌다 → bash scripts/server22/01_install_system.sh"
  NEED_SYS_FIX=1
else
  ok "venv Python $(.venv/bin/python -c 'import sys;print(sys.version.split()[0])')"
fi
MISS=()
command -v gcc     >/dev/null || MISS+=(gcc)       # torch.compile(Triton) 이 첫 실행 때 쓴다
command -v xdotool >/dev/null || MISS+=(xdotool)   # 대시보드의 브라우저 조작
command -v Xvfb >/dev/null || command -v Xtigervnc >/dev/null || MISS+=(Xvfb)
if [ ${#MISS[@]} -eq 0 ]; then
  ok "gcc · xdotool · X 서버 있음"
elif [ "$IS_WSL" = 1 ]; then
  warn "없음: ${MISS[*]} (WSL 이라 막지 않는다)"
else
  bad "시스템 패키지 없음" "${MISS[*]} — 컨테이너(pod)가 새로 떴으면 apt 로 깐 것이 지워진다 → bash scripts/server22/01_install_system.sh"
  NEED_SYS_FIX=1
fi
command -v tmux >/dev/null || warn "tmux 없음 — SSH 가 끊기면 그 셸에서 띄운 learner/actor 가 같이 죽는다 → 01_install_system.sh"

echo "== 1. GPU 드라이버 / nvidia-gl ============================================"
NEED_GL_FIX=0
if ! command -v nvidia-smi >/dev/null; then
  warn "nvidia-smi 없음 — GPU 없는 환경. EGL 은 CPU(llvmpipe)로 돈다(느림)."
  DRV=""
else
  DRV="$(awk '{print $8}' /proc/driver/nvidia/version 2>/dev/null)"
  [ -n "$DRV" ] || DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
  NATIVE=$(ldconfig -p 2>/dev/null | grep -c libEGL_nvidia || true)
  GLVER="$(ls nvidia-gl/libEGL_nvidia.so.*.*.* 2>/dev/null | sed 's/.*\.so\.//' | head -1)"
  echo "  커널 드라이버 $DRV · 시스템 libEGL_nvidia ${NATIVE}개 · nvidia-gl/ ${GLVER:-없음}"
  if [ "$NATIVE" -gt 0 ]; then
    if [ -n "$GLVER" ]; then
      warn "이 서버는 그래픽 라이브러리를 이미 갖고 있다. nvidia-gl/ 이 오히려 덮어쓴다 → 지우는 게 맞다: rm -rf nvidia-gl"
    else
      ok "시스템 그래픽 라이브러리를 쓴다 (nvidia-gl/ 불필요)"
    fi
  elif [ -z "$GLVER" ]; then
    bad "nvidia-gl/ 없음" "그래픽 라이브러리 없는 컨테이너인데 nvidia-gl/ 이 없다 → bash scripts/server22/05_nvidia_gl.sh"
    NEED_GL_FIX=1
  elif [ "$GLVER" != "$DRV" ]; then
    bad "드라이버 버전 불일치" "nvidia-gl/ 은 $GLVER, 드라이버는 $DRV → bash scripts/server22/05_nvidia_gl.sh"
    NEED_GL_FIX=1
  else
    ok "버전 일치 ($DRV)"
  fi
  ICD="nvidia-gl/nvidia_egl_icd.json"
  if [ -f "$ICD" ] && ! grep -q "\"$ROOT/nvidia-gl/" "$ICD"; then
    bad "ICD 경로 어긋남" "$ICD 가 다른 폴더를 가리킨다(프로젝트를 옮겼다) → bash scripts/server22/05_nvidia_gl.sh --force"
    NEED_GL_FIX=1
  fi
fi

if [ "$QUICK" = 1 ]; then
echo "== 2. EGL 실제 렌더링 ================================= (--quick: 생략) =="
echo "  1번의 버전 대조가 통과하면 거의 항상 같이 통과한다. 확실히 보려면 --quick 없이."
else
echo "== 2. EGL 실제 렌더링 ======================================================"
R="$(bash -c 'source scripts/env.sh >/dev/null 2>&1; MUJOCO_GL=egl timeout 90 python - 2>&1' <<'PY'
import os, time, mujoco, gym_hil
from OpenGL import GL
m = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(gym_hil.__file__), "assets", "scene.xml"))
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 128, 128)
r.update_scene(d); r.render()
t = time.time()
for _ in range(20): r.update_scene(d); r.render()
print("RESULT", GL.glGetString(GL.GL_RENDERER).decode(), f"{(time.time()-t)/20*1000:.1f}ms")
PY
)"
LINE="$(printf '%s\n' "$R" | grep '^RESULT' | head -1)"
if [ -z "$LINE" ]; then
  bad "EGL 실패" "렌더링 자체가 안 된다: $(printf '%s\n' "$R" | grep -E 'Error|error' | tail -1 | cut -c1-120)"
elif printf '%s' "$LINE" | grep -qi "llvmpipe\|softpipe"; then
  bad "EGL 이 CPU 로 폴백" "${LINE#RESULT } — GPU 가 안 잡혔다(1번 확인)"
else
  ok "${LINE#RESULT }"
fi
fi

echo "== 3. X 디스플레이 :1 (VNC 또는 Xvfb) ======================================"
if x1_up; then
  if pgrep -x Xtigervnc >/dev/null; then
    ok "X :1 떠 있음 (VNC)"
    if ss -ltn 2>/dev/null | grep -q ':6080 '; then ok "noVNC 6080 열림 (브라우저로 볼 때만 필요)"
    else warn "noVNC(6080) 없음 — 키보드 개입엔 지장 없고, VNC 화면을 브라우저로 볼 때만 필요"; fi
  else
    ok "X :1 떠 있음 (Xvfb 등 — 화면은 대시보드로 본다)"
  fi
elif command -v vncserver >/dev/null; then
  bad "X :1 꺼짐" "키보드 개입이 안 된다 → bash scripts/server22/04_start_vnc.sh  (또는 --fix)"
else
  bad "X :1 꺼짐" "키보드 개입이 안 된다 → bash scripts/preflight.sh --fix  (Xvfb :1 을 띄운다)"
fi

echo "== 4. 디스크 ================================================================"
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
USE=$(df --output=pcent . | tail -1 | tr -dc 0-9)
if   [ "$FREE_GB" -lt 10 ]; then bad "디스크 부족" "${FREE_GB}GB 남음(${USE}%) — 저장 중 꽉 차면 버퍼 스냅샷이 깨진다. 체크포인트부터 정리"
elif [ "$FREE_GB" -lt 50 ]; then warn "${FREE_GB}GB 남음(${USE}%). 체크포인트가 2,000 스텝마다 81MB 씩 쌓인다 — 정리를 권함"
else ok "${FREE_GB}GB 남음 (${USE}%)"; fi

echo "== 5. 이미 떠 있는 것 ======================================================"
for pat in lerobot.rl.learner lerobot.rl.actor progress_dashboard; do
  p=$(pgrep -f "python.*$pat" | head -1)
  if [ -n "$p" ]; then warn "$pat 이미 실행 중 (PID $p) — 새로 띄우면 중복"; else ok "$pat 없음"; fi
done
}

check
if [ "$FIX" = 1 ] && [ ${#BLOCK[@]} -gt 0 ]; then
  echo; echo "== 고치기 =================================================================="
  if [ "${NEED_SYS_FIX:-0}" = 1 ]; then bash scripts/server22/01_install_system.sh || true; fi
  if [ "${NEED_GL_FIX:-0}" = 1 ]; then bash scripts/server22/05_nvidia_gl.sh --force || true; fi
  if ! x1_up; then
    if command -v vncserver >/dev/null; then
      bash scripts/server22/04_start_vnc.sh || true
    elif command -v Xvfb >/dev/null; then
      # setsid + nohup: 이 셸이나 SSH 가 끊겨도 X 서버가 남는다 (그냥 & 로 띄우면 같이 죽는다)
      # Xvfb 는 GLX 를 켜면서 EGL 을 연다. RunPod 컨테이너(드라이버 595)에는 NVIDIA 의 GBM 백엔드
      # (libnvidia-egl-gbm, 15_nvidia_gbm.json)가 들어와 있는데 /dev/dri 가 없어서, EGL 이 NVIDIA 를
      # 고르면 Xvfb 가 시작하다 SIGSEGV 로 죽는다(2026-11-19). 그래서 Xvfb 프로세스만 Mesa EGL 을 쓴다.
      # MuJoCo 의 EGL(GPU) 렌더는 다른 프로세스라 영향이 없다 — 이렇게 띄운 뒤 2번 점검이 그대로 GPU 였다.
      MESA_EGL=/usr/share/glvnd/egl_vendor.d/50_mesa.json
      ( [ -f "$MESA_EGL" ] && export __EGL_VENDOR_LIBRARY_FILENAMES="$MESA_EGL"
        setsid nohup Xvfb :1 -screen 0 1280x800x24 -nolisten tcp >/tmp/xvfb_1.log 2>&1 < /dev/null & )
      for _ in 1 2 3 4 5 6; do x1_up && break; sleep 0.5; done
      if x1_up; then
        echo "  Xvfb :1 을 띄웠다 (로그 /tmp/xvfb_1.log). 화면은 대시보드(8088)로 본다."
      else
        echo "  !! Xvfb :1 이 바로 죽었다. /tmp/xvfb_1.log 끝부분:"
        tail -8 /tmp/xvfb_1.log 2>/dev/null | sed 's/^/     /'
      fi
    else
      echo "  X 서버가 없다 → bash scripts/server22/01_install_system.sh"
    fi
  fi
  echo; echo "== 다시 점검 ==============================================================="
  check
fi

echo
if [ ${#BLOCK[@]} -eq 0 ]; then
  echo "준비됨 — 학습을 띄워도 된다."
else
  echo "막힘: ${BLOCK[*]}"
  [ "$FIX" = 0 ] && echo "  고칠 수 있는 것은:  bash scripts/preflight.sh --fix"
  exit 1
fi
