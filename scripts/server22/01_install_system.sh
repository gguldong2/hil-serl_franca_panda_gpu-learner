#!/usr/bin/env bash
# [시스템 패키지 + Python 3.12] Ubuntu 22.04 · 24.04. root 또는 sudo 필요.
# 사용법: bash scripts/server22/01_install_system.sh [--with-vnc] [--python deadsnakes|uv]
#   --with-vnc : TigerVNC + noVNC + fluxbox (사람 개입용 원격 데스크톱)
#   --python   : 3.12 설치 방법. 기본 deadsnakes(PPA). PPA 를 못 쓰면 uv (sudo 불필요)
#   xdotool 은 대시보드(8088)의 브라우저 조작에, tmux 는 learner/actor 를 SSH 가 끊겨도 살려 두는 데 쓴다.
#   build-essential(gcc) 는 torch.compile 이 쓰는 Triton 이 첫 실행 때 C 로 작은 모듈을 빌드하는 데 필요하다
#   (본 학습 설정은 use_torch_compile: true. 없으면 "Failed to find C compiler" 로 learner 가 죽는다).
#   libgl1-mesa-glx 는 24.04 에서 없어진 이름이라 그 내용물인 libgl1 + libglx-mesa0 을 직접 적는다(22.04 에서도 같다).
#   Python 3.12: 이미 있으면(24.04 는 기본 저장소에 있다. RunPod 24.04 이미지는 venv · dev 까지 들어 있다) 건너뛴다.
#     deadsnakes 는 22.04 처럼 기본 저장소에 3.12 가 없을 때만, 그리고 아직 등록되지 않았을 때만 넣는다.
#     RunPod 이미지는 deadsnakes 를 미리 등록해 둔다(signed-by 키링). 거기에 add-apt-repository 가 같은 저장소를
#     키를 박은 Signed-By 로 한 번 더 넣으면 apt 전체가 "Conflicting values set for option Signed-By" 로 멈췄다(2026-11-18).
set -euo pipefail
WITH_VNC=0; PY_METHOD=deadsnakes
while [ $# -gt 0 ]; do
  case "$1" in
    --with-vnc) WITH_VNC=1 ;;
    --python) PY_METHOD="$2"; shift ;;
    *) echo "알 수 없는 옵션: $1"; exit 2 ;;
  esac; shift
done
if [ "$(id -u)" = 0 ]; then SUDO=""; else SUDO="sudo"; fi
export DEBIAN_FRONTEND=noninteractive

APT_SRC=(/etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources)
deadsnakes_files() { grep -lsE '^[^#]*deadsnakes/ppa' "${APT_SRC[@]}" || true; }
# 옛 01 이 이미 등록된 deadsnakes 위에 같은 저장소를 한 번 더 넣어 둔 것(add-apt-repository 가 만든
# deadsnakes-ubuntu-ppa-<코드명>.sources|.list)이 있으면 끈다. 끄지 않으면 apt-get update 부터 실패한다.
# .disabled 로 이름만 바꾼다 — apt 가 조용히 무시하고, 필요하면 되돌릴 수 있다.
dedup_deadsnakes() {
  local f
  for f in /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa-*.sources /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa-*.list; do
    [ -e "$f" ] || continue
    if deadsnakes_files | grep -qvxF "$f"; then
      $SUDO mv "$f" "$f.disabled"
      echo "[01] deadsnakes 가 다른 파일에 이미 등록돼 있어 중복 등록을 껐다: $f -> $f.disabled"
    fi
  done
}
py312_ready() {   # 실행 파일 + venv(ensurepip 는 python3.12-venv 에 들어 있다) + 헤더(python3.12-dev)
  command -v python3.12 >/dev/null 2>&1 || return 1
  python3.12 - <<'PY' >/dev/null 2>&1
import ensurepip, os, sys, sysconfig, venv
sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)
PY
}

dedup_deadsnakes

echo "[01] apt 패키지"
$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends \
  ca-certificates curl git software-properties-common gnupg \
  ffmpeg \
  build-essential \
  libglfw3 libgl1 libglx-mesa0 libgl1-mesa-dri libegl1 libglew2.2 libosmesa6 mesa-utils \
  libx11-6 libxext6 libxrender1 libxrandr2 libxinerama1 libxcursor1 libxi6 libxkbcommon0 \
  x11-utils xvfb xdotool tmux
if [ "$WITH_VNC" = 1 ]; then
  echo "[01] VNC/noVNC"
  $SUDO apt-get install -y --no-install-recommends tigervnc-standalone-server tigervnc-common novnc websockify fluxbox xterm
fi

echo "[01] Python 3.12 ($PY_METHOD)"
case "$PY_METHOD" in
  deadsnakes)
    if py312_ready; then
      echo "  python3.12 · venv · 헤더가 이미 있다 → 건너뜀"
    else
      if ! apt-cache show python3.12-venv >/dev/null 2>&1; then   # 22.04: 기본 저장소에 없다
        if [ -n "$(deadsnakes_files)" ]; then
          echo "  deadsnakes 는 이미 등록돼 있다 → 다시 넣지 않는다"
        else
          $SUDO add-apt-repository -y ppa:deadsnakes/ppa
        fi
        $SUDO apt-get update
      fi
      $SUDO apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev
    fi
    python3.12 --version ;;
  uv)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv python install 3.12
    uv python find 3.12 ;;
  *) echo "--python 은 deadsnakes 또는 uv"; exit 2 ;;
esac
echo "[01] 완료. 다음: bash scripts/server22/02_setup_project.sh [--cu128|--cu126|--cpu]"
