#!/usr/bin/env bash
# [시스템 패키지 + Python 3.12] Ubuntu 22.04 · 24.04. root 또는 sudo 필요.
# 사용법: bash scripts/server22/01_install_system.sh [--with-vnc] [--python deadsnakes|uv]
#   --with-vnc : TigerVNC + noVNC + fluxbox (사람 개입용 원격 데스크톱)
#   --python   : 3.12 설치 방법. 기본 deadsnakes(PPA). PPA 를 못 쓰면 uv (sudo 불필요)
#   xdotool 은 대시보드(8088)의 브라우저 조작에, tmux 는 learner/actor 를 SSH 가 끊겨도 살려 두는 데 쓴다.
#   build-essential(gcc) 는 torch.compile 이 쓰는 Triton 이 첫 실행 때 C 로 작은 모듈을 빌드하는 데 필요하다
#   (본 학습 설정은 use_torch_compile: true. 없으면 "Failed to find C compiler" 로 learner 가 죽는다).
#   libgl1-mesa-glx 는 24.04 에서 없어진 이름이라 그 내용물인 libgl1 + libglx-mesa0 을 직접 적는다(22.04 에서도 같다).
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
    $SUDO add-apt-repository -y ppa:deadsnakes/ppa
    $SUDO apt-get update
    $SUDO apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev
    python3.12 --version ;;
  uv)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv python install 3.12
    uv python find 3.12 ;;
  *) echo "--python 은 deadsnakes 또는 uv"; exit 2 ;;
esac
echo "[01] 완료. 다음: bash scripts/server22/02_setup_project.sh [--cu128|--cu126|--cpu]"
