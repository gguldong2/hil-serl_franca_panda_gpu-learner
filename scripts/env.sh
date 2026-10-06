# shellcheck shell=bash
# 이 파일은 실행하지 않고 source 한다:  source scripts/env.sh
# 역할:
#   1) ROS Jazzy 가 ~/.bashrc 에서 주입하는 PYTHONPATH 를 제거한다 (venv 패키지와 충돌 방지)
#   2) 프로젝트 전용 venv 를 활성화한다
#   3) Hugging Face 캐시를 프로젝트 안(WSL ext4 파일시스템)에 둔다
#   4) MuJoCo 렌더링 백엔드를 명시한다 (WSLg 창 표시용 glfw)

_HIL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HIL_ROOT="$_HIL_ROOT"

unset PYTHONPATH
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH

# shellcheck disable=SC1091
source "$_HIL_ROOT/.venv/bin/activate"

export HF_HOME="$_HIL_ROOT/cache/huggingface"
export HF_LEROBOT_HOME="$_HIL_ROOT/cache/huggingface/lerobot"
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$_HIL_ROOT/outputs"

# MuJoCo 오프스크린(정책 카메라) 렌더 백엔드. 뷰어 창은 MUJOCO_GL 과 무관하게 항상 GLFW 를 쓴다.
#   - WSL(WSLg) 또는 DISPLAY 가 있는 데스크톱: glfw
#   - DISPLAY 없는 서버/컨테이너: egl (NVIDIA 드라이버가 있으면 GPU 가속, 없으면 Mesa 소프트웨어)
#   - NVIDIA 서버에서 VNC 등으로 DISPLAY 를 쓸 때도 MUJOCO_GL=egl 을 명시하면 카메라 렌더만 GPU 로 돌릴 수 있다.
IS_WSL=0; grep -qi microsoft /proc/version 2>/dev/null && IS_WSL=1
if [ -z "${MUJOCO_GL:-}" ]; then
  if [ "$IS_WSL" = 1 ] || [ -n "${DISPLAY:-}" ]; then MUJOCO_GL=glfw; else MUJOCO_GL=egl; fi
fi
export MUJOCO_GL
# 렌더 드라이버. 기본은 Mesa 기본값(llvmpipe, CPU 소프트웨어 렌더).
#   GALLIUM_DRIVER=d3d12 로 두면 WSLg 의 Mesa d3d12 드라이버가 Windows GPU(Intel Arc) 가속 OpenGL 을 사용해
#   MuJoCo 카메라 렌더가 ~10배 빨라진다(2캠 343ms -> 27ms). 단, 검증 결과 **뷰어 창(passive viewer)과
#   오프스크린 렌더러를 한 프로세스에서 같이 쓰는 actor/teleop 에서는 "D3D12: Removing Device" 크래시**가
#   재현되므로, 뷰어가 없는 headless 실행(예: scripts/eval_policy.py 를 --viewer 없이)에서만 켠다.
if [ -n "${GALLIUM_DRIVER:-}" ]; then export GALLIUM_DRIVER; fi
# --- 서버 전용: NVIDIA EGL 라이브러리 (compute-only 컨테이너 대응) ---------------
# 이 컨테이너에는 NVIDIA 그래픽 라이브러리(libEGL_nvidia)가 없어서 MUJOCO_GL=egl 이
# Mesa llvmpipe(CPU)로 조용히 폴백한다. nvidia-gl/ 에 커널 드라이버와 **같은 버전**의
# 유저스페이스 GL 라이브러리를 풀어두고 EGL ICD 를 그쪽으로 돌린다(시스템 설치 아님).
# 드라이버가 올라가면 scripts/server22/05_nvidia_gl.sh 로 다시 받아야 한다.
if [ -f "$_HIL_ROOT/nvidia-gl/nvidia_egl_icd.json" ]; then
  export LD_LIBRARY_PATH="$_HIL_ROOT/nvidia-gl${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$_HIL_ROOT/nvidia-gl/nvidia_egl_icd.json"
fi

# 조작자용 화면 스트림. 이 경로에 JPEG 가 계속 갱신되고 대시보드가 MJPEG 로 흘려준다.
# VNC 의 GLX 가 소프트웨어라 MuJoCo 창을 띄우면 스텝당 ~690 ms 가 된다. 창 대신 이걸 쓴다.
export HIL_VIEW_STREAM="${HIL_VIEW_STREAM:-/dev/shm/hil_view.jpg}"
export HIL_VIEW_SIZE="${HIL_VIEW_SIZE:-512}"
export HIL_VIEW_CAMERAS="${HIL_VIEW_CAMERAS:-front,handcam_rgb}"

# 키보드 배치 파일. configs/keymap.json 을 고치면 조작 키가 바뀐다.
export GYM_HIL_KEYMAP="$_HIL_ROOT/configs/keymap.json"

export TOKENIZERS_PARALLELISM=false
# CPU 스레드: 기본 8 (WSL 노트북 기준). 서버에서는 OMP_NUM_THREADS 를 미리 export 해 덮어쓴다.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

cd "$_HIL_ROOT" || return 1
echo "[hil-serl] venv=$(command -v python)  HF_HOME=$HF_HOME  MUJOCO_GL=$MUJOCO_GL  EGL_ICD=${__EGL_VENDOR_LIBRARY_FILENAMES:+on}  DISPLAY=${DISPLAY:-<unset>}  WSL=$IS_WSL  GALLIUM_DRIVER=${GALLIUM_DRIVER:-<mesa default>}  cwd=$(pwd)"
