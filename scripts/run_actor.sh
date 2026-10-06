#!/usr/bin/env bash
# Actor 실행 (뷰어 창 + 키보드). 사용법: scripts/run_actor.sh [config json] [추가 --옵션들]
# learner 가 먼저 떠 있어야 한다. learner 와 같은 output_dir 을 쓰면 lerobot v0.6.1 의
# TrainPipelineConfig.validate() 가 FileExistsError 를 내므로, actor 는 <output_dir>_actor 를 사용한다.
set -euo pipefail
source "$(dirname "$0")/env.sh"
# env.sh 는 DISPLAY 가 있으면 MUJOCO_GL=glfw 로 잡는다. 이 서버의 GLX 는 소프트웨어(llvmpipe)라
# 뷰어 창이 없어도 오프스크린 카메라 렌더가 스텝당 ~690 ms 로 느려진다. 키 입력에는 DISPLAY 가
# 필요하므로 DISPLAY 는 두고 렌더 백엔드만 EGL(GPU)로 되돌린다. 창을 꼭 띄우려면 MUJOCO_GL=glfw 로 덮어쓴다.
export DISPLAY="${DISPLAY:-:1}"
export MUJOCO_GL="${HIL_ACTOR_GL:-egl}"

# --window 를 첫 인자로 주면 MuJoCo 창도 같이 띄운다. EGL 이면 속도 차이가 거의 없다
# (측정: 창+EGL 28 ms/스텝 vs 창 없음 23 ms/스텝).
WINDOW=()
if [ "${1:-}" = "--window" ]; then shift; WINDOW=(--env.task=PandaPlaceCubeKeyboardWindow-v0); fi

echo "[actor] MUJOCO_GL=$MUJOCO_GL  DISPLAY=$DISPLAY  창=$([ ${#WINDOW[@]} -gt 0 ] && echo 있음 || echo 없음)  (화면: bash scripts/dashboard.sh -> http://localhost:8088/)"
CFG="${1:-configs/train_config_panda_keyboard_cpu.json}"; shift || true
OUT="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['output_dir'])" "$CFG")"
exec python -m lerobot.rl.actor --config_path="$CFG" --output_dir="${OUT}_actor" "${WINDOW[@]}" "$@"
