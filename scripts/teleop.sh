#!/usr/bin/env bash
# 조작 연습. 아무것도 저장하지 않는다.
#
#   bash scripts/teleop.sh            # 화면은 대시보드(8088)에서 본다
#   bash scripts/teleop.sh --window   # MuJoCo 창도 같이 띄운다 (VNC 6080). 속도는 거의 같다
#
# 조작은 Space 로 조종권을 가져온 뒤에 먹는다. 키 배치는 configs/keymap.json.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh >/dev/null 2>&1
export DISPLAY="${DISPLAY:-:1}"

# 렌더 백엔드는 항상 EGL(GPU). 느려지는 원인은 MuJoCo 창이 아니라 MUJOCO_GL=glfw 였다.
# 측정: 창+EGL 28 ms/스텝(35 fps) vs 창+GLFW 1116 ms/스텝(0.9 fps). 창 자체는 거의 무료다.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

TASK_OVERRIDE=()
if [ "${1:-}" = "--window" ]; then
  shift
  TASK_OVERRIDE=(--env.task=PandaPlaceCubeKeyboardWindow-v0)
  cat <<'MSG'
[teleop] MuJoCo 창 모드 (VNC 6080 에 창이 뜬다). MUJOCO_GL=egl 이라 속도는 정상이다.
  창 안을 한 번 클릭해 포커스를 준 뒤 Space 부터 누른다.
MSG
else
  cat <<'MSG'
[teleop] 창 없이 돈다. 화면은 대시보드에서 본다:
  bash scripts/dashboard.sh  ->  http://localhost:8088/  (화면을 클릭하면 거기서 조작된다)
MSG
fi

echo "[teleop] Space 를 먼저 눌러야 나머지 키가 먹는다. 끝낼 때는 Ctrl+C."
echo
exec python -u -m lerobot.rl.gym_manipulator \
  --config_path=configs/env_config_panda_place_teleop.json \
  "${TASK_OVERRIDE[@]}" "$@"
