#!/usr/bin/env bash
# 시연 녹화. 배치 번호만 바꿔 가며 여러 번 돌린다.
#
#   bash scripts/record.sh 01           # data/place_demo_01 에 10개
#   bash scripts/record.sh 02 10        # data/place_demo_02 에 10개
#   bash scripts/record.sh 03 5         # 개수도 지정
#   bash scripts/record.sh --window 04  # MuJoCo 창도 같이 띄운다 (VNC 6080)
#   bash scripts/record.sh              # 인자 없이 = 지금까지 녹화한 배치 목록
#
# 같은 폴더에 이어서 녹화할 수 없다(FileExistsError). 그래서 배치마다 번호를 바꾼다.
# 다 모은 뒤 scripts/merge_demos.py 로 합친다.
#
# 주의: 시뮬레이터는 한 번에 하나만. teleop 이 떠 있으면 먼저 끄고 시작할 것
#       (같은 화면 파일을 두고 다투고, 키가 양쪽 로봇에 동시에 간다).
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh >/dev/null 2>&1
export DISPLAY="${DISPLAY:-:1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"   # 창을 띄워도 EGL 이면 속도가 정상이다

# --window 를 앞에 붙이면 MuJoCo 창도 같이 띄운다. 속도 차이는 거의 없다(23 -> 28 ms/스텝).
TASK_OVERRIDE=()
if [ "${1:-}" = "--window" ]; then
  shift
  TASK_OVERRIDE=(--env.task=PandaPlaceCubeKeyboardWindow-v0)
fi

BATCH="${1:-}"
N="${2:-10}"

if [ -z "$BATCH" ]; then
  echo "사용법: bash scripts/record.sh [--window] <배치번호> [에피소드수]"
  echo "  예:   bash scripts/record.sh 01"
  echo
  echo "지금까지 녹화한 배치:"
  found=0
  for d in data/place_demo_*; do
    [ -d "$d" ] || continue
    found=1
    if [ -f "$d/meta/info.json" ]; then
      python - "$d" <<'PY'
import json, sys, os
d = sys.argv[1]
i = json.load(open(os.path.join(d, "meta", "info.json")))
n, f = i.get("total_episodes", 0), i.get("total_frames", 0)
print(f"  완료   {d:28s} {n:3d} 에피소드  {f:6d} 프레임")
PY
    else
      echo "  미완성 $d   <- 한 개도 저장되지 않았다. 지우고 다시: rm -rf $d"
    fi
  done
  [ "$found" = 1 ] || echo "  (아직 없음)"
  echo
  echo "성공/실패까지 보려면:  python scripts/check_dataset.py --repo_id local/<이름> --root <폴더>"
  exit 1
fi

ROOT="data/place_demo_${BATCH}"
if [ -e "$ROOT" ]; then
  echo "[record] $ROOT 이 이미 있다. 다른 번호를 쓰거나, 버릴 것이면 지우고 다시 해라:"
  echo "  rm -rf $ROOT"
  exit 1
fi

# 시뮬레이터가 이미 돌고 있으면 경고한다. 두 개가 같이 돌면 화면이 깜빡이고 키가 양쪽으로 간다.
others="$(pgrep -f "lerobot.rl.gym_manipulator|lerobot.rl.actor|scripts/eval_policy.py" 2>/dev/null | tr '\n' ' ')"
if [ -n "${others// /}" ]; then
  echo "[record] 경고: 시뮬레이터가 이미 돌고 있다 (PID: $others)"
  echo "         teleop 이나 학습 actor 를 먼저 끄고 시작할 것. 안 그러면"
  echo "         화면이 깜빡이고 키가 두 로봇에 동시에 간다."
  echo "         확인:  pgrep -af gym_manipulator"
  echo
fi

cat <<MSG
[record] $ROOT 에 ${N} 개를 녹화한다. 20 fps, 한 에피소드 최대 40초.
  화면:  bash scripts/dashboard.sh  ->  http://localhost:8088/  (화면을 클릭하면 거기서 조작된다)
  순서:  Space(조종권) -> 방향키로 블록 위 -> 왼쪽Ctrl 내리기 -> g 잡기
         -> w 로 3cm 이상 들기 -> 방향키로 초록 패드 위 -> 왼쪽Ctrl 내리기 -> 왼쪽Shift 놓기
  실패하면 Esc 말고 r 을 눌러 그 에피소드를 버린다(개수에 들어가지 않는다).
  저장될 때마다 (성공, 보상 1.0, N초) 가 찍힌다. (실패, 보상 0.0, 40초) 면 r 로 버릴 것.
  ${N} 개를 채우면 알아서 끝난다. 중간에 Ctrl+C 해도 그때까지 저장된 것은 그대로 남는다.

MSG
exec python -u -m lerobot.rl.gym_manipulator \
  --config_path=configs/env_config_panda_place_record.json \
  "${TASK_OVERRIDE[@]}" \
  --dataset.root="$ROOT" \
  --dataset.repo_id="local/place_demo_${BATCH}" \
  --dataset.num_episodes_to_record="$N" \
  "${@:3}"
