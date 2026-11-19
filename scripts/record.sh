#!/usr/bin/env bash
# 시연 녹화. 배치 번호만 바꿔 가며 여러 번 돌린다.
#
#   bash scripts/record.sh 01           # data/place_demo_01 에 10개
#   bash scripts/record.sh 02 10        # data/place_demo_02 에 10개
#   bash scripts/record.sh 03 5         # 개수도 지정
#   bash scripts/record.sh --window 04  # MuJoCo 창도 같이 띄운다 (VNC 6080)
#   bash scripts/record.sh              # 인자 없이 = 지금까지 녹화한 배치 목록
#
# 다른 로봇은 CFG 로 녹화 설정을 바꾼다. 배치 폴더는 설정의 dataset.root 뒤에 _번호 를 붙인다:
#   CFG=configs/env_config_ur5e_place_record.json bash scripts/record.sh 01
#       -> data/ur5e_place_cube_demo_01   (병합하면 data/ur5e_place_cube_demo)
#   (Panda 기본 설정만 예전 이름 data/place_demo_NN 을 그대로 쓴다)
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

# 띄우기 전 자동 점검 (0.05초). 통과하면 아무것도 찍지 않고, 막히면 이유를 보여주고 멈춘다.
# 이 서버는 컨테이너가 다시 뜰 때 드라이버 버전이 바뀌어 EGL 이 죽는 일이 되풀이된다.
# 건너뛰려면 앞에 HIL_SKIP_PREFLIGHT=1 (예: xvfb-run 으로 VNC 없이 돌릴 때).
if [ "${HIL_SKIP_PREFLIGHT:-}" != "1" ] && [ -x scripts/preflight.sh ]; then
  if ! _pf="$(bash scripts/preflight.sh --quick 2>&1)"; then
    printf '%s\n' "$_pf"
    echo
    echo "[preflight] 환경이 준비되지 않았다.  고치기:  bash scripts/preflight.sh --fix"
    echo "            (점검을 건너뛰려면 앞에 HIL_SKIP_PREFLIGHT=1)"
    exit 1
  fi
fi

# 녹화 설정. 다른 로봇은 CFG 로 바꾼다 (teleop.sh 와 같은 방식).
CFG="${CFG:-configs/env_config_panda_place_record.json}"
# RunPod 의 tmux 세션(hil)은 CFG 를 학습 설정(train_config_*)으로 잡아 둔다. 그 값을 그대로 쓰면
# record.sh 가 학습 설정을 읽고 죽는다(2026-11-18). 그래서 학습 설정이면 멈추고 맞는 명령을 알려 준다.
case "$(basename "$CFG")" in
  train_config_*)
    echo "[record] CFG 가 학습 설정이다: $CFG  (tmux 세션 hil 이 잡아 둔 값)"
    # 안내하는 로봇을 세션의 학습 설정에 맞춘다(RunPod 세션은 ur5e). 전에는 늘 panda 를 안내해 헷갈렸다(11-19).
    _robot=panda; case "$(basename "$CFG")" in *ur5e*) _robot=ur5e;; esac
    echo "  시연 녹화에는 env_config 를 쓴다. 이렇게 띄운다:"
    echo "    CFG=configs/env_config_${_robot}_place_record.json bash scripts/record.sh ${1:-01}"
    exit 1;;
esac
read -r TASK DROOT < <(python -c "import json,sys; c=json.load(open(sys.argv[1])); print(c['env']['task'], c['dataset']['root'])" "$CFG") || exit 1
# 배치 폴더 접두어. Panda 기본 설정은 예전부터 쓰던 data/place_demo_NN 을 유지한다.
if [ "$DROOT" = "data/panda_place_cube_demo" ]; then PREFIX="data/place_demo"; else PREFIX="$DROOT"; fi

# --window 를 앞에 붙이면 MuJoCo 창도 같이 띄운다. 속도 차이는 거의 없다(23 -> 28 ms/스텝).
# 창 태스크는 설정의 태스크를 따른다 (예전에는 Panda 로 박혀 있었다).
TASK_OVERRIDE=()
if [ "${1:-}" = "--window" ]; then
  shift
  TASK_OVERRIDE=(--env.task="${TASK/Keyboard-v0/KeyboardWindow-v0}")
fi

BATCH="${1:-}"
N="${2:-10}"

if [ -z "$BATCH" ]; then
  echo "사용법: bash scripts/record.sh [--window] <배치번호> [에피소드수]"
  echo "  예:   bash scripts/record.sh 01"
  echo
  echo "지금까지 녹화한 배치:"
  found=0
  for d in "${PREFIX}"_*; do
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

ROOT="${PREFIX}_${BATCH}"
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
[record] $TASK  ($CFG)
[record] $ROOT 에 ${N} 개를 녹화한다. 20 fps, 한 에피소드 최대 40초.
  화면:  bash scripts/dashboard.sh  ->  http://localhost:8088/  (화면을 클릭하면 거기서 조작된다)
  순서:  Space(조종권) -> 방향키로 블록 위 -> 왼쪽Ctrl 내리기 -> g 잡기
         -> w 로 3cm 이상 들기 -> 방향키로 초록 패드 위 -> 왼쪽Ctrl 내리기 -> 왼쪽Shift 놓기
  실패하면 Esc 말고 r 을 눌러 그 에피소드를 버린다(개수에 들어가지 않는다).
  저장될 때마다 (성공, 보상 1.0, N초) 가 찍힌다. (실패, 보상 0.0, 40초) 면 r 로 버릴 것.
  ${N} 개를 채우면 알아서 끝난다. 중간에 Ctrl+C 해도 그때까지 저장된 것은 그대로 남는다.

MSG
exec python -u -m lerobot.rl.gym_manipulator \
  --config_path="$CFG" \
  "${TASK_OVERRIDE[@]}" \
  --dataset.root="$ROOT" \
  --dataset.repo_id="local/$(basename "$PREFIX")_${BATCH}" \
  --dataset.num_episodes_to_record="$N" \
  "${@:3}"
