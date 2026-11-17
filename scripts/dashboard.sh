#!/usr/bin/env bash
# 학습 진행 대시보드. 화면 보기 + 브라우저 조작 + 성공률 그래프.
#
#   bash scripts/dashboard.sh              # 기본 (포트 8088)
#   bash scripts/dashboard.sh --demo       # 학습 전에 화면 모양만 미리 보기
#   bash scripts/dashboard.sh --port 9000
#
# env.sh 를 직접 source 하지 않아도 되도록 여기서 대신 해 준다.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh >/dev/null 2>&1 || true
export DISPLAY="${DISPLAY:-:1}"

# --port 값 찾기 (없으면 8088)
PORT=8088
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  if [ "${args[$i]}" = "--port" ] && [ $((i + 1)) -lt ${#args[@]} ]; then
    PORT="${args[$((i + 1))]}"
  fi
done

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "[대시보드] 포트 $PORT 이 이미 쓰이고 있다. 이미 떠 있는 것 같다:"
  pgrep -af progress_dashboard || true
  # Ctrl+Z 로 멈춘(STAT 에 T) 대시보드는 보통 종료 신호(pkill)를 받지 못한 채 포트만 붙잡고 있다.
  # 2026-11-16 실제로 이랬다. 그래서 다시 띄우는 명령은 늘 -9 로 안내한다.
  for _pid in $(pgrep -f progress_dashboard); do
    case "$(ps -o stat= -p "$_pid" 2>/dev/null)" in
      T*) echo "  PID $_pid 는 멈춤(Stopped) 상태다 — Ctrl+Z 로 멈춘 것. 응답도 하지 않는다." ;;
    esac
  done
  echo "  브라우저에서 http://localhost:$PORT/ 를 열어 보고,"
  echo "  다시 띄우려면:  pkill -9 -f progress_dashboard; sleep 1; bash scripts/dashboard.sh"
  exit 1
fi

# --run 을 안 주면, 지금 돌고 있는 learner 의 출력 폴더를 자동으로 따라간다.
# (기본값을 쓰다가 엉뚱한 실험을 들여다본 적이 있다)
RUNARG=()
case " $* " in
  *" --run "*) ;;   # 직접 지정했으면 그대로 둔다
  *)
    for _p in /proc/[0-9]*; do
      _p=${_p#/proc/}
      [ "$_p" = "$$" ] && continue
      _comm=$(cat "/proc/$_p/comm" 2>/dev/null) || continue
      case "$_comm" in python*) ;; *) continue;; esac
      _args=$(tr '\0' '\n' < "/proc/$_p/cmdline" 2>/dev/null) || continue
      case "$_args" in
        *lerobot.rl.learner*)
          _cfg=$(printf '%s\n' "$_args" | sed -n 's/^--config_path=//p' | head -1)
          if [ -n "$_cfg" ] && [ -f "$_cfg" ]; then
            _out=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['output_dir'])" "$_cfg" 2>/dev/null)
            if [ -n "$_out" ]; then
              RUNARG=(--run "$_out")
              echo "[대시보드] 돌고 있는 learner 를 따라간다."
              echo "           설정 : $_cfg"
              echo "           런   : $_out"
              echo "           (다른 런을 보려면:  bash scripts/dashboard.sh --run outputs/train/<이름>)"
            fi
          fi
          break;;
      esac
    done
    if [ ${#RUNARG[@]} -eq 0 ]; then
      echo "[대시보드] 돌고 있는 learner 가 없다. 기본 런을 본다."
      echo "           특정 런을 보려면:  bash scripts/dashboard.sh --run outputs/train/<이름>"
      echo "           learner 를 먼저 띄우면 자동으로 그 런을 따라간다."
    fi;;
esac

exec python -u scripts/progress_dashboard.py "${RUNARG[@]}" "$@"
