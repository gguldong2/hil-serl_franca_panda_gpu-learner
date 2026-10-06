#!/usr/bin/env bash
# 짧은 Actor–Learner 실행 검증 (PRD §8.2). 추가 구현.
#  1) learner 시작 -> gRPC Ready 응답 대기  2) actor 시작(뷰어 창 열림, 개입 없이 정책만 실행)
#  3) actor 가 online_steps 만큼 수집을 끝내면 checkpoint 생성을 기다린 뒤 learner 를 SIGINT 로 정리 종료
# 사용법: scripts/run_smoke_test.sh [config] [최대 대기 초]
set -uo pipefail
cd "$(dirname "$0")/.." && source scripts/env.sh
# env.sh 는 DISPLAY 가 있으면 glfw 를 고른다. 이 서버의 GLX 는 소프트웨어라
# 창이 없어도 스텝당 859 ms 가 된다. 렌더는 EGL(GPU)로 고정한다.
export DISPLAY="${DISPLAY:-:1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
CFG="${1:-configs/train_config_smoke.json}"; MAX_WAIT="${2:-2400}"; export SMOKE_CFG="$CFG"
OUT="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['output_dir'])" "$CFG")"
rm -rf "$OUT" "${OUT}_actor"; mkdir -p outputs
echo "[smoke] learner 시작 ($CFG) -> $OUT"
python -m lerobot.rl.learner --config_path="$CFG" > outputs/learner_smoke_stdout.log 2>&1 &
LPID=$!
echo "[smoke] gRPC Ready 대기"
for i in $(seq 1 120); do
  if python - <<'PY' 2>/dev/null; then echo "[smoke] learner Ready (after ~$((i*3))s)"; break; fi
import grpc; from lerobot.transport import services_pb2, services_pb2_grpc
import json, os
c = json.load(open(os.environ["SMOKE_CFG"]))["policy"]["actor_learner_config"]
stub = services_pb2_grpc.LearnerServiceStub(grpc.insecure_channel(f"{c['learner_host']}:{c['learner_port']}")); stub.Ready(services_pb2.Empty(), timeout=3)
PY
  if ! kill -0 $LPID 2>/dev/null; then echo "[smoke] learner 종료됨(실패)"; tail -20 outputs/learner_smoke_stdout.log; exit 1; fi
  sleep 3
done
echo "[smoke] actor 시작"
python -m lerobot.rl.actor --config_path="$CFG" --output_dir="${OUT}_actor" > outputs/actor_smoke_stdout.log 2>&1 &
APID=$!
t0=$(date +%s)
while kill -0 $APID 2>/dev/null; do
  if [ $(( $(date +%s) - t0 )) -gt "$MAX_WAIT" ]; then echo "[smoke] actor 시간 초과, 종료"; kill -INT $APID; fi
  sleep 10
done
echo "[smoke] actor 종료 (exit=$(wait $APID; echo $?)) ; checkpoint 생성 대기"
for i in $(seq 1 60); do
  [ -e "$OUT/checkpoints/last" ] && break; sleep 5
done
ls -la "$OUT/checkpoints" 2>/dev/null
echo "[smoke] learner 에 SIGINT"
kill -INT $LPID; wait $LPID; echo "[smoke] learner exit=$?"
echo "[smoke] 완료: $OUT"
