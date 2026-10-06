#!/usr/bin/env bash
# 저장된 체크포인트를 순서대로 재생해서 "점점 나아지는 것"을 눈으로 본다.
#
# 학습 중에 보는 화면은 내가 개입한 것이 섞여 있어서 정책 실력이 아니다.
# 이 스크립트는 개입 래퍼가 없는 PandaPlaceCube-v0 에서 정책만으로 돌린다.
# 어린 체크포인트 -> 나이든 체크포인트 순서라, 같은 과제를 점점 잘 하는 게 보인다.
#
#   bash scripts/dashboard.sh                 # 먼저 다른 칸에서 (화면 보는 곳)
#   bash scripts/replay_checkpoints.sh        # 그다음 여기
#   bash scripts/replay_checkpoints.sh 5      # 체크포인트당 5 에피소드 (기본 3)
#   FROM=178000 TO=190000 bash scripts/replay_checkpoints.sh 5   # 그 구간만
#   EVERY=20000 bash scripts/replay_checkpoints.sh               # 2만 스텝 간격으로 건너뛰며
#   RUN=outputs/train/다른런 bash scripts/replay_checkpoints.sh
#
# 화면: http://localhost:8088/  의 위쪽 영상 칸. 학습 actor 와 같은 파일을 쓰므로
#       학습을 돌리는 중이라면 화면이 서로 엎치락뒤치락한다. 하나씩 돌릴 것.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh >/dev/null 2>&1
export MUJOCO_GL=egl

N="${1:-3}"
RUN="${RUN:-outputs/train/panda_place_gpu}"
TASK="${TASK:-PandaPlaceCube-v0}"
CKPT_DIR="$RUN/checkpoints"
OUT="$RUN/replay"

if [ ! -d "$CKPT_DIR" ]; then
  echo "[replay] 체크포인트가 없다: $CKPT_DIR"
  echo "         본 학습을 2,000 스텝 이상 돌린 뒤에 쓰는 스크립트다."
  echo "         다른 런을 보려면:  RUN=outputs/train/<이름> bash scripts/replay_checkpoints.sh"
  exit 1
fi

# 숫자 이름 체크포인트만, 오래된 것부터. 'last' 는 심볼릭 링크라 제외한다.
# FROM / TO 로 구간을 좁힐 수 있다 (특정 시기의 정책만 다시 보고 싶을 때).
#   FROM=178000 TO=190000 bash scripts/replay_checkpoints.sh 5
# EVERY 로 건너뛸 수도 있다 (체크포인트가 많을 때).
#   EVERY=10000 bash scripts/replay_checkpoints.sh
mapfile -t ALL < <(ls "$CKPT_DIR" | grep -E '^[0-9]+$' | sort -n)
STEPS=()
_last_kept=-1
for _s in "${ALL[@]}"; do
  _n=$((10#$_s))
  [ -n "${FROM:-}" ] && [ "$_n" -lt "$FROM" ] && continue
  [ -n "${TO:-}" ]   && [ "$_n" -gt "$TO" ]   && continue
  if [ -n "${EVERY:-}" ] && [ "$_last_kept" -ge 0 ]; then
    [ $((_n - _last_kept)) -lt "$EVERY" ] && continue
  fi
  STEPS+=("$_s")
  _last_kept=$_n
done
if [ "${#STEPS[@]}" -eq 0 ]; then
  echo "[replay] $CKPT_DIR 에 숫자 체크포인트가 없다."
  exit 1
fi

mkdir -p "$OUT"
echo "=============================================================="
echo "  체크포인트 ${#STEPS[@]} 개를 오래된 것부터 재생한다 (각 ${N} 에피소드)"
[ -n "${FROM:-}${TO:-}${EVERY:-}" ] && echo "  구간: FROM=${FROM:-처음} TO=${TO:-끝} EVERY=${EVERY:-전부}"
echo "  영상으로 남기려면 다른 칸에서 먼저:  python scripts/record_clips.py --continuous 60"
echo "  화면:  http://localhost:8088/     (bash scripts/dashboard.sh 로 먼저 띄울 것)"
echo "=============================================================="
echo

SUMMARY=()
for s in "${STEPS[@]}"; do
  P="$CKPT_DIR/$s/pretrained_model"
  [ -d "$P" ] || { echo "[replay] $s  건너뜀 (pretrained_model 없음)"; continue; }
  echo "--------------------------------------------------------------"
  echo "[replay] 스텝 $s  재생 중 ..."
  echo "--------------------------------------------------------------"
  J="$OUT/eval_$s.json"
  python -u scripts/eval_policy.py \
    --checkpoint "$P" --task "$TASK" \
    --n_episodes "$N" --out "$J"
  rate=$(python - "$J" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception:
    print("?"); raise SystemExit
eps = d.get("episodes", d if isinstance(d, list) else [])
n = len(eps)
k = sum(1 for e in eps if e.get("success"))
print(f"{k}/{n}" + (f" ({100*k//n}%)" if n else ""))
PY
)
  SUMMARY+=("  스텝 $s : 성공 $rate")
  echo
done

echo "=============================================================="
echo "  요약 — 아래로 갈수록 나중에 학습된 것이다"
echo "=============================================================="
printf '%s\n' "${SUMMARY[@]}"
echo
echo "  기록: $OUT/eval_<스텝>.json"
