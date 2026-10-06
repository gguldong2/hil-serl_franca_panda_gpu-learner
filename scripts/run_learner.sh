#!/usr/bin/env bash
# Learner 실행. 사용법: scripts/run_learner.sh [config json] [추가 --옵션들]
# 예: scripts/run_learner.sh configs/train_config_panda_keyboard_cpu.json
#     scripts/run_learner.sh configs/train_config_panda_keyboard_cpu.json --resume=true --config_path=outputs/train/panda_keyboard_cpu/checkpoints/last/pretrained_model/train_config.json
set -euo pipefail
source "$(dirname "$0")/env.sh"
CFG="${1:-configs/train_config_panda_keyboard_cpu.json}"; shift || true
# 렌더링을 하지 않는 learner 는 GL 이 필요 없다.
exec python -m lerobot.rl.learner --config_path="$CFG" "$@"
