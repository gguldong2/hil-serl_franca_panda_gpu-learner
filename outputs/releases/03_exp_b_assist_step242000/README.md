# 03_exp_b_assist — 학습 스텝 242,000 (보관본)

`outputs/train/03_exp_b_assist/checkpoints/242000` 을 2026-09-29 에 그대로 복사한 것이다.
원본은 학습 폴더에 그대로 남아 있지만, `train.sh restart` 나 폴더 정리로 사라질 수 있어
여기에 따로 둔다. sha256 전량 대조로 원본과 같음을 확인했다.

## 이 시점의 성적

| 항목 | 값 |
| --- | --- |
| 체크포인트 저장 시각 | 2026-09-29 09:23 |
| 학습 스텝(최적화 횟수) | 242,000 |
| 상호작용 스텝 | 822,025 누적 (actor 재시작 8회) |
| 에피소드 | 2,534판 · 전체 성공 1,653 (65%) |
| **최근 20판 성공률** | **100%** |
| 최근 20판 개입률 | 0% (사람 도움 없이) |
| 학습 시간 | 13시간 4분 |

태스크: `PandaPlaceCubeKeyboard-v0` (블록을 집어 바닥 목표 패드에 내려놓기)
설정: `configs/train_config_03_exp_b_assist.json`

## 폴더 구성 — 무엇이 무엇에 필요한가

| 폴더 | 크기 | 용도 |
| --- | --- | --- |
| `pretrained_model/` | 23 MB | **추론에 필요한 전부.** 정책 가중치(`model.safetensors`, 파라미터 5,797,641개) + 정규화 전/후처리기 + `train_config.json` |
| `algorithm/` | 45 MB | SAC critic·discrete critic·temperature. 학습을 이어 갈 때만 필요 |
| `training_state/` | 14 MB | optimizer 상태 + RNG. 학습을 이어 갈 때만 필요 |

리플레이 버퍼는 여기 없다 (원본 학습 폴더의 `dataset/`, `dataset_offline/`).
버퍼 없이 이어서 학습하면 버퍼가 빈 상태에서 다시 채워진다.

## 쓰는 법

평가(자율 실행, 사람 개입 없음):

    python scripts/eval_policy.py \
      --checkpoint outputs/releases/03_exp_b_assist_step242000/pretrained_model \
      --task PandaPlaceCube-v0 --n_episodes 20

`--task` 는 입력 래퍼가 없는 환경 ID 여야 한다(개입 불가 = 정책 실력만 측정).
