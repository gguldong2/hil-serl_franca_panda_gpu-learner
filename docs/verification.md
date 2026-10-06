# 검증 결과와 완료 기준 (PRD §10)

작성일 2026-09-16. 표기: **검증 완료** / **사용자 확인 필요** / **미검증** / **차단됨**.

## 1. 완료 기준 표

| 항목 | 상태 | 근거 (로그·파일) |
| --- | --- | --- |
| 환경 설치 | **검증 완료** | lerobot v0.6.1 + gym-hil 0.1.14 + torch 2.11.0+cpu 를 `.venv` 에 설치, `pip check` 이상 없음, `lerobot.rl.{actor,learner,gym_manipulator}` import 성공. `outputs/install_log.txt`, `docs/requirements-lock.txt` |
| GPU 사용 | **차단됨 (GPU 미보유)** — CPU 연산은 검증 완료 | NVIDIA GPU 없음(Intel Arc 내장). `torch.cuda.is_available()==False`. CPU 행렬곱·resnet18 fwd/bwd 정상. CUDA 학습 검증은 이 PC에서 불가 |
| 환경 동작 | **검증 완료** | `scripts/check_env.py`: `PandaPickCube-v0`/`PandaPickCubeKeyboard-v0` reset·step 30스텝, 자동 reset. `outputs/check_env_headless.log`, `outputs/gl_isolation.log` |
| 실시간 화면 | **사용자 확인 필요** (창 생성은 확인) | `PassiveViewerWrapper` 가 X 창 `"MuJoCo : Scene" 1706x1066` 을 생성함을 `xwininfo -root -tree` 로 확인(actor·teleop 이 실제로 스텝하는 환경의 뷰어). 화면에 그려진 내용은 사람이 봐야 함 → README §5.2 |
| 정책 관측 | **검증 완료** | `observation.images.front/wrist (1,3,128,128) float32 0~1`, `observation.state (1,18)`. PNG: `outputs/check_env/PandaPickCube-v0_{front,wrist}.png`. 정책 입력 카메라(`front`, `handcam_rgb` 128×128)와 뷰어 창은 별도 |
| 키보드 조작 | **사용자 확인 필요** (코드·리스너·홀드키는 확인) | 조작표는 `KeyboardController` 코드로 확정. 합성 키 주입에서 홀드 키(방향키·Shift·Ctrl)와 Enter/Esc 반영 확인, Space 토글은 합성 이벤트가 2회씩 도착해 판정 불가 → 실제 키보드로 `scripts/check_keyboard.py` 실행 필요 (README §5.4). `outputs/gl_isolation.log` T1 |
| 데이터 로딩 | **검증 완료** | `scripts/check_dataset.py`: v3.0, 30 ep/458 프레임, 영상 디코딩, 18차원 상태가 설정 min/max 내, 4차원 행동 유지, `ReplayBuffer.from_lerobot_dataset` → `sample(8)` 배치 성공, 이미지 스케일 불일치 발견·패치 후 일치. `outputs/check_dataset.log` |
| 학습 루프 | **검증 완료** | 스모크 실행(`configs/train_config_smoke.json`): actor 300스텝(3 에피소드) 전이 전송 → learner 온라인 버퍼 257(+오프라인 458) → SAC 업데이트 45회(손실 4종 기록) → 50 s 마다 `Push parameters`, actor `Load new parameters from Learner` 확인. `outputs/train/smoke/logs/learner_metrics.jsonl`, `learner_smoke.log`, `outputs/train/smoke_actor/logs/actor_smoke.log` |
| 저장·로딩 | **검증 완료** | `outputs/train/smoke/checkpoints/{000020,000040,last}` 생성(pretrained_model·algorithm·training_state, 39 MB 각). 새 프로세스 `scripts/eval_policy.py` 가 `last/pretrained_model` 로딩·5 에피소드 실행. `outputs/train/smoke/eval_000040.json` |
| 재개 | **검증 완료** | `--resume=true`: `Resuming from step 40`, `Loaded algorithm state`, 온라인 버퍼 257·오프라인 458 복원, optimizer 복원 후 55스텝까지 계속. `outputs/learner_resume_stdout.log`, metrics step 45~55. 복원 범위는 §3 |
| 자율 평가 | **검증 완료 (실행)** / 성능 개선은 **미검증** | 개입 없이 5 에피소드(seed 1000~1004): 성공 0/5, 모두 100스텝 truncated. 40회 업데이트만 한 정책이며 초기 정책과 비교 측정하지 않았으므로 개선 여부는 미검증 |
| 재현 문서 | **검증 완료** | `README.md` — 실제 성공한 명령·설정·버전으로 구성, 사용자 실행 항목은 🧑 로 구분 |

**실행 환경 준비**: 완료. **정책 성능 검증**: 미완료(충분한 인간 개입 학습이 남아 있음).

## 2. 스모크 실행 기록

설정 `configs/train_config_smoke.json` (batch 32, save_freq 20, online_steps 300, 나머지는 본 설정과 동일).

| 시각 | 사건 |
| --- | --- |
| 22:37 | learner 시작, `lerobot/resnet10` 로딩, 오프라인 버퍼 458 전이 변환(10 s), `Starting learner thread`, gRPC Ready(9 s) |
| 22:37 | actor 시작, `Connection with Learner established`, 뷰어 창 생성, 초기 파라미터 수신 |
| 22:38:36 | 에피소드 1 종료: 56스텝, **reward 1.0**(초기 정책이 우연히 큐브를 들어 올림), 전이 56개 전송, `Load new parameters` |
| 22:40:07 | 에피소드 2 종료: 100스텝 truncated, reward 0 |
| 22:40:54~22:44:12 | 최적화 5→45 스텝 (약 4.5 s/스텝, 0.21~0.22 Hz). 체크포인트 20, 40 저장 |
| 22:43:26 | actor 300스텝 완료(`Policy loop finished`), 종료 시 큐 close 순서로 인한 traceback 3개(상류 코드의 종료 경합, 결과 영향 없음) |
| 22:44:16 | learner SIGINT 로 정리 종료 |

지표 예(`learner_metrics.jsonl`, step 45): `loss_critic 0.022`, `loss_discrete_critic 0.019`, `loss_actor 1.478`, `loss_temperature 0.028`, `temperature 0.0099`, `replay_buffer_size 257`, `offline_replay_buffer_size 458`.
에피소드 메시지: `Episodic reward` 1.0/0.0/0.0, `Episode intervention` 0, `Intervention rate` 0.0 (**타임스텝 기준** = 개입 스텝/에피소드 스텝), `Policy frequency [Hz]` 9.0 → 2.7 (learner 학습 시작 후 CPU 경합).

첫 시도(batch 128, `outputs/train/smoke_run1_batch128/`)는 최적화 1스텝 15~25 s 로 체크포인트까지 수 시간이 필요해 중단하고 batch 32 로 재실행했다(루프 동작 자체는 동일하게 확인).

메모리: learner+actor 동시 실행 중 시스템 사용 4.3~4.9 GB / 7.5 GB (버퍼 5000/2000 설정).

## 3. 저장·재개 범위 (코드·실행으로 확인)

| 구성 요소 | 체크포인트 위치 | resume 시 복원 | 가중치만 로딩(eval) |
| --- | --- | --- | --- |
| 정책(actor MLP, 관측 인코더, 이산 그리퍼 critic) | `pretrained_model/model.safetensors` | ✅ (`make_policy(pretrained_path)`) | ✅ |
| 전·후처리 정규화 통계 | `pretrained_model/policy_*processor*` | ✅ | ✅ |
| critic 앙상블·타깃 critic·이산 타깃 critic·temperature | `algorithm/model.safetensors` | ✅ `Loaded algorithm state` | ✗ |
| optimizer 4종(actor, critic, discrete_critic, temperature) | `training_state/<name>/` | ✅ | ✗ |
| RNG 상태 | `training_state/rng_state.safetensors` | ✅ | ✗ |
| optimization step | `training_state/training_step.json` `step` | ✅ (`Resuming from step 40`) | ✗ |
| interaction step | 같은 파일 `interaction_step` | ⚠️ 저장 시점에 처리 중인 메시지가 없으면 0 으로 기록됨(상류 동작). 이번 실행 0 | ✗ |
| 온라인 replay buffer | `<output_dir>/dataset/` (LeRobotDataset 덤프, 저장마다 덮어씀) | ✅ 257 전이 복원 | ✗ |
| 오프라인 replay buffer(시연+개입) | `<output_dir>/dataset_offline/` | ✅ 458 복원 | ✗ |

체크포인트 이후의 업데이트(예: 41~45 스텝)는 재개 시 사라지고 마지막 체크포인트부터 다시 진행된다.

## 4. 산출물 위치

| 산출물 | 위치 |
| --- | --- |
| 설치 로그 | `outputs/install_log.txt` |
| 데이터셋 검증 | `outputs/check_dataset.log` |
| 환경 검증·카메라 PNG | `outputs/check_env_headless.log`, `outputs/check_env/*.png` |
| GL 격리 테스트·키 주입 | `outputs/gl_isolation.log`, `outputs/check_keyboard_inject.log` |
| 스모크 실행 | `outputs/smoke_run.log`, `outputs/train/smoke/` (checkpoints, logs, dataset, dataset_offline, eval_000040.json), `outputs/train/smoke_actor/logs/` |
| 첫 스모크(batch 128) | `outputs/train/smoke_run1_batch128/`, `..._actor/` |
| 재개 테스트 | `outputs/resume_run.log`, `outputs/learner_resume_stdout.log`, `outputs/actor_resume_stdout.log`, `outputs/train/smoke_actor_resume/` |
| 평가 | `outputs/train/smoke/eval_000040.json`, `outputs/eval_smoke_000040.log` |
| 패치 | `docs/patches/0001~0003` |
| 버전 고정 | `docs/requirements-lock.txt`, `docs/environment.md` §7 |
