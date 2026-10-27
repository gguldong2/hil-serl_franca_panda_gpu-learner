# 공식 구성 대비 변경 내역 (PRD §9)

네 가지로 나눠 기록한다. "공식 구성"은 그대로 사용한 것, "환경별 설정"은 이 PC 때문에 바꾼 값, "호환성 수정"은 실제 오류를 고치기 위한 코드 패치, "추가 기능"은 공식에 없는 보조 스크립트다.

## 1. 공식 구성 (그대로 사용)

| 항목 | 출처 | 사용한 파일·커밋 |
| --- | --- | --- |
| HIL-SERL 시뮬레이션 절차 | https://huggingface.co/docs/lerobot/hilserl_sim | `third_party/lerobot/docs/source/hilserl_sim.mdx` @ v0.6.1 |
| LeRobot 코드 | https://github.com/huggingface/lerobot | tag `v0.6.1` = `7e241bd630a3719a56157a497ce5d08f244784f1` |
| gym_hil 코드 | https://github.com/huggingface/gym-hil | `f89b3e5556798c44ef95083d92b670afa89d110a` (= PyPI 0.1.14) |
| 학습 설정 원본 | https://huggingface.co/datasets/lerobot/config_examples/blob/main/rl/gym_hil/train_config.json | repo sha `4803f5f3…` → `configs/train_config_panda_keyboard_cpu.json` 의 출발점 |
| 환경/기록 설정 원본 | 같은 저장소 `rl/gym_hil/env_config.json` | → `configs/env_config_panda_keyboard_*.json` |
| 시연 데이터 | https://huggingface.co/datasets/lilkm/pick_cube_franka_panda_30 | revision `f80a680a117d0c8573ab453e3fd9e792e308dfa4` |
| 알고리즘 | SAC + 이산 그리퍼 critic (`lerobot.rl.algorithms.sac`), 정책 `gaussian_actor`, 비전 인코더 `lerobot/resnet10` (frozen) | 하이퍼파라미터(lr, discount 0.97, utd 2, temperature_init 0.01, hidden 256×2, latent 64 등) 공식값 유지 |
| 보상·종료 | gym_hil `PandaPickCubeGymEnv` sparse 보상: 블록이 초기 높이보다 0.1 m 이상 올라가면 1.0 & terminated. 블록이 샘플링 영역 ±0.05 m 밖이면 terminated. `gripper_penalty=-0.02` (`complementary_info.discrete_penalty`) | 변경 없음 |
| 행동 의미 | 4차원 `[dx, dy, dz, gripper]`; xyz ∈ [-1,1] × step 0.025 m, gripper ∈ {0: -1(ctrl 감소), 1: 유지, 2: +1(ctrl 증가)} | 변경 없음 |

## 2. 환경별 설정 (값 변경, 코드 변경 없음)

파일: `configs/train_config_panda_keyboard_cpu.json` (원본 `train_config.json` 대비)

| 키 | 공식값 | 변경값 | 이유 |
| --- | --- | --- | --- |
| `env.task` | `PandaPickCubeGamepad-v0` | `PandaPickCubeKeyboard-v0` | 기본 장치를 키보드로. lerobot `make_robot_env` 는 `processor.control_mode` 를 gym 에 넘기지 않고 **task 이름으로만** 컨트롤러가 결정됨 (`gym_hil/__init__.py` 등록 kwargs `use_gamepad` 유무) |
| `env.processor.control_mode` | `gamepad` | `keyboard` | 문서 일관성용. gym_hil 경로에서는 실제 효과 없음 (위 참조) |
| `policy.device`, `policy.storage_device` | `cuda`, `cpu` | `cpu`, `cpu` | NVIDIA GPU 없음 |
| `algorithm.use_torch_compile` | `true` | `false` | CPU + 7.5 GiB RAM 에서 컴파일 비용·메모리 부담. 알고리즘 수식은 동일 |
| `batch_size` | 256 | 64 (스모크: 32) | CPU 학습 속도. 실측: batch 128 → 최적화 1스텝 15~25 s, batch 32 → 4.5 s. 알고리즘 변경 아님 |
| `policy.online_buffer_capacity` | 100000 | 5000 | 버퍼가 이미지를 float32 로 저장(전이당 2×3×128×128×4 B ≈ 393 KB). 100k → 39 GB 로 7.5 GiB RAM 초과. 5000 → 최대 약 2 GB |
| `policy.offline_buffer_capacity` | 100000 | 2000 | 시연 458 프레임 + 개입 전이 저장용. 최대 약 0.8 GB |
| `policy.online_steps` | 1000000 | 20000 (스모크: 300) | 실행 예산 명시. 짧은 검증과 수렴 실험을 구분 |
| `save_freq` | 2000000 | 200 (스모크: 20) | 공식값은 `online_steps`(1M)에 도달해야만 저장 → 짧은 실행에서 체크포인트가 생기지 않음 |
| `log_freq` | 1 | 10 (스모크: 5) | 로컬 JSONL 지표 파일 크기 |
| `wandb.enable` | `true` | `false` (`mode: offline` 설정만 남김) | 기본 실행은 로컬 기록만. `wandb` 패키지는 hilserl extra 에 포함되지 않아 미설치 |
| `dataset.revision` | 없음 | `f80a680a…` | 재현성 고정 |
| `output_dir` / `job_name` | null / default | `outputs/train/panda_keyboard_cpu` / `panda_keyboard_cpu` | 프로젝트 내부 저장 |
| `policy.actor_learner_config.learner_host` | `127.0.0.1` | `localhost` | 이 WSL(mirrored 네트워킹)에서 IPv4 루프백 TCP 연결이 타임아웃됨(순수 소켓으로 재현). `localhost` 는 `[::1]` 에도 바인딩되어 IPv6 경로로 연결됨. `docs/troubleshooting.md` §7 |
| `policy.concurrency` | threads/threads | (공식값 유지) | 조사 중 `processes` 로 바꿔 봤으나 원인이 아니어서 복원 |

실행 방식:

| 항목 | 공식 | 변경 | 이유 |
| --- | --- | --- | --- |
| actor `output_dir` | learner 와 동일 | `<output_dir>_actor` (`scripts/run_actor.sh` 가 `--output_dir` 로 지정) | v0.6.1 `TrainPipelineConfig.validate()` 는 `output_dir` 이 이미 존재하고 `resume=false` 면 `FileExistsError`. learner 가 먼저 디렉터리를 만들기 때문에 actor 가 같은 경로로는 시작 불가 (dry-run 으로 재현) |
| 렌더 드라이버 | (문서에 언급 없음) | 기본 llvmpipe, headless 평가만 `GALLIUM_DRIVER=d3d12` | `docs/environment.md` §4 |
| 환경 변수 | - | `scripts/env.sh`: `unset PYTHONPATH`, `HF_HOME`, `HF_LEROBOT_HOME`, `MUJOCO_GL=glfw`, `OMP_NUM_THREADS=8` | ROS 간섭 차단, 캐시를 WSL 파일시스템에 고정 |

## 3. 호환성 수정 (코드 패치, `docs/patches/*.patch`)

모두 `third_party/lerobot` (editable install) 에 적용했고 `git diff` 를 패치 파일로 남겼다. 재적용: `cd third_party/lerobot && git apply ../../docs/patches/000*.patch`.

| 패치 | 실제 오류 | 수정 파일·내용 | 재실행 결과 |
| --- | --- | --- | --- |
| `0001-buffer-uint8-image-scale.patch` | `learner.initialize_offline_replay_buffer` → `make_dataset(cfg)` 가 `return_uint8=True` 로 데이터셋을 만들어 오프라인 버퍼 이미지가 **0~255 float** 로 저장되는 반면, actor 온라인 관측은 `VanillaObservationProcessorStep` 에서 **0~1** 로 정규화됨. `scripts/check_dataset.py` 로 배치 max=255.0 확인. 같은 정규화(ImageNet mean/std)를 두 스케일에 적용하면 학습 데이터가 불일치. upstream `main` 에도 미수정 | `src/lerobot/rl/buffer.py` `_lerobotdataset_to_transitions`: state/next_state 의 uint8 텐서를 `float32/255` 로 변환 | 재실행 후 배치 max=1.0, "일치" (`outputs/check_dataset.log`) |
| `0002-learner-local-metrics-jsonl.patch` (추가 기능 성격) | 학습 손실·개입률 등이 `wandb_logger` 로만 전송되어 W&B 미사용 시 로컬에 남지 않음 | `src/lerobot/rl/learner.py`: `_append_jsonl()` 추가, 학습 지표(`kind=train`)와 actor 에피소드 메시지(`kind=interaction`)를 `<output_dir>/logs/learner_metrics.jsonl` 에 기록, 에피소드 메시지를 INFO 로그로도 출력 | 스모크 실행 결과 참조 (`docs/verification.md`) |
| `0003-envs-configs-reset-joint-positions-type.patch` | learner 시작 직후 `cfg.to_dict()` → `draccus.encode` 에서 `TypeError: typing.Any cannot be used with isinstance()` (필드 `fixed_reset_joint_positions: Any | None`) | `src/lerobot/envs/configs.py`: 타입을 `list[float] | None` 으로 — upstream `main` 과 동일한 수정 | learner 가 설정 출력·체크포인트 저장 단계까지 진행 |

## 4. 추가 기능 (공식에 없는 보조 스크립트, `scripts/`)

| 파일 | 범위·목적 |
| --- | --- |
| `env.sh` | venv 활성화 + 환경 변수. 매 터미널에서 `source scripts/env.sh` |
| `check_dataset.py` | PRD §6 데이터셋 검증(포맷·차원·보상·통계 일치·영상 디코딩·버퍼 배치·스케일) |
| `check_env.py` | 환경 reset/step, 프로세서 파이프라인, 정책 관측 형태, 스텝 시간, 카메라 프레임 PNG 저장 |
| `check_keyboard.py` | 키보드 개입 경로 확인. `--inject` 는 합성 키 주입 자동 검증, 옵션 없이 실행하면 사용자가 실제 키로 확인 |
| `run_learner.sh`, `run_actor.sh` | 공식 `python -m lerobot.rl.{learner,actor}` 를 env.sh 와 함께 실행. actor 는 `_actor` output_dir 지정 |
| `run_smoke_test.sh` | learner → Ready 확인 → actor → 종료 → 체크포인트 확인 → learner SIGINT 순서 자동화 |
| `server22/00~04_*.sh`, `server22/make_bundle.sh` | Ubuntu 22.04 + NVIDIA 서버(컨테이너) 이식용. 사전 점검(읽기 전용), apt+Python 3.12(deadsnakes/uv), 고정 커밋 clone+venv+torch cu128+패치, CUDA·EGL·데이터·환경 검증, TigerVNC+noVNC 디스플레이, 전송 번들. **22.04 에서 미실행** — `docs/server-ubuntu22.04-gpu.md` |
| `configs/train_config_panda_keyboard_gpu.json`, `train_config_smoke_gpu.json` | GPU 서버용 설정(device cuda, batch 256·torch.compile 공식값 복원, buffer 20000/5000, online_steps 100000, save_freq 2000/50). 미실행 |
| `env.sh` WSL 감지 | `/proc/version` 으로 WSL 을 감지해 `MUJOCO_GL` 기본값을 분기(WSL/DISPLAY → glfw, 없으면 egl). 24.04 WSL 동작은 재확인 |
| `eval_policy.py` | 자율 평가. 공식 `lerobot.rl.eval_policy` 는 v0.6.1 에서 존재하지 않는 `env_cfg.pretrained_policy_name_or_path` 참조, `make_robot_env` 반환 tuple 을 env 로 사용, 전처리 없이 raw obs 를 `select_action` 에 전달하는 등 그대로 실행 불가. actor 의 추론 경로를 재사용해 가중치만 로딩·개입 없이 N 에피소드 실행, JSON 기록 |

## 5. 확인한 문서–코드 불일치 (수정하지 않고 기록)

| 문서/출력 | 실제 코드 |
| --- | --- |
| gym_hil `KeyboardController` 도움말 "Backspace: End episode with FAILURE", "ESC: Exit" | 코드는 `Key.esc` → 실패 표시(episode_end_status="failure"). Backspace 는 처리 없음. 종료 키 없음(Ctrl+C) |
| 코드 `elif key == keyboard.Key.r` (재기록) | pynput `Key` 에 `r` 멤버 없음 → `AttributeError` 가 조용히 무시되어 키보드 재기록 기능 없음 |
| 도움말 "Right Ctrl: Open gripper" (행동값 2.0) | `EEActionWrapper` 는 2.0 → +1 → `ctrl` 증가. Robotiq 2f85 `fingers_actuator` 는 ctrl 255 = 닫힘. 즉 **우Ctrl 은 손가락을 닫고(집기), 좌Ctrl 은 연다.** 시연 데이터의 gripper 열도 {1.0, 2.0}만 존재(2.0 으로 집음) |
| hilserl_sim 문서 "`pip install -e ".[hilserl]"`" | 그대로 동작. 단 `wandb` 는 포함되지 않음 |
| hilserl_sim 문서 "A Nvidia GPU" 필요 | CPU 로도 import·환경·학습 업데이트 동작(속도는 별도) |
| `env.processor.reset.control_time_s=15.0`(150 스텝) | gymnasium 등록 `max_episode_steps=100` 의 `TimeLimit` 래퍼가 먼저 100 스텝(시뮬레이션 10 s)에 truncated → 실효 에피소드 상한 100 스텝 |
| `env_config.json` 의 `observation.add_joint_velocity_to_observation`, `image_preprocessing` | gym_hil 경로의 `make_processors` 는 이 항목을 사용하지 않음(실물 로봇 전용). 상태 18차원은 gym_hil 이 직접 qpos7+qvel7+gripper1+tcp3 으로 구성 |

## 6. 22.04 GPU 서버(A100 컨테이너)에서 추가한 것 (2026-09-17)

실행 기록·측정값·문제별 근거는 [`docs/server-ubuntu22.04-gpu.md` §8](server-ubuntu22.04-gpu.md). 여기는 목록만.

### 6.1 호환성 수정 (코드 패치)

| 대상 | 파일 | 내용 |
| --- | --- | --- |
| gym-hil | `docs/patches/gym-hil/0001-keyboard-listener-join-on-stop.patch` | `KeyboardController.stop()` 에서 pynput 리스너를 `join(timeout=2.0)` 한다. 안 하면 살아 있는 리스너 스레드와 `glfw.terminate()` 가 겹쳐 **종료 시점에** SIGSEGV(패치 전 3/19, 후 0/27). 작업이 끝난 뒤 터지므로 결과에는 영향 없음 |
| 이 저장소 | `scripts/eval_policy.py` | `select_action()`(cuda)과 `post.process_action()`(cpu) 결과를 `torch.cat` 하기 전에 device 를 맞춘다. CPU 전용 WSL 에서는 드러날 수 없던 **GPU 전용** 오류 |

### 6.2 검증 스크립트 수정

| 파일 | 내용 |
| --- | --- |
| `scripts/check_keyboard.py` | `--inject` 의 키 주입을 pynput `Controller` → **XTEST 키코드 직접**(`--inject-backend xtest`, 기본값)으로 교체. pynput Controller 는 같은 프로세스 리스너에 키를 2번 넣어 SPACE 토글이 무효가 되고, xdotool 키심 이름은 왼쪽 modifier 를 같이 눌러 ±z 가 상쇄된다. **사람이 실제로 누르는 키는 원래 정상** — 즉 이전 실패는 거짓 실패였다 |

### 6.3 추가·수정한 서버 스크립트

| 파일 | 내용 |
| --- | --- |
| `scripts/server22/05_nvidia_gl.sh` (신규) | compute-only 컨테이너에서 EGL 을 GPU 로 돌리기 위해, 커널 드라이버와 **같은 버전**의 런파일을 받아 압축만 풀어 `nvidia-gl/` 에 둔다(시스템 설치 아님). 호스트 드라이버가 바뀌면 재실행 필요 |
| `scripts/env.sh` | `nvidia-gl/nvidia_egl_icd.json` 이 있으면 `LD_LIBRARY_PATH` 와 `__EGL_VENDOR_LIBRARY_FILENAMES` 를 자동 설정(WSL·일반 데스크톱에서는 no-op). 헤더 로그에 `EGL_ICD=on` 표시 |
| `scripts/server22/02_setup_project.sh` | 패치 적용 루프를 저장소별로 일반화해 `docs/patches/gym-hil/*.patch` 도 적용 |
| `scripts/server22/04_start_vnc.sh` | websockify 를 `localhost:6080` 에 바인드(이전에는 `0.0.0.0:6080` 으로 외부 노출) |

원본은 각 파일 옆에 `*.bundle-orig` 로 남겨 뒀다.

## 7. PandaPlaceCube 태스크 추가 (2026-09-21)

블록 하나를 집어 바닥의 목표 패드 위에 내려놓는 새 태스크. 상류 gym-hil 에는 없다.
운영 절차는 별도 런북(HTML) 참조.

### 7.1 새 환경

| 파일 | 내용 |
|---|---|
| `third_party/gym-hil/gym_hil/assets/place_cube_scene.xml` | 블록 1개(빨강) + 충돌하지 않는 목표 패드(초록, 반지름 3 cm). `goal_pos` 센서 추가. IK용 mocap `target` 은 그대로 둔다 |
| `third_party/gym-hil/gym_hil/envs/panda_place_gym_env.py` | `PandaPlaceCubeGymEnv`. reset 마다 블록을 `x 0.3~0.5, y ±0.15` 에서 연속 랜덤(목표에서 최소 8 cm), sparse/dense 보상, `get_task_diagnostics()` 제공 |
| `gym_hil/__init__.py` | `PandaPlaceCube{Base,Viewer,,Keyboard,Gamepad}-v0` 5종 등록, `max_episode_steps=200` |
| `gym_hil/envs/__init__.py`, `gym_hil/wrappers/factory.py` | 클래스 노출 및 factory 화이트리스트 추가 |

성공 판정은 세 조건 동시 만족: 블록–목표 수평거리 < 3 cm, 블록이 바닥에 놓임, 그리퍼 열림.
추가로 "한 번은 3 cm 이상 들어올렸을 것"(`require_lift`)을 요구해 밀어 넣기를 배제한다.
검증: 스크립트 pick&place 8/8 성공(75스텝), 거부 대상 4종(쥔 채 대기·쥔 채 통과·밀어넣기·옆에 놓기) 4/4 정상 거부.

### 7.2 설정

`configs/env_config_panda_place_{teleop,record}.json`, `configs/train_config_{panda_place_gpu,place_smoke_gpu}.json`.
pick 설정과 다른 점: task, `control_time_s=20.0`(200스텝), `discount 0.97 -> 0.99`(에피소드가 길어짐),
`offline_buffer_capacity 5000 -> 10000`(시연 30개 × 최대 200스텝).

### 7.3 추가 스크립트

- `scripts/merge_demos.py` — 배치로 나눠 녹화한 데이터셋을 `lerobot.datasets.aggregate` 로 합친다.
  record 모드가 `LeRobotDataset.create()` 를 쓰기 때문에 기존 디렉터리에 이어서 녹화할 수 없다(FileExistsError).
- `scripts/watch_progress.py` — `learner_metrics.jsonl` 을 읽어 구간별 성공률·개입률·버퍼 크기 추이를 보여준다. `--follow` 로 상주 가능.
- `scripts/update_dataset_stats.py` — 녹화본에서 `observation.state` min/max 를 재계산해 train config 에 써 넣는다.
- `scripts/check_dataset.py` — `--root` 옵션 추가(로컬 데이터셋). 그리퍼 액션 주석 수정(실제는 0=열기, 2=닫기).

### 7.4 새로 고친 버그

| 패치 | 증상 | 원인 |
|---|---|---|
| `docs/patches/0004-gym-manipulator-close-env-on-exit.patch` | 녹화 종료 시 매번 SIGSEGV(exit 139) | `gym_manipulator.main()` 이 `env.close()` 를 부르지 않아 MuJoCo 뷰어 스레드와 pynput X RECORD 스레드가 인터프리터 종료 중 충돌. 데이터 손실은 없었다. 수정 후 3/3 exit=0 |
| `docs/patches/0005-actor-clean-shutdown.patch` | actor 가 `online_steps` 를 다 돌고도 종료되지 않음 | 정리 단계에서 daemon 스레드를 타임아웃 없이 join. `receive_policy` 가 learner 파라미터 스트림에 블록되면 메인 스레드가 영영 빠져나오지 못한다. 정상 종료 경로에서 `shutdown_event` 도 세우지 않았다 |
| `docs/patches/gym-hil/0003-keyboard-rerecord-key.patch` | 키보드 `r`(재녹화)가 전혀 동작하지 않음 | 존재하지 않는 `keyboard.Key.r` 을 읽어 `AttributeError` 로 빠짐(문자 키는 `KeyCode`). 게다가 키보드 경로에서 `episode_end_status = "rerecord_episode"` 를 설정하지 않아 두 군데가 막혀 있었다 |

### 7.5 알아둘 동작

- **actor 는 에피소드가 끝날 때만 전이를 learner 로 보낸다**(`actor.py` 의 `if done or truncated:`).
  따라서 스모크의 `online_steps` 가 `max_episode_steps` 보다 작으면 learner 가 아무것도 학습하지 않고 체크포인트도 안 생긴다.
  place 스모크는 `online_steps=500`(≈2.5 에피소드)으로 잡았다.
- **그리퍼 방향이 코드 변수명과 반대다.** `hil_wrappers` 의 `gripper_command == "open"` 이 액션 2.0 을 내고,
  액션 2.0 은 `ctrl=255` 로 손가락을 **닫는다**(driver 관절 0.796 rad 측정). 즉 오른쪽 Ctrl = 잡기, 왼쪽 Ctrl = 놓기.
- **`Enter` 는 환경 판정을 무시하고 보상을 1.0 으로 덮어쓴다.** 공개 데이터셋 `ansocho/gym_hil_arrange_boxes_test` 는
  15 에피소드 전부 블록을 쥔 채로 Enter 를 눌러 "성공"으로 기록돼 있어 재사용할 수 없었다.

### 7.6 배선 검증 결과

`run_smoke_test.sh` 로 learner+actor 전 구간 통과: learner Ready 9s, actor exit=0, learner exit=0,
체크포인트 000100~000800 8개, `learner_metrics.jsonl` 91줄, 최적화 3.8~6.1 Hz.
`eval_policy.py --task PandaPlaceCube-v0` 정상 동작 확인.
모든 패치는 깨끗한 체크아웃에 순서대로 적용되며 결과가 현재 작업본과 바이트 단위로 일치함을 확인했다.

### 7.7 조작 게이트 (2026-09-22 확인)

키보드 입력은 **`Space` 로 개입을 켜야만** 로봇에 전달된다.
`hil_wrappers.InputsControlWrapper.step()` 의 `if is_intervention: action = gamepad_action` 때문이며,
연습(teleop)·녹화(record)·학습(actor) 모두 동일하다. 화면에 상태 표시가 없어 착각하기 쉽다.

XTEST 로 키를 주입해 VNC 화면을 전후 비교한 실측:

| 상태 | 방향키·Shift 3초 조작 후 화면 변화 |
|---|---|
| 개입 OFF | 187 픽셀 (노이즈) |
| 개입 ON | 4,590 픽셀 |

### 7.8 진행 시각화

- `scripts/progress_dashboard.py` — `learner_metrics.jsonl` 을 읽는 로컬 웹 대시보드(기본 127.0.0.1:8088).
  성공률·개입률 이동평균, 버퍼 크기, critic loss 를 SVG 로 그리고 10초마다 갱신한다. 외부 의존성 없음.
  VS Code 포트 포워딩으로 랩탑 브라우저에서 본다.
- `scripts/watch_progress.py` — 같은 내용의 터미널 버전(`--follow`).

### 7.9 렌더링 경로 교체 — 조작이 느렸던 진짜 이유 (2026-09-22)

VNC(Xtigervnc) 의 GLX 는 `llvmpipe`(소프트웨어)다. MuJoCo 패시브 뷰어 창을 띄우면
`MUJOCO_GL=glfw` 가 되어 정책 카메라 렌더링까지 전부 CPU 로 간다. 실측:

| 조합 | 스텝당 | 가능 최대 fps |
|---|---|---|
| 뷰어 O + glfw (기존) | 688 ms | 1.5 |
| 뷰어 X + glfw | 487 ms | 2.1 |
| **뷰어 X + egl (GPU)** | **15.7 ms** | **64** |

즉 `fps: 10` 설정으로 녹화해도 실제로는 1.5 fps 로 돌고 있었다. 조치:

- `gym_hil/PandaPlaceCubeKeyboard-v0` 을 `use_viewer: False` 로 바꿨다. 창이 필요하면
  새로 등록한 `PandaPlaceCubeKeyboardWindow-v0` 을 쓴다(GPU GLX 가 있는 데스크톱용).
- 새 래퍼 `gym_hil/wrappers/view_stream_wrapper.py` — EGL 로 조작자용 화면(정면+손목, 512px)을
  한 장 더 렌더해 JPEG 로 원자적 교체 저장한다. `HIL_VIEW_STREAM` 이 있을 때만 붙는다.
  `MujocoGymEnv.__init__` 이 모델의 오프스크린 한도를 128 로 덮어쓰므로 래퍼에서 다시 올린다.
- 모든 실행을 `MUJOCO_GL=egl` 로 바꿨다.

### 7.10 fps 를 설정에서 바꿀 수 있게

`gym_manipulator` 의 `gym.make` 가 `control_dt` 를 넘기지 않아 설정의 `fps` 와 환경의
제어 주기(기본 10 Hz)가 따로 놀았다. `control_dt = 1/cfg.fps`, `ee_step_size = 0.25/cfg.fps`
(초당 25 cm 유지)를 전달하도록 고쳤다(`docs/patches/0004-gym-manipulator-close-env-and-fps.patch`).

설정을 20 fps 로 올렸다: `control_time_s=20.0` → 400 스텝, `max_episode_steps=600`,
`discount 0.99 -> 0.995`(지평 보정). 녹화 실측 199/200 프레임, 실시간 유지.

### 7.11 키 배치를 파일로

`KeyboardController` 를 표 기반으로 다시 짜고 `configs/keymap.json`(`GYM_HIL_KEYMAP`)으로 뺐다.
글자 키는 대소문자와 Ctrl 조합(제어문자)을 정규화하고, 좌/우 모디파이어는 `Key.shift`↔`Key.shift_l`
같은 별칭을 함께 등록한다. 중복 키는 실행 시 경고한다.

기본 배치: 화살표(x·y) · `w` 위로 · 왼쪽 Ctrl 아래로 · `g` 잡기 · 왼쪽 Shift 놓기 ·
Space 조종권 · Enter 성공 · Esc 실패 · `r` 재녹화.

**조종권 게이트**: `InputsControlWrapper.step()` 의 `if is_intervention: action = keyboard_action`
때문에 Space 를 누르기 전에는 모든 키가 무시된다. 실측(화면 변화 픽셀): OFF 0 → ON 160,622(↑),
181,580(w), 176,295(왼쪽 Ctrl), 33,744(g), 33,548(왼쪽 Shift).

### 7.12 대시보드에서 직접 조작

`scripts/progress_dashboard.py` 에 추가:
- `/view.mjpg` — 화면 스트림을 MJPEG 로 중계. 페이지 맨 위 "지금 시뮬레이터" 카드.
- `/key` (POST) — 브라우저 키 입력을 `xdotool` 로 X 디스플레이에 주입. `configs/keymap.json`
  에 있는 키만 허용한다. `127.0.0.1` 바인딩. `--no-control` 로 끌 수 있다.
- `--demo` — 학습 전에 화면 모양을 미리 보는 가짜 데이터 모드.

브라우저 조작 실측: ↑ 160,622 · `w` 181,580 · `g` 33,637 · 왼쪽 Shift 33,381 픽셀 변화.

### 7.13 실행 스크립트 (2026-09-22)

`python: command not found` 는 `source scripts/env.sh` 누락이다. 매번 빠뜨리기 쉬워
환경설정을 안에서 처리하는 래퍼를 추가했다.

| 스크립트 | 역할 |
|---|---|
| `scripts/dashboard.sh` | 대시보드. 포트 중복이면 알려 준다 |
| `scripts/teleop.sh` | 연습. `--window` 로 `PandaPlaceCubeKeyboardWindow-v0`(창 모드, 느림) |
| `scripts/record.sh <번호> [개수]` | 배치 녹화. 인자 없이 실행하면 배치 목록과 완료/미완성을 보여준다 |

`scripts/progress_dashboard.py` 는 표준 라이브러리만 쓰므로 가상환경 없이 시스템 `python3` 으로도 돈다.
출력 버퍼링 때문에 안내문이 늦게 뜨던 것은 `python -u` 로 해결.

전체 흐름 검증: dashboard.sh HTTP 200 → teleop.sh 화면 스트림 26 KB →
브라우저 `/key` 경유 조작 (Space+↑ 165,649 픽셀, g 41,008 픽셀) → `--window` 모드에서 MuJoCo 창 1개 확인.

### 7.14 체크포인트 재생 — "나아지는 것"을 영상으로 보기

**문제.** 학습 중 대시보드 영상에는 사람의 개입이 섞여 있어 정책 실력의 지표가 되지 못한다.
상류 HIL-SERL 에는 정책 성능을 **영상으로** 비교하는 수단이 없다(실물 로봇을 눈으로 보는 전제).

**추가한 것.** `scripts/replay_checkpoints.sh`
- `outputs/train/<run>/checkpoints/` 의 숫자 체크포인트를 `sort -n` 으로 오래된 것부터 순회
- 각각에 대해 `scripts/eval_policy.py --task PandaPlaceCube-v0` (키보드 래퍼 없음 = 개입 불가) 실행
- 에피소드별 결과 JSON 을 `outputs/train/<run>/replay/eval_<스텝>.json` 에 남기고 성공률 요약표 출력
- `RUN=` / `TASK=` 환경변수로 다른 런·다른 태스크 지정 가능

**영상 경로.** `gym_hil/PandaPlaceCube-v0` 도 `wrappers/factory:make_env` 를 거치므로
`maybe_wrap_view_stream`(factory.py:82)이 붙는다. 즉 eval 도 `/dev/shm/hil_view.jpg` 를 갱신하고
대시보드의 `/view.mjpg` 로 그대로 나간다. **학습 actor 와 같은 파일을 공유**하므로 동시 실행 시 화면이 섞인다.

**검증.** `outputs/train/smoke_gpu` 의 체크포인트 000050 / 000500 두 개를 심볼릭 링크한 임시 런으로
전 구간 실행 → exit 0, 두 체크포인트 모두 eval 수행, JSON 생성, 요약표 출력 확인.
실행 직후 `/dev/shm/hil_view.jpg` mtime 이 갱신됨(25 KB)을 확인 → eval 도 영상이 나간다.

**런북.** v6 으로 갱신. §07 에 "영상으로 보기 — 로봇이 움직이는 화면"(학습 중 화면은 개입이 섞여
실력이 아니라는 경고 포함), §09 에 체크포인트 재생 절차, §10 에 `fbsetbg` 경고 행 추가.

**부수.** VNC 기동 시 뜨는 `fbsetbg: Something went wrong while setting the wallpaper` 는
fluxbox 의 바탕화면 설정 실패일 뿐이며 Xtigervnc / fluxbox / websockify 는 모두 정상 동작한다.

**같이 고친 것 — `scripts/run_actor.sh` 의 렌더 백엔드.**
`env.sh` 는 `DISPLAY` 가 있으면 `MUJOCO_GL=glfw` 로 잡는다. 키 입력(pynput X)에는 `DISPLAY` 가 필요한데,
이 서버의 GLX 는 llvmpipe 라 **뷰어 창이 없어도** 오프스크린 카메라 렌더가 스텝당 ~690 ms 로 떨어진다.
`run_actor.sh` 가 `DISPLAY` 는 유지하고 `MUJOCO_GL` 만 `egl` 로 되돌리도록 고쳤다(`HIL_ACTOR_GL` 로 덮어쓰기 가능).
첫 줄에 `[actor] MUJOCO_GL=egl DISPLAY=:1` 을 찍어 확인할 수 있게 했다. 실행 확인 완료.

### 7.15 녹화 진행 표시 · 중간 종료 검증

**문제 1 — 녹화 중 아무 출력이 없었다.** `gym_manipulator.py:83` 의 `logging.basicConfig(level=logging.INFO)` 는
루트 로거에 핸들러가 이미 붙어 있으면 아무 일도 하지 않는다. 다른 패키지가 먼저 핸들러를 달아
루트 레벨이 `WARNING` 으로 남았다(측정: `logging.getLogger().level == WARNING`, `INFO` 라인 0개).
그래서 `Saving episode N` 이 한 줄도 보이지 않았고, 사용자는 지금 몇 개째인지 알 수 없었다.

**문제 2 — 에피소드 시간이 누적됐다.** `episode_start_time` 을 루프 진입 전 한 번만 잡고 다시 잡지 않아
`episode_time` 이 계속 커졌다(상류 버그).

**패치 `docs/patches/0006-record-progress-output.patch`** (43줄)
- `logging.getLogger().setLevel(logging.INFO)` 를 추가해 레벨을 직접 올린다
- 에피소드 종료 시 `episode_start_time` 을 다시 잡는다
- 로깅 설정과 무관하게 보이도록 `print(..., flush=True)` 로 진행을 찍는다:
  `[녹화] 3/10 개 저장됨  (성공, 보상 1.0, 11초)   <- 여기서 Ctrl+C 로 멈춰도 이만큼은 남는다`
  `r` 로 버리면 `[녹화] 버림 (r). 저장된 것은 그대로 2/10 개. 다시 녹화한다.`
- `git apply --check --reverse` 로 현재 트리에 정확히 대응함을 확인

**중간 종료 검증 (런북의 기존 서술이 틀렸다).**
런북은 "Ctrl+C 로 끊으면 그 배치 전체가 쓸 수 없게 될 수 있다"고 적고 있었다. 실측 결과 **틀렸다**:
- 8개 설정으로 띄워 3개 저장 후 SIGINT → `meta/info.json` 의 `total_episodes=3`, `LeRobotDataset` 로딩 성공(237 프레임)
- 같은 방법의 2개짜리 배치와 `merge_demos.py` 로 병합 → 5 에피소드 395 프레임, 로딩 성공
- 이유: `dataset.save_episode()` 가 에피소드마다 메타까지 갱신한다. `finalize()` 는 `push_to_hub` 일 때만 불리지만
  로컬 사용에는 필요하지 않았다.
런북 §04 를 사실에 맞게 고쳤다.

**측정 중 빠질 뻔한 함정.** SIGINT 테스트를 백그라운드 잡(`cmd &`)으로 돌렸더니
셸이 POSIX 규칙대로 SIGINT/SIGQUIT 를 `SIG_IGN` 으로 바꿔 놓아(`/proc/<pid>/status` 의 `SigIgn=0x1001006`,
비트 1 = SIGINT) **Ctrl+C 가 안 먹는다고 잘못 결론 낼 뻔했다.** `signal.signal(SIGINT, SIG_DFL)` 후
`execvp` 하는 런처를 거쳐 다시 재면 `SigIgn=0x1001000`, exit 130 으로 정상 종료한다.
포그라운드 터미널에서 직접 누르는 Ctrl+C 는 처음부터 문제가 없었다.

**부수 확인.** `update_dataset_stats.py` 의 인자는 `--repo_id / --root / --train_config` 다
(`--dataset / --config` 가 아니다). 런북 §05 는 처음부터 옳게 적혀 있었다.
`--env.processor.reset.control_time_s=4.0` 같은 CLI 덮어쓰기가 동작함도 확인(632프레임/8에피소드 = 79프레임 ≈ 4초 @20fps).

### 7.16 느렸던 진짜 원인 — 창이 아니라 `MUJOCO_GL=glfw` (앞선 결론 수정)

`PandaPlaceCubeViewer-v0` / `PandaPlaceCubeKeyboardWindow-v0` 를 두 백엔드로 각각 80스텝 측정
(창이 실제로 떴는지 `DISPLAY=:1 xdotool search --onlyvisible` 로 `MuJoCo : PlaceCubeScene` 확인):

| 창 | `MUJOCO_GL` | 스텝당 | 실효 fps |
|---|---|---|---|
| 없음 | `egl` | **22.9 ms** | 43.7 |
| **있음** | `egl` | **28.2 ms** | 35.5 |
| 없음 | `glfw` | 859 ms | 1.2 |
| 있음 | `glfw` | 1,116 ms | 0.9 |

**결론: MuJoCo 창은 거의 무료다(23 → 28 ms).** 40배 느려지게 만든 것은 오직 `MUJOCO_GL=glfw` 이며,
창이 없어도 똑같이 느려진다(정책 카메라 오프스크린 렌더가 llvmpipe 경로를 탄다).
`glxinfo -B`: `OpenGL renderer string: llvmpipe (LLVM 15.0.7, 256 bits)`.

앞서 §7.x 에서 "창을 띄우면 40배 느려진다"고 적었던 것은 **틀렸다.** 두 변수를 함께 바꿔 측정한 탓이다.
런북 §01·§07·§10 과 스크립트를 모두 고쳤다:
- `scripts/teleop.sh --window` → `MUJOCO_GL=glfw` 설정을 제거, 항상 `egl`
- `scripts/record.sh --window`, `scripts/run_actor.sh --window` 추가 (둘 다 `egl` 유지)

### 7.17 대시보드에서 방향키가 페이지를 스크롤하던 문제

`preventDefault()` 는 있었지만 `CTL`(조작 켜짐)일 때만 동작했고, 그 토글은 버튼 클릭뿐이었다.
사용자는 화면만 클릭하고 방향키를 눌러 페이지가 스크롤됐다.
- 시뮬레이터 이미지에 `mousedown` → 조작 자동 켜짐 (켜지면 이미지에 초록 외곽선)
- 조작이 켜진 동안 `ArrowUp/Down/Left/Right`, `Space`, `PageUp/Down`, `Home/End` 의 기본 동작을 차단
  (`{passive:false}` 로 리스너 등록)

### 7.18 성공 클립 자동 녹화 — `scripts/record_clips.py`

view-stream 이 이미 만들어 둔 JPEG 을 링버퍼(기본 25초 = 500프레임)에 모으고,
actor 로그의 `Global step N: Episode reward: X` 를 tail 해서 `X >= --threshold`(기본 0.5) 일 때만
`ffmpeg -f image2pipe -vcodec mjpeg` 로 mp4 를 떨어뜨린다. **렌더·디코드를 다시 하지 않는다.**

- `--all` 실패도 저장 · `--seconds` 링버퍼 길이 · `--max_clips` 상한 · `touch /dev/shm/hil_clip_now` 수동 저장
- 결과: `outputs/clips/<스텝>_성공_<시각>.mp4`. 파일명이 스텝 순이라 정렬하면 발전 순서가 된다

**검증.** eval_policy 로 화면을 흘리며 로그에 성공/실패/성공을 주입 →
성공 2건 저장·실패 1건 건너뜀·수동 저장 1건 동작. 산출물 `ffprobe` 확인:
h264 1024×512, 20 fps, 120 프레임, 6.0초, 178 KB. 첫 프레임을 PNG 로 뽑아 눈으로도 확인.
CPU 사용률 `top -b` 3회 측정 **0.0 / 0.5 / 0.2 %**, 메모리 0.0%.

### 7.19 확인한 버퍼 동작 (질문 답변용 근거)

- `ReplayBuffer.add()` 는 `self.position = (self.position + 1) % self.capacity` — **원형(ring) 버퍼, FIFO**.
  꽉 차면 **가장 오래된 것부터** 덮어쓴다 (`buffer.py:231-232`).
- 시연 버퍼는 `initialize_offline_replay_buffer()` 가 `ReplayBuffer.from_lerobot_dataset(...)` 로
  **한 번만** 만든다 (`learner.py:853-889`). `from_lerobot_dataset` 도 내부에서 `add()` 를 쓰므로
  적재 후 `position = len(dataset)` → 개입 전이는 시연 뒤에 이어 붙고, 10,000 을 넘기면 인덱스 0 부터,
  즉 **원본 시연부터** 밀려난다.
- 개입 전이가 들어가는 곳은 그 **같은 객체**다 (`learner.py:1010` → `offline_replay_buffer.add(**transition)`).
- 온라인 버퍼는 새 학습에서 **빈 채로** 시작한다 (`learner.py:823-830`; `from_lerobot_dataset` 는 `--resume` 때만).
- `from_lerobot_dataset` 는 `capacity < len(dataset)` 이면 예외를 던진다 → `offline_buffer_capacity` 는
  시연 프레임 수보다 커야 한다 (30개 ≈ 4,500 < 10,000, 문제없음).

### 7.20 한 에피소드 길이

`TimeLimitProcessorStep(max_episode_steps=int(control_time_s * fps))` (`gym_manipulator.py:404,463`)
→ 20.0 × 20 = **400 스텝 = 20초**. 성공하면 `terminated=True` 로 **즉시** 끝나고 20초를 기다리지 않는다.
CLI 로 덮어쓸 수 있음을 확인: `--env.processor.reset.control_time_s=4.0` → 8에피소드 632프레임(79프레임 ≈ 4초 @20fps).

### 7.21 에피소드 길이 20초 → 40초

`control_time_s` 를 네 설정 모두 40.0 으로 올렸다 (`env_config_panda_place_teleop/_record`,
`train_config_panda_place_gpu`, `train_config_place_smoke_gpu`). 20 fps 기준 400 → **800 스텝**.

스모크는 한 에피소드가 800 스텝이 되어 `online_steps` 1,000 으로는 에피소드가 하나도 끝나지 않는다
(예전에 겪은 "체크포인트가 안 생김" 문제와 같은 원인). `online_steps` 3,000, `steps` 2,000, `save_freq` 200 으로 조정.

**비용 분석.** 실질적으로 벽시계 시간 하나뿐이다.
- 실패 롤아웃이 2배 길어져 학습 초반 시간당 에피소드 수가 절반. 성공 시 즉시 종료라 잘하게 된 뒤엔 차이 없음.
- 온라인 버퍼 20,000 프레임 = 에피소드 50 → 25개분. 필요하면 `online_buffer_capacity` 를 올린다(메모리 384 GB 중 8 GB 사용).
- 할인율 0.995 의 유효 시야는 200 스텝(10초)인데 과제는 스크립트로 75 스텝이면 끝난다 → 신용 할당에 문제 없음.

### 7.22 `scripts/record_clips.py` 확장 — 정보 띠 + 연속 녹화

- **정보 띠**: 클립 위쪽에 두 줄. `스텝 · 학습 횟수 · 경과 시간 · 보상/성공` / `최근 N판 성공률 · 개입률 · 온라인·시연 버퍼`.
  값은 actor 로그와 `learner_metrics.jsonl` 에서 읽는다(대시보드와 같은 소스).
  PIL 로 **띠 PNG 한 장**만 그리고 ffmpeg `overlay` 필터가 얹는다 — 프레임마다 파이썬을 돌리지 않는다.
  폰트는 `NanumGothicCoding` (한글), 없으면 DejaVuSansMono 폴백.
- **연속 녹화** `--continuous N`: 성공 여부와 무관하게 N초마다 `seg_<스텝>_<시각>.mp4` 로 통째 저장.
  `--no-episode-clips` 로 에피소드 클립을 끄고 연속만 남길 수 있다.
- 파일명 접두사 정리: `ep_` (에피소드) / `seg_` (구간) / `manual_` (수동).

**검증.** 가짜 `learner_metrics.jsonl` + 주입한 actor 로그로 실행 →
성공 클립 1건·실패 건너뜀·연속 구간 2건 저장. 산출물에서 프레임을 뽑아 띠가 제대로 렌더됨을 눈으로 확인
(`스텝 14,400 · 학습 24,417회 · 경과 0분 9초 · 보상 +1.00 성공` / `최근3판 성공 66% · 개입 18.0% · 온라인 16,278 · 시연 4,512`).
**한계**: 인코딩이 동기라 구간 저장 중 몇 프레임이 빠진다(12초 간격 테스트에서 구간이 17초로 밀림). 120초 간격이면 영향 미미.

### 7.23 `scripts/train.sh` — 시작·재개·진행확인 한 곳에서

```
bash scripts/train.sh status     # 어디까지 왔나 (학습 안 돌 때도 읽힌다)
bash scripts/train.sh learner    # 새로 시작인지 이어서인지 알아서 판단
bash scripts/train.sh actor      # --window 로 MuJoCo 창도
bash scripts/train.sh restart    # 처음부터 (확인 물음)
```
- `checkpoints/last/pretrained_model/train_config.json` 이 있으면 `--resume=true` 를 자동으로 붙인다.
- actor 는 출력 폴더가 있으면 `--resume=true` (`TrainPipelineConfig.validate()` 의 `FileExistsError` 회피용).
- `status` 는 `training_step.json`, 두 버퍼의 `meta/info.json`, `learner_metrics.jsonl` 을 읽어
  학습 스텝/상호작용 스텝/버퍼 크기/성공률/개입률/체크포인트 수를 출력하고 `<출력폴더>/STATUS.txt` 에도 남긴다.
  **폴더가 없으면 만들지 않는다** — 만들면 learner 가 "이미 있다"며 거부한다.

**재개 검증 (런북의 기존 서술이 또 틀렸다).** 런북은 "재개하면 온라인 버퍼는 비어서 시작한다"고 적고 있었다.
실제로 smoke_gpu 를 재개해 보니:
```
Valid checkpoint found: resume=True detected, resuming previous run
Resume training load the online dataset      <- 온라인 버퍼 복원
load offline dataset                         <- 시연+개입 버퍼 복원
Loading training state from .../checkpoints/last
```
그리고 500 → 700 스텝까지 실제로 이어서 학습했다. 복원 경로:
`learner.py:648-668` 이 체크포인트마다 `replay_buffer.to_lerobot_dataset()` 로 `<출력폴더>/dataset`,
`offline_replay_buffer.to_lerobot_dataset()` 로 `<출력폴더>/dataset_offline` 을 통째로 쓴다.
`initialize_replay_buffer` / `initialize_offline_replay_buffer` 가 `cfg.resume` 일 때 그걸 다시 읽는다.

**주의로 추가한 것.** 재개 시 lerobot 은 체크포인트에 저장된 설정을 읽으므로
(`handle_resume_logic` → `TrainRLServerPipelineConfig.from_pretrained`),
**학습 시작 후 `configs/*.json` 을 고쳐도 반영되지 않는다.** 에피소드 길이 같은 것은 시작 전에 정해야 한다.

### 7.24 런북 전면 갱신 (v11)

에피소드를 40초로 바꾸고 측정값이 바뀌면서 문서 곳곳이 어긋나 있었다. 13곳을 사실에 맞게 고쳤다:

| 위치 | 전 | 후 |
|---|---|---|
| 머리말 요약 | 20 fps (400스텝 / 20초) | 20 fps (800스텝 / 40초) |
| 렌더 성능 표 | 창+GLFW 688 / 창없음+EGL 15.7 | 창없음+EGL **22.9** · 창+EGL **28.2** · 창없음+GLFW 859 · 창+GLFW 1,116 |
| 에피소드 시간 | 20초 기준 서술 | 40초 |
| 녹화 지침 | "200 스텝(20초)까지" | "800 스텝(40초)까지" |
| 개입 설명 | "200 스텝(20초)이 한계" | "800 스텝(40초)이 한계" |
| 창 띄우기 | `run_actor.sh --window` | `train.sh actor --window` |
| 스모크 | "200 스텝만", 대기 400초, "online_steps 200 밑" | "3,000 스텝", 대기 900초, "800 밑" |
| 환경 규격 | 에피소드 400스텝 · 스텝 15.7 ms | 800스텝 · 22.9 ms(창 28.2) |
| 할인율 설명 | "400 스텝이라 유효 지평을 맞췄다" | "유효 지평 200스텝(10초). 과제는 75스텝이면 끝나므로 800 상한과 무관" |

추가한 것:
- §00 에 **"오늘 칠 명령만 모아서"** — 0~5단계를 칸 번호와 함께 한 블록으로
- 새 스크립트 4개를 파일 목록에 등재 (`train.sh`, `record_clips.py`, `replay_checkpoints.sh`, `eval_policy.py`)

**같이 고친 것 — `scripts/run_smoke_test.sh` 의 렌더 백엔드.** `run_actor.sh` 와 같은 문제
(`env.sh` 가 `DISPLAY` 를 보고 `glfw` 선택)가 있어 `MUJOCO_GL=egl` 을 강제하도록 고쳤다.

### 7.25 40초 설정이 실제로는 30초에서 잘리고 있었다 (7.21 의 누락)

`control_time_s` 만 40.0 으로 올리고 **gymnasium 등록의 `max_episode_steps` 를 그대로 두었다.**
`gym_hil/__init__.py` 의 PandaPlaceCube 계열 6개가 `max_episode_steps=600` 이었고,
gymnasium 의 `TimeLimit` 래퍼는 lerobot 의 `TimeLimitProcessorStep`(800 스텝)보다 **바깥에** 있으므로
600 스텝 = **30초에서 먼저 truncate** 되었다.

사용자가 실제로 녹화한 `data/place_demo_01` 이 증거다 — 4 에피소드 전부 정확히 **600 프레임(30초)**.

**수정.** PandaPlaceCube 계열 6개 등록을 `max_episode_steps=2000` (100초 @20fps) 으로 올렸다.
이 값은 상한일 뿐이고 실제 길이는 `control_time_s × fps` 가 정하므로 넉넉히 두는 편이 맞다.
`docs/patches/gym-hil/0002-panda-place-cube-env.patch` 재생성(635줄), `git apply --check --reverse` 로 검증.

**검증.** 녹화를 다시 돌려 에피소드 2개 측정 → **799 프레임 = 40초** (터미널에도 `(실패, 보상 0.0, 40초)`).
수정 전 600 프레임 → 수정 후 799 프레임.

**교훈.** 에피소드 길이를 바꿀 때 손봐야 할 곳이 두 군데다:
`configs/*.json` 의 `control_time_s` **그리고** `gym_hil/__init__.py` 의 `max_episode_steps`.
후자가 더 작으면 전자는 무시된다.

### 7.26 그리퍼 키가 붙잡히면 조작이 죽던 문제 (실제 녹화에서 발견)

사용자가 `data/place_demo_01` 10개를 녹화했는데 **뒤쪽 3개에서 `g` 가 전혀 먹지 않았다.**
데이터가 원인을 그대로 보여줬다:

| ep | 프레임 | 결과 | `g`(닫기) 프레임 | `Shift`(열기) 프레임 | 그리퍼 ctrl 최대 |
|---|---|---|---|---|---|
| 0~5 | 398~668 | **성공** | 2~25 | 2~6 | 255 |
| 6 | 799 | 실패(시간초과) | 33 | 159 | 255 |
| 7 | 474 | 실패 | **0** | 176 | **0** |
| 8 | 719 | 실패 | **0** | 563 | **0** |
| 9 | 717 | 실패 | **0** | 549 | **0** |

7~9번은 `Shift` 가 전체 프레임의 70~78% 동안 눌린 것으로 기록됐다. 사람이 그렇게 누를 리 없다 —
**pynput/X 에서 keyup 이 유실되어 `Shift` 가 눌린 채로 남았다.**

**버그 1 — 동시에 눌리면 아무 일도 안 한다.** `InputController.gripper_command()` 는
`open_gripper_command == close_gripper_command` 일 때 `"no-op"` 을 돌려준다.
`Shift` 가 붙잡힌 상태에서 `g` 를 누르면 둘 다 True → **no-op → 그리퍼 완전 먹통.**

**버그 2 — 에피소드가 바뀌어도 안 풀린다.** `KeyboardController.reset()` 은 `key_states` 만 지우고
`open_gripper_command`/`close_gripper_command` 는 건드리지 않는다. 그래서 한 번 붙잡히면
**다음 에피소드까지 그대로 끌고 간다.** 7·8·9 가 연달아 망가진 이유다.

**수정** (`docs/patches/gym-hil/0003-*.patch` 재생성, 323줄, 역적용 검증):
1. 그리퍼 명령을 세울 때마다 순번(`_grip_seq`)을 남기고, 둘 다 눌린 상태면 **나중에 누른 쪽**을 따른다.
   → 붙잡힌 키가 있어도 다시 누르면 빠져나올 수 있다.
2. `reset()` 이 그리퍼 명령도 초기화한다. → 망가짐이 다음 에피소드로 번지지 않는다.

**검증.** `Shift` 를 붙잡힌 상태로 만들고 `g` → `open`(닫힘) 정상 동작, `g` 를 떼면 `close` 로 복귀,
`reset()` 후 `no-op`. 반대 방향(`g` 붙잡힘 + `Shift`)도 확인.

### 7.27 `merge_demos.py --success-only` — 실패 시연 걸러내기

시연 50개 중 7개가 실패(보상 0)였다. 실패가 오프라인 버퍼에 들어가면 매 배치의 절반이
그쪽에서 뽑히므로 정책이 실패를 따라 배운다. 병합 단계에서 걸러내는 옵션을 넣었다.

```
python scripts/merge_demos.py --glob 'data/place_demo_*' \
  --out data/panda_place_cube_demo --out_repo_id local/panda_place_cube_demo --success-only
```

동작: 평소처럼 `aggregate_datasets` 로 합친 뒤, 에피소드 **마지막 5프레임의 최대 보상**이
`--threshold`(기본 0.5)를 넘는 것만 새 데이터셋으로 다시 쓰고 원본과 바꿔치기한다.

**구현 중 걸린 것 둘.**
1. `LeRobotDataset[i]` 가 돌려주는 프레임을 그대로 `add_frame` 에 넘기면 거부당한다 —
   `timestamp` 는 데이터셋이 매기는 값이라 빼야 하고, 스칼라 특징(`next.reward`, `next.done`,
   `complementary_info.discrete_penalty`)은 0차원으로 오는데 `(1,)` 을 기대한다. 특징 명세의
   `shape` 로 reshape 하도록 했다.
2. `save_episode()` 만으로는 `meta/episodes/*.parquet` 이 기록되지 않아 다시 읽을 때
   `Provided directory does not contain any parquet file` 로 실패한다. **`finalize()` 가 필요하다.**
   (녹화 경로가 finalize 없이도 되는 것은 프로세스 종료 때 쓰기 때문으로 보인다.)

**검증.** `place_demo_01`(성공 6/실패 4) 로 시험 → 6 에피소드 2,993 프레임, 다시 로딩 성공,
전부 보상 +1.0, 특징 shape 정상(이미지 (3,128,128) · state (18,) · action (4,)).
시험은 `data/_t_so` 에만 했고 실제 배치는 건드리지 않았다.

**녹화 현황 (5배치 50개).**

| 배치 | 에피 | 성공 | 실패 |
|---|---:|---:|---:|
| 01 | 10 | 6 | 4 |
| 02 | 10 | 9 | 1 |
| 03 | 10 | 9 | 1 |
| 04 | 10 | 10 | 0 |
| 05 | 10 | 9 | 1 |
| **합계** | **50** | **43** | **7** |

01번이 낮은 것은 §7.26 의 그리퍼 붙잡힘 버그에 걸린 배치이기 때문이다. 수정 후 90~100% 로 올라갔다.

### 7.28 롤아웃 화면 표시(HUD) · 일시정지 · 원하는 순간 저장/종료

**질문 셋에 대한 사실 확인부터.**
1. **개입은 오직 수동이다.** `hil_wrappers.py:236` 의 `is_intervention` 은 전적으로
   `controller.should_intervene()`, 즉 `Space` 토글에서 온다. 자동으로 개입하는 코드는 없다.
2. 학습 중 화면에는 아무 정보도 표시되지 않았다.
3. 일시정지 기능이 없었다. 체크포인트도 `optimization_step % save_freq == 0` 에서만 저장되고
   Ctrl+C 로 끊으면 그 사이 진행이 사라졌다.

**(A) 화면 위 표시 — `view_stream_wrapper.py` 전면 개편**

래퍼는 `InputsControlWrapper` 바깥에 있어 `step()` 의 `info["is_intervention"]` 을 그대로 본다.
통계도 래퍼가 스스로 센다(누적 스텝 = actor 의 상호작용 스텝, 에피소드 수, 성공 수, 최근 20판, 개입 비율).
별도 프로세스 연동이 필요 없다.

- 왼쪽 위 검은 띠 두 줄: `스텝 12,340 · 에피소드 27 (스텝 142)` / `성공 18/26 (69%) · 최근20판 65% · 개입 18% · 경과 0:02`
- **개입 중: 빨간 테두리(7px) + 오른쪽 위 `● 개입 중 (REC)` 배지**
- 일시정지 중: 노란 테두리 + `|| 일시정지`
- `HIL_VIEW_HUD=0` 으로 끌 수 있다. 글꼴은 NanumGothicCoding → DejaVuSansMono 순으로 찾는다.

**(B) 일시정지 — 파일 하나로**

`HIL_PAUSE_FILE`(기본 `/dev/shm/hil_pause`) 이 있으면 `step()` 진입 직전에 0.2초 간격으로 대기한다.
프로세스·gRPC 연결·버퍼가 전부 그대로 살아 있으므로 아무 때나 멈췄다 재개할 수 있다.
멈춰 있던 시간은 HUD 경과 시간에서 뺀다.

검증: 3초에 189스텝 진행 → 파일 생성 후 3초에 **0스텝** → 파일 삭제 후 3초에 128스텝.

**(C) 원하는 순간 저장 / 저장 후 종료 — learner 패치**

`docs/patches/0002-learner-local-metrics-jsonl.patch` (145줄, 역적용 검증):
메인 루프에서 매 스텝 두 파일을 확인한다.
- `<output_dir>/SAVE_NOW` → 저장 조건에 OR 로 붙여 즉시 체크포인트를 남기고 파일을 지운다
- `<output_dir>/STOP_NOW` → 체크포인트를 남기고 `shutdown_event.set()` 후 루프를 빠져나온다

검증: smoke_gpu 재개 후 `save` → **스텝 707**(save_freq 배수가 아님)에 체크포인트 생성, 14→15개.
이어서 `stop` → 15→16개 저장 후 learner 가 23초 만에 **스스로 exit=0** 으로 종료. 두 파일 모두 자동 삭제.

**(D) `scripts/train.sh` 서브커맨드 추가**

```
pause / resume        시뮬레이터 멈춤·재개
save                  지금 체크포인트를 남긴다
stop                  체크포인트를 남기고 learner 정상 종료
new <이름>            output_dir·job_name 만 바꾼 새 설정을 만든다 (기존 학습은 그대로)
```
`new` 로 만든 설정은 `CFG=configs/train_config_<이름>.json bash scripts/train.sh learner` 로 쓴다.
검증: 네 커맨드 모두 실행 확인, `new place_v2` 가 `output_dir=outputs/train/place_v2`, `resume=false`,
`control_time_s=40.0` 을 그대로 물려받는 것 확인.

### 7.29 GPU 사용 현황과 고르는 법

**측정 (smoke 설정, batch 256, utd_ratio 2, resnet10 동결, torch.compile 켬).**

| | 값 |
|---|---|
| learner GPU 메모리 | **4,246 MiB / 81,920 MiB (5.2%)** |
| learner GPU 사용률 | 시작 직후 50% → 정상 진행 중 0~5% |
| actor(추론) GPU 메모리 | ~650 MiB |
| 두 번째 GPU | 놀고 있었다 (3 MiB) |
| 리플레이 버퍼 | `storage_device: cpu` — RAM(384 GB 중 8 GB 사용) |
| 물리 시뮬레이션 | CPU. GPU 는 렌더(EGL)와 신경망에만 쓴다 |

즉 **A100 80GB 두 장이 거의 놀고 있다.** 성능 병목은 GPU 가 아니라
에피소드가 실시간(20 fps)으로 흘러가는 것 자체다.

**`scripts/train.sh` 에 GPU 선택 추가.**
```
HIL_LEARNER_GPU=0 bash scripts/train.sh learner    # learner 를 0번 GPU 에
HIL_ACTOR_GPU=1   bash scripts/train.sh actor      # actor 를 1번 GPU 에
HIL_ACTOR_GPU=cpu bash scripts/train.sh actor      # actor 정책 추론만 CPU 로
```
숫자를 주면 `CUDA_VISIBLE_DEVICES` 를 설정하고, `cpu` 를 주면 `--policy.device=cpu` 를 붙인다.
첫 줄에 `[train] CUDA_VISIBLE_DEVICES=1` 이 찍혀 확인할 수 있다.

검증: `HIL_LEARNER_GPU=1` 로 띄우니 GPU 0 은 0 MiB, GPU 1 에 4,148 MiB. 의도대로 옮겨갔다.

**GPU 부하를 조절하는 실제 손잡이** (모두 `configs/train_config_*.json`):
`batch_size`(256) · `algorithm.utd_ratio`(2, 환경 스텝당 경사 갱신 횟수) ·
`algorithm.use_torch_compile`(true) · `policy.device` · `policy.storage_device`(cpu 로 두면 GPU 메모리 절약).

### 7.30 시연 개수 — 30 은 기본값이 아니라 권장값이다

- 상류 lerobot 의 기본값은 `num_episodes_to_record: int = 5` (`gym_manipulator.py:97`)
- 우리 `env_config_panda_place_record.json` 에는 30 으로 적어 두었지만,
  `record.sh` 가 `--dataset.num_episodes_to_record="$N"` 으로 **덮어쓴다**(배치당 기본 10)
- 즉 "30개"는 HIL-SERL 논문이 비슷한 과제에서 쓰는 규모(20~30)를 따른 내 권장값이고, 코드가 강제하는 값이 아니다

사용자는 5배치 50개(성공 43개)를 모았다. 30보다 많은 것은 손해가 아니다 —
오프라인 버퍼 상한이 10,000 프레임인데 43개 × 평균 ~480 프레임 ≈ 20,600 프레임이라
**상한을 넘는다.** `offline_buffer_capacity` 를 올리거나 일부만 쓰는 편이 낫다(§7.31 참조).

### 7.31 버퍼 상한이 시연보다 작아 learner 가 죽을 상태였다 (시작 전에 발견)

`ReplayBuffer.from_lerobot_dataset` 는 `capacity < len(dataset)` 이면 `ValueError` 를 던진다
(`buffer.py:443`). 성공 시연 43개가 **19,612 프레임**인데 설정은 이랬다:

| 설정 | offline 상한 | 판정 |
|---|---:|---|
| `train_config_panda_place_gpu.json` | 10,000 | **시작 즉시 죽음** |
| `train_config_place_smoke_gpu.json` | 2,000 | **시작 즉시 죽음** |

시연을 30개(≈13,000 프레임) 기준으로 잡아 둔 값인데, 50개를 녹화하면서 넘어섰다.

**수정.**

| 설정 | offline | online |
|---|---|---|
| 본학습 | 10,000 → **40,000** | 20,000 → **40,000** |
| 스모크 | 2,000 → **40,000** | 2,000 → **5,000** |

**메모리 실측.** 버퍼는 이미지를 **float32** 로 들고 있다(uint8 이 아니다):
`observation.images.front/wrist` 각 (N, 3, 128, 128) float32 → **전이 하나당 384 KB**.
- 본학습 80,000 전이 ≈ **29 GB**
- 스모크 45,000 전이 ≈ **16 GB**
`storage_device: cpu` 라 RAM 이고, 서버 RAM 384 GB 중 8 GB 만 쓰고 있어 여유는 충분하다.

**검증.** 새 상한 40,000 으로 `place_demo_01`(5,702 프레임) 적재 → `size=5,702 position=5,702`,
`sample(4)` 정상. 시연 19,612 을 넣고도 **개입 전이 20,388 개까지** 원본 시연을 밀어내지 않는다.

**교훈.** 시연을 더 모으면 `offline_buffer_capacity` 도 같이 올려야 한다.
대략 **에피소드 수 × 평균 프레임 수** 보다 크게 잡는다(40초 에피소드면 에피소드당 최대 800).

### 7.32 학습 속도 병목은 GPU 연산이 아니라 CPU→GPU 데이터 전송이었다

**측정 1 — learner 가 actor 를 못 따라간다.**
smoke 설정(batch 256, utd_ratio 2)으로 30초 측정:

| | 값 |
|---|---|
| learner 최적화 | **초당 3.2 스텝** |
| actor 환경 스텝 | 초당 20 (20 fps 고정) |
| GPU 0 사용률 | 평균 31% · 중앙값 45% · 최대 57% |

GPU 가 절반 이상 놀고 있는데도 learner 가 데이터를 못 따라간다 → **GPU 연산이 병목이 아니다.**

**측정 2 — 범인은 배치 샘플링.**
리플레이 버퍼는 이미지를 float32 로 들고 있어 전이 하나가 384 KB다.
배치 256 = **96 MB**를 매번 CPU 에서 GPU 로 옮긴다. `sample(256)` 100회 평균:

| `storage_device` | 배치당 | 최대 배치/초 |
|---|---:|---:|
| `cpu` (기존) | **41.2 ms** | 24 |
| `cuda` | **1.0 ms** | 971 |

**40배 차이.** 최적화 한 스텝(312 ms)에서 믹서가 온라인·오프라인 양쪽을 뽑고 utd_ratio 만큼 반복하므로
샘플링이 절반 가까이를 먹고 있었다. GPU 유휴 45% 와 맞아떨어진다.

**수정 — 버퍼를 GPU 에 올린다.** `policy.storage_device: cpu → cuda`

메모리 실측(상한 40,000 두 개를 실제로 할당해 확인):
- offline 40,000 → 14.7 GB · online 40,000 → 누적 **29.3 GB**
- A100 79 GB 중 45 GB 가 남는다. 모델·옵티마이저(약 4 GB)를 더해도 여유가 크다

되돌리려면 실행할 때 `--policy.storage_device=cpu` 를 붙인다.

**정리 — "GPU 를 더 쓰면 빨라지나?"에 대한 답.**
1. **데이터 수집은 안 빨라진다.** actor 는 20 fps 실시간으로 묶여 있고 물리는 CPU 다.
2. **학습(최적화)은 빨라진다.** 위 변경으로 샘플링 41 ms → 1 ms.
3. **GPU 두 장을 한 학습에 쓰는 병렬화는 없다.** 두 번째 A100 은 **다른 실험**을 동시에 돌리는 데 쓴다
   (`train.sh new <이름>` + `HIL_LEARNER_GPU=1`).
4. 더 쓰고 싶으면 `algorithm.utd_ratio`(현재 2)를 올린다 — 환경 스텝당 갱신 횟수가 늘어
   **같은 데이터로 더 많이 배운다**(RLPD 계열은 4~20 을 쓴다). 샘플링이 40배 빨라졌으니 이제 여유가 있다.

**부수.** 앞선 SIGINT/STOP_NOW 시험 때 저장 도중 죽인 탓에
`outputs/train/smoke_gpu/dataset_offline` 의 parquet 이 깨져 있었다. 시험용 런이라 폴더째 지웠다.

### 7.33 `steps` 와 `num_workers` 는 RL 경로에서 쓰이지 않는다

`grep -rn "cfg\.steps" third_party/lerobot/src/lerobot/rl/*.py` → **한 건도 없다.**
learner 메인 루프(`learner.py:410`)는 `while True:` 이고 `shutdown_event` 로만 빠져나온다.
`num_workers` 도 마찬가지로 RL 은 데이터로더 대신 리플레이 버퍼에서 직접 뽑으므로 쓰이지 않는다.
둘 다 `TrainPipelineConfig`(지도학습용)에서 물려받은 항목이다.

**실제 종료 조건은 `policy.online_steps`** 다 — actor 의
`for interaction_step in range(cfg.policy.online_steps)` (`actor.py:315`).
100,000 스텝 @ 20 fps = **약 1.4시간**.

**`train.sh status` 를 고쳤다.** 진행률을 `cfg["steps"]`(쓰이지 않는 값, 현재 300,000) 기준으로
보여주고 있어 틀린 퍼센트가 나왔다. `policy.online_steps` 기준으로 바꾸고 막대와 예상 남은 시간을 붙였다.

```
  학습 스텝      : 24,400   (최적화 횟수. 상한 없음 — 멈출 때까지 돈다)
  상호작용 스텝  : 18,720 / 100,000 (18.7%)  [#####.........................]
                   남은 시간 약 1.1 시간 (20 fps 기준)
```
검증: 임시 디렉터리에 가짜 `training_step.json` 만 두고 실행 → 위 형식대로 출력되고,
지표 파일이 없을 때는 성공률·개입률 줄을 **찍지 않는** 것도 확인.

**런북 §07b "설정 파일 읽는 법" 신설.** 실행·기록 / 학습 길이·데이터 혼합 / 속도·GPU 세 표로 나눠
각 항목의 뜻과 올렸을 때의 영향을 정리했다. 목차에도 추가.

### 7.34 출처 정리 · 스텝/배치/혼합 주기 (질문 답변용 근거)

**`scripts/update_dataset_stats.py` 는 내가 만든 것이다.**
- `policy.dataset_stats` **필드 자체는 상류**다. `actor.py:300` · `learner.py:356` 이
  `make_policy(..., dataset_stats=cfg.policy.dataset_stats)` 로 읽어 관측 정규화에 쓴다.
- 다만 상류는 그 값을 **설정 JSON 에 손으로 적어 넣는 것을 전제**한다. 녹화한 데이터셋에서
  자동으로 계산해 써 넣는 스크립트는 lerobot 에도(`src/lerobot/scripts/` 어디에도) 없다.
  가장 가까운 것이 `lerobot_find_joint_limits.py` 인데 실물 로봇의 관절 범위를 찾는 도구라 다르다.
- 그래서 `qpos` 는 데이터 실측 min/max(여유 포함), `qvel` 은 ±100 고정,
  `gripper` 0~255, `tcp` 는 환경의 cartesian bounds 로 채우는 방식을 만들었다.

**스텝은 두 종류다.**

| | 무엇 | 실측 속도 | 세는 곳 |
|---|---|---|---|
| 상호작용(환경) 스텝 | `env.step()` 1회 | 초당 **20** (fps 고정) | actor (`actor.py:315`) |
| 최적화(학습) 스텝 | 가중치 갱신 한 묶음 | 초당 **3.2** | learner |

둘은 독립적으로 돈다. learner 는 actor 를 기다리지 않는다.

**batch 는 "스텝 개수"가 아니라 전이 개수다.** `utd_ratio=2`, `batch_size=256` 일 때
최적화 스텝 하나는 (`sac_algorithm.py:201-218`):
1. `range(utd_ratio - 1)` = 1회: 배치 1개 → 크리틱 + 이산 크리틱 갱신, 타깃 소프트 갱신
2. 마지막 배치 1개 → 크리틱 + 이산 크리틱 + 액터 + 온도 갱신

→ **최적화 스텝당 배치 2개 = 전이 512개.**

**5:5 혼합은 주기가 아니라 매 배치다.** `OnlineOfflineMixer.sample()`
(`rl/data_sources/data_mixer.py`)이 호출될 때마다
`n_online = max(1, int(256 × 0.5)) = 128`, `n_offline = 256 − 128 = 128` 을
두 버퍼에서 따로 뽑아 `concatenate_batch_transitions` 로 이어 붙인다.
"몇 스텝마다 한 번 섞는다"는 개념은 없다. 실측 기준 초당 약 6.4회 섞인다.

**actor/learner 분리는 상류 그대로다.**
- `third_party/lerobot` = `github.com/huggingface/lerobot` v0.6.1 (`7e241bd`), Apache-2.0
- `actor.py` 의 docstring: *"Actor server runner for distributed HILSerl robot policy training"*
- 분리 구조의 출처는 SERL/HIL-SERL(UC Berkeley) 계열이다. 실물 로봇에서 학습 서버와
  로봇 PC 를 떼어 놓을 수 있게 gRPC 로 통신한다. 우리는 둘 다 localhost:50051 로 붙인다.
- `third_party/gym-hil` = `github.com/huggingface/gym-hil` (`f89b3e5`)
- 내가 만든 것은 **태스크(PandaPlaceCube)와 운영 도구**(스크립트·대시보드·화면 스트림)이지
  학습 알고리즘이나 actor/learner 구조가 아니다.

### 7.35 HUD 의 에피소드 카운터가 1 에 멈춰 있던 버그 (실제 학습 중 발견)

사용자가 화면에서 `전체 8,290 스텝 · 에피소드 1` 을 보고 "40초면 끝나는 것 아니었나" 하고 물었다.
**환경은 정상이었다** — actor 로그의 에피소드 종료가 1597, 2396, 3195, 3994, ... 정확히 **799 스텝 간격**
(= 40초 @20fps)이다.

**원인.** 시간 초과 판정은 gym 환경이 아니라 그 **바깥**의 lerobot 파이프라인
(`TimeLimitProcessorStep`, `gym_manipulator.py:404`)이 한다. 그래서 `env.step()` 의 반환값
`truncated` 는 계속 False 이고, 그것만 보던 HUD 는 시간 초과로 끝난 에피소드를 **한 번도 세지 못했다.**
(에피소드 내 스텝 `_ep_step` 은 `reset()` 에서 0 으로 돌아가므로 정상이었다.)

**수정.** 에피소드 집계를 `step()` 이 아니라 **`reset()`** 에서 한다 — 리셋이 곧 에피소드 종료이고,
시간 초과가 어디서 판정되든 상관없다. 성공 여부는 그 판의 최대 보상(`_last_reward`)으로 본다.
표시 문구도 덜 헷갈리게 바꿨다: `전체 8,290 스텝 · 11번째 에피소드 300 스텝째`.

**검증.** 환경을 끝내지 않은 채 바깥에서 `reset()` 만 호출하는(=actor 와 같은) 상황을 만들어
3판 중 1판 성공 → `_ep=3`, `_succ=1`, `_recent=[0,1,0]`, `_ep_step=0` 확인.

### 7.36 `train.sh status` 두 가지 수정

1. **진행률이 항상 0% 로 나왔다.** 체크포인트의 `training_step.json` 은 저장 시점에
   interaction 메시지가 없으면 `interaction_step: 0` 으로 남는다(상류 특성).
   `learner_metrics.jsonl` 의 최신 `Interaction step` 을 우선 쓰도록 했다 → 8.8% 로 정상 표시.
2. **버퍼 크기가 틀렸다.** 디스크의 `dataset/`·`dataset_offline/` 은 체크포인트 시점 스냅샷이라
   저장이 진행 중이면 덜 쓰인 값이 보인다(시연 버퍼가 20,279 인데 5,605 로 표시됐다).
   지표 파일의 `replay_buffer_size` / `offline_replay_buffer_size` 를 쓰도록 바꿨다.

**확인한 실제 학습 상태** (상호작용 8,788 스텝 시점):
`온라인 8,789` · `시연+개입 20,279` (시연 19,612 + 개입 667) · 에피소드 11개 · 개입률 8%.
오프라인 버퍼가 정상적으로 적재되어 쓰이고 있다(`make_dataset offline buffer` →
`Convert to a offline replay buffer`).

### 7.37 `scripts/report.py` — 실험 하나를 HTML 리포트 한 장으로

`learner_metrics.jsonl` 을 읽어 `<run>/report.html` 을 만든다. 외부 라이브러리 없이 SVG 를
직접 그리므로 파일 하나만 열면 되고 인터넷도 필요 없다. 학습 중에 다시 돌리면 갱신된다.

```
python scripts/report.py                          # 기본 런
python scripts/report.py --run outputs/train/place_v2 --window 20
```

담긴 것: 타일 6개(에피소드·성공·최근 성공률·최근 개입률·최적화 스텝·상호작용 스텝) +
그래프 3개 + 최근 200판 표.

| 그래프 | 축 | 계열 |
|---|---|---|
| 성공률과 개입률 | 상호작용 스텝 × % | 성공률 · 개입률 (둘 다 %, 한 축) |
| 버퍼 크기 | 최적화 스텝 × 전이 수 | 온라인 · 시연+개입 |
| 크리틱 손실 | 최적화 스텝 × 손실 | 단일 계열(범례 없음) |

**dataviz 스킬 규칙을 따랐다.** 색은 검증된 기본 팔레트의 1~3번 슬롯을 순서 그대로 사용
(`#2a78d6`/`#eb6834`/`#1baf7a`, 다크는 `#3987e5`/`#d95926`/`#199e70`).
이중 축을 쓰지 않았고(단위가 다른 손실은 별도 차트로 분리), 2계열 이상에는 범례를 항상 두고
마지막 점에만 직접 라벨을 달았으며, 선 그래프에는 십자선+툴팁을 붙였고, 표 보기를 넣었고,
다크 모드는 미디어쿼리와 `data-theme` 양쪽 스코프에 따로 선언했다.
**검증기는 `node` 가 설치돼 있지 않아 돌리지 못했다** — 대신 스킬 문서가 이미 검증됐다고
명시한 팔레트 값을 그대로 썼고 새 색을 만들지 않았다.

검증: 실제 런으로 생성 → 3개 차트, 선 5개, 빈 차트 0개,
c1 86점씩 · c2 401점씩 · c3 401점, 타일 값이 `train.sh status` 와 일치.

### 7.38 `train.sh actor` 가 `--resume=true` 때문에 죽던 버그

사용자가 actor 를 재시작하니 이렇게 죽었다:
```
FileNotFoundError: No such file or directory: configs/model.safetensors
```

**원인은 내가 넣은 `--resume=true` 다.** actor 출력 폴더가 이미 있으면
`TrainPipelineConfig.validate()` 의 `FileExistsError` 를 피하려고 붙였는데,
lerobot 의 파서는 `resume=true` 일 때 `--config_path` 를 **"사전학습 정책 폴더"** 로 해석한다.
그래서 `configs/` 밑에서 `model.safetensors` 를 찾다가 실패한다.
learner 는 `--config_path` 가 체크포인트 안의 `train_config.json` 을 가리키므로 문제가 없었고,
actor 경로만 이 조합에 걸렸다. **내가 그 경로를 실제로 실행해 보지 않고 넣은 탓이다.**

**수정.** actor 에서 `--resume` 을 완전히 뺐다. actor 는 가중치를 learner 에게서 gRPC 로 받으므로
resume 이 필요 없다. 출력 폴더 충돌은 로그를 `<run>_actor_prevlogs/` 로 옮기고 폴더를 비워 해결한다.
에피소드 기록의 원본은 learner 쪽 `learner_metrics.jsonl` 이라 영향이 없다(1,492줄 그대로).

**검증.** 수정 후 실행 → actor 기동, MuJoCo 창 생성(`MuJoCo : PlaceCubeScene`),
`/dev/shm/hil_view.jpg` 가 16초 만에 생기고 계속 갱신됨을 확인.
8088 화면이 안 나오던 것도 actor 가 죽어 있어서였고 같이 해결됐다.

### 7.39 `scripts/stop_after.py` — 에피소드 N 개로 실험을 끊기

lerobot 에는 "몇 에피소드까지" 라는 종료 조건이 없다. actor 는 `online_steps`(환경 스텝 수)로만
끝나는데, 에피소드 길이가 성공/실패에 따라 달라지므로 **실험끼리 조건을 맞추려면 에피소드 수로
끊는 편이 낫다**(성공하면 800스텝보다 짧게 끝난다).

```
python scripts/stop_after.py --episodes 100
python scripts/stop_after.py --episodes 100 --run outputs/train/place_v2
python scripts/stop_after.py --episodes 100 --no-stop     # 세기만
```

`learner_metrics.jsonl` 의 `kind: "interaction"` 레코드 수를 센다(에피소드당 1개).
도달하면 ① `report.py` 실행 → ② `<run>/STOP_NOW` 생성(learner 가 저장 후 종료) →
③ actor 에 SIGINT. 진행 중에는 막대와 성공률·개입률을 실시간으로 찍는다.

**검증.** 실제로 도는 런에 붙여 97 → 98 에피소드로 올라가는 것,
`find_actor_pid()` 가 실행 중인 actor PID(3975055)를 정확히 찾는 것 확인.

**실험 비교 절차** (런북 §07c 로 정리):
```
# 실험 A — 개입 없이
bash scripts/train.sh learner / actor --window
python scripts/stop_after.py --episodes 100

# 실험 B — 개입하면서
bash scripts/train.sh new place_assisted
CFG=configs/train_config_place_assisted.json bash scripts/train.sh learner / actor --window
CFG=... python scripts/stop_after.py --episodes 100 --run outputs/train/place_assisted

# 비교: 두 report.html 을 나란히 연다
```

`online_steps` 는 상한 역할만 하면 되므로 넉넉히 둔다
(100 에피소드 × 최대 800 스텝 = 80,000 < 현재 100,000).

### 7.40 리포트의 x축을 에피소드 번호로 (actor 재시작 시 스텝이 되감기던 문제)

`Interaction step` 은 actor 프로세스가 `range(cfg.policy.online_steps)` 로 세기 때문에
**actor 를 재시작하면 0 부터 다시 센다.** 실제 기록에서 확인: `71,869 → 798` 로 한 번 되감겼다.
그걸 x축으로 쓰던 성공률/개입률 그래프가 뒤로 감기고 선이 겹쳤다.

**수정.**
- 그래프 x축을 **에피소드 번호**(1,2,3…)로 바꿨다. 기록이 쌓이는 순서라 항상 단조 증가하고,
  "100 에피소드" 라는 실험 단위와도 맞는다. 스텝은 표와 툴팁에만 남겼다.
- 상호작용 스텝 타일은 되감긴 지점마다 이전 최대값을 더해 **누적**으로 계산한다.
  재시작 횟수도 같이 표시한다: `81,456` / `로봇이 움직인 스텝 (누적) · actor 재시작 1회`.

검증: x축 `[1,2,3] … [102,103,104]` 단조 증가 확인. 타일 81,456 = 71,869 + 9,587.

### 7.41 새 실험이 시연 데이터를 읽는 곳

질문: "새 실험도 `train/panda_place_gpu/dataset_offline` 를 읽나?"
**아니다.** `initialize_offline_replay_buffer` (`learner.py:903-911`):

```python
if not cfg.resume:
    offline_dataset = make_dataset(cfg)            # -> cfg.dataset.root
else:
    dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")
```

- **새 실험(resume=false)** → `cfg.dataset.root` = `data/panda_place_cube_demo` (43판 19,612프레임)에서 읽는다
- **재개(resume=true)** → 그 런의 `<output_dir>/dataset_offline` 에서 읽는다(개입까지 포함된 상태)

즉 `data/panda_place_cube_demo` 가 **원본이자 고정값**이고, 각 런의 `dataset_offline/` 은
그 런에서 개입이 더해진 사본이다. 실험끼리 섞이지 않는다.

### 7.42 정정 — 재개하면 온라인 버퍼가 전부 복원되지는 않는다

§7.23 에서 "온라인 버퍼도 전부 복원된다"고 적었다. **틀렸다.** 실측:

| 시점 | 온라인 버퍼 |
|---|---|
| 죽기 전 (메모리) | **40,000** (상한까지 가득) |
| 저장된 스냅샷 `dataset/meta/info.json` | **17 에피소드 / 12,834 프레임** |
| 재개 직후 | **12,834** → 이후 새 경험으로 다시 채워짐 |

약 68% 가 사라진다. `ReplayBuffer.to_lerobot_dataset` 은 버퍼를 순회하며
`dones[i] or truncateds[i]` 인 지점에서만 `save_episode()` 를 부르는데,
40,000 전이(에피소드 50판 분량) 중 17판 분만 기록됐다. **정확한 원인은 아직 규명하지 못했다** —
에피소드 경계 플래그가 기대만큼 서 있지 않은 것으로 보인다.

**실용적 결론.**
- 정책 가중치·옵티마이저·학습 스텝·시연 버퍼는 온전히 복원된다(체크포인트 번호가 이어지는 것으로 확인).
- **온라인 버퍼는 상당 부분 잃는다.** 그러니 불필요하게 actor/learner 를 재시작하지 않는 편이 좋다.
- 잃어도 치명적이지는 않다. actor 가 계속 새 경험을 넣으므로 다시 찬다. 다만 그 동안 학습이 한 번 흔들린다.

런북 §08 의 해당 서술을 고쳐야 한다.

### 7.43 `train.sh status` 의 상호작용 스텝도 누적으로

`report.py` 와 같은 방식(되감긴 지점마다 직전 최대값을 더함)을 적용했다.
`상호작용 스텝 : 83,336 / 100,000 (83.3%)` · `actor 재시작 2회` 처럼 보인다.

### 7.44 정리한 것 — 멈춰 있던 중복 actor

`State: T (stopped)` 인 actor(PID 3975055)가 남아 있었다. 터미널에서 `Ctrl+Z` 로 멈춘 것으로 보인다.
새 actor(3984234)가 따로 돌고 있었으므로, 저게 `SIGCONT` 를 받으면 **같은 화면 파일과 같은 X 디스플레이를
두고 다투게 된다.** `kill -CONT` 후 종료시켰다.

점검 명령: `for p in $(pgrep -f lerobot); do echo "$p $(awk '/^State:/{print $2}' /proc/$p/status)"; done`

### 7.45 개입의 효과 — 한 런 안에서 측정된 전후 비교

같은 런(`panda_place_gpu`) 안에서 101판째부터 사람이 개입하기 시작했다. 10판 단위 집계:

| 구간 | 성공 | 성공률 | 개입률 |
|---|---|---|---|
| 1–100 (개입 없음) | 1/100 | **1%** | ~1% |
| 101–110 | 3/10 | 30% | 42% |
| 111–120 | 7/10 | **70%** | 56% |
| 121–130 | 7/10 | **70%** | 59% |
| 131–140 | 4/10 | 40% | **27%** |

**개입 없이 100판 = 1%. 개입 40판 = 52%.**
131–140 구간이 중요하다 — **개입률이 59%→27% 로 떨어졌는데 성공률 40% 를 유지**했다.
정책이 스스로 하기 시작했다는 신호다.

이것은 두 런을 따로 돌린 비교보다 오히려 강하다. 정책·시드·시연 데이터가 전부 동일하고
개입 도입 시점만 다르기 때문이다.

### 7.46 에피소드 길이를 20초로 줄이면 안 되는 이유 (실측)

성공한 21판의 길이(현재 actor 세션):

| | 스텝 | 초 |
|---|---|---|
| 최소 | 231 | 12 |
| **중앙** | **473** | **24** |
| 최대 | 793 | 40 |

상한을 줄였을 때 살아남는 성공:

| 상한 | 살아남는 성공 |
|---|---|
| 10초 | 0/21 (0%) |
| 15초 | 4/21 (19%) |
| **20초** | **8/21 (38%)** |
| 30초 | 17/21 (80%) |
| 40초 (현재) | 21/21 (100%) |

**20초로 줄이면 성공의 62% 가 시간 초과로 잘려 실패로 기록된다.** 지금은 해롭다.
30초는 80% 를 유지하면서 시간당 판 수를 33% 늘린다 — 정책이 더 빨라진 뒤에 고려할 만하다.
(설정 변경은 재개에 반영되지 않으므로 새 실험에서만 가능하다.)

### 7.47 병렬 실행은 이 파이프라인에 없다

- `grep -rn "VectorEnv|num_envs|make_vec" lerobot/rl/ gym_hil/` → **0건**.
  actor 는 `for interaction_step in range(online_steps)` 로 **환경 하나를 한 스텝씩** 돈다.
- gym-hil 에 **MJX(GPU 물리) 흔적도 없다.** MuJoCo CPU 물리다.
- Isaac Lab 식 수천 개 병렬은 GPU 물리 시뮬레이터가 전제다. 구조가 다르다.
- 더 근본적으로 **HIL-SERL 은 사람이 루프 안에 있어야 한다.** 4096개 환경에 동시에 개입할 수 없다.

실제로 가능한 가속: `utd_ratio` 올리기(GPU 30% 만 쓰는 중) · 에피소드 길이 줄이기(위 제약 안에서) ·
개입 계속하기. 서로 다른 실험 두 개를 GPU 한 장씩 동시에 돌리는 것은 가능하다.

### 7.48 `train.sh status` — 누적과 현재 세션을 분리

`상호작용 스텝 102,498 / 100,000 (102.5%)` 처럼 100% 를 넘는데 끝나지 않아 혼란스러웠다.
종료 조건은 **지금 도는 actor** 가 `online_steps` 를 채우는 것이고, 재시작하면 그 카운터는 0 부터다.
두 값을 분리해 표시하도록 고쳤다:

```
  상호작용 스텝  : 102,498 누적  (actor 재시작 2회)
  현재 actor     : 21,042 / 100,000 (21.0%)  [######........................]
                   이 actor 가 끝나기까지 약 1.1 시간
```

### 7.49 런북 v20 — 순차 실행 문서로 정리

사용자가 "데이터 만드는 것부터 일련의 작업을 순차적으로" 볼 수 있는 문서를 요청했다.
기존 `docs/place-cube-runbook.html` 을 갱신했다(새로 만들지 않음).

**§00 "명령만 모아서 — 처음부터 끝까지"** 를 8개 블록으로 다시 썼다:
준비 → 연습 → 녹화 → 병합·통계 → 스모크 → 본 학습 → 학습 중 조작 → 기록·평가 → 복구/새 실험.
각 블록에 어느 tmux 칸에서 치는지 주석을 달았고, 마지막에 중복·멈춘 프로세스 점검 명령을 넣었다.

**새로 반영한 실측 내용:**
- §07 개입 효과 표 (1–100판 1% → 101–140판 52%, 개입률 59%→27% 인데 성공률 유지)
- §07b 에피소드 길이 표 (20초로 줄이면 성공의 62% 가 잘린다 — 중앙값 24초)
- §07b "더 빠르게 학습시키려면" 표 (병렬 불가 이유 · utd_ratio 권장 · 개입이 가장 효과적)
- §09 를 "평가 · 기록" 으로 바꾸고 `report.py` 를 맨 앞에 배치
- `train.sh status` 출력 예시를 현재 형식(누적/현재 actor 분리)으로 교체
- 파일 목록에 `report.py` · `stop_after.py` · `check_dataset.py` · `run_smoke_test.sh` 추가

검증: 14개 절 전부 유지, 태그 균형 확인, 주요 명령 13종이 모두 문서에 존재하는지 확인.

### 7.50 "이어서" 와 "새로" 를 화면에서 구분할 수 있게

**문제.** actor 프로세스는 자기 스텝·에피소드를 0 부터 센다. 그래서 이어서 하는 실험인데도
HUD 에 `1번째 에피소드` 가 떠서 새로 시작한 것처럼 보였다. 대시보드 그래프도 x 축이
`Interaction step` 이라 actor 재시작마다 뒤로 감겼다.

**수정 1 — HUD 가 런의 기록을 이어받는다.**
`view_stream_wrapper.py` 에 `_load_history(run_dir)` 를 넣었다.
`HIL_RUN_DIR` 이 주어지면 그 런의 `logs/learner_metrics.jsonl` 을 읽어
에피소드 수·성공 수·최근 20판·누적 스텝을 이어받는다(되감긴 지점은 더해서 누적).
`train.sh actor` 가 `export HIL_RUN_DIR="$OUT"` 로 알려 준다.
연습(`teleop.sh`)·녹화(`record.sh`)는 이 변수가 없으므로 0 부터 센다 — 의도한 동작이다.

화면 문구에도 어느 쪽인지 찍는다:
```
전체 199,950 스텝 · 267번째 에피소드 12 스텝째 (이어서, 266판부터)
전체 0 스텝 · 1번째 에피소드 0 스텝째 (새 실험)
```

**수정 2 — `train.sh actor` 가 시작할 때 알린다.**
```
[train] === 이어서 하는 실험 === 지금까지 266판. 화면 카운터도 이어서 센다.
[train] === 새 실험 === 0판부터 시작한다.
```

**수정 3 — 대시보드 그래프 x 축도 에피소드 번호로.**
`progress_dashboard.py` 의 `collect()` 가 `Interaction step` 대신 순번을 쓴다.
표 머리글과 축 라벨도 "에피소드"로 바꿨다. (`report.py` 는 §7.40 에서 이미 고쳤다.)

**검증.** `_load_history('outputs/train/panda_place_gpu')` → `266판 · 성공 22 · 누적 199,950 스텝`,
없는 폴더 → `0판`. 대시보드 재시작 후 `/data.json` 의 성공률 x 축이 `1.0 → 266.0` 으로 단조 증가.

### 7.51 `train.sh new` 는 설정만 만든다 (출력 폴더는 아니다)

사용자가 `configs/train_config_02_place_assisted.json` 을 만들었는데
`outputs/train/02_place_assisted` 가 없어서 의아해했다. **정상이다.**

- `train.sh new <이름>` → `configs/train_config_<이름>.json` 만 만든다
  (`output_dir` 과 `job_name` 만 바꾼 사본)
- `outputs/train/<이름>/` 은 **그 설정으로 learner 를 실제로 돌릴 때** 생긴다

즉 설정을 만든 뒤 반드시 `CFG=` 를 붙여 실행해야 한다:
```
CFG=configs/train_config_02_place_assisted.json bash scripts/train.sh learner
```

### 7.52 정정 — 재개 시 온라인 버퍼 손실은 68% 가 아니라 약 24%

§7.42 에서 "재개하면 온라인 버퍼의 68% 를 잃는다"고 적었다. **측정이 틀렸다.**

그때 비교한 스냅샷(12,834 프레임)은 **직전 재개 직후에 찍힌 것**이었다. 그 시점의 버퍼는
아직 다시 차는 중이었는데, 그것을 *재개 전*의 40,000 과 비교했다. 사과할 대상이 아니라
같은 시점의 값이 아니었다.

**다시 측정 (버퍼가 상한 40,000 에 도달한 상태에서 찍힌 스냅샷):**

| | 값 |
|---|---|
| 메모리의 온라인 버퍼 | 40,000 (상한) |
| 저장된 스냅샷 `dataset/` | **41판 / 30,271 프레임** |
| 손실 | **9,729 (약 24%)** |

원인은 `to_lerobot_dataset` 이 `done|truncated` 지점에서만 `save_episode()` 를 부르고
**마지막 경계 이후 프레임을 버리기** 때문으로 보인다. 다만 그 설명대로면 손실이 한 판(800프레임)
이하여야 하는데 9,729 가 사라졌다. **정확한 원인은 여전히 규명하지 못했다.**
문서에는 "일부(약 1/4) 잃는다" 정도로만 적는다.

결론 자체는 유지: 정책·옵티마이저·학습스텝·시연버퍼는 온전하고, 온라인 버퍼만 일부 잃으므로
**필요 없이 재시작하지 않는 편이 좋다.**

### 7.53 포트 포워딩은 SSH 세션에 딸려 있다

사용자가 장소를 옮겨 재접속한 뒤 8088 이 안 보인다고 했다. 서버 쪽은 정상이었다
(`127.0.0.1:8088 LISTEN`, `HTTP 200`, 대시보드 PID 살아 있음). 문제는 터널이었다:
`ss -tnp | grep -c "ESTAB.*:8088"` → **0**.

tmux 안의 프로세스는 SSH 가 끊겨도 살아남지만, **VS Code 의 포트 포워딩은 SSH 세션에 딸린 것**이라
재접속하면 다시 잡아 줘야 한다. 런북 §10 에 추가할 것.

### 7.54 경과 시간을 런 단위로 누적

**문제.** HUD 의 경과 시간이 actor 프로세스 시작 시각 기준이라, 267번째 에피소드인데
`경과 11분` 처럼 말이 안 되는 값이 떴다.

**수정.** `<run>/elapsed.json` 에 **실제로 돌아간 초**를 쌓는다.
- `_load_elapsed()` 로 시작할 때 이어받고, `_elapsed()` 가 `이어받은 값 + 이번 프로세스 경과` 를 돌려준다
- `step()` 에서 10초마다, `close()` 에서 한 번 더 원자적으로 저장(`os.replace`)
- 일시정지 중에는 `_t0` 를 밀어 그 시간을 빼는 기존 동작이 그대로 유지된다
- `HIL_RUN_DIR` 이 없을 때(연습·녹화)는 파일을 만들지 않고 0 부터 센다

로그에서 공백을 추정하는 방법보다 정확하다. (참고로 추정해 보면 벽시계 5.46시간 중
실제 학습은 3.12시간이었다 — 그 값을 기존 런의 초기값으로 넣어 뒀다.)

**표시 추가.**
- `train.sh status` → `학습 시간 : 3시간 7분  (실제로 돌아간 시간, 누적)`
- `report.py` → `학습 시간` 타일

**검증.** `elapsed.json` 에 3,661초를 넣고 시작 → HUD `1:01:02`, 13초 돌린 뒤 `1:01:15`,
종료 후 파일에 3,675초. 이어받기·누적·저장 모두 정상.

### 7.55 경과 시간 보정본을 돌고 있는 actor 가 덮어쓰던 버그

§7.54 로 `elapsed.json` 을 넣은 뒤, 이미 돌고 있던 실험의 시간을 로그에서 역산해 채워 넣었더니
**돌고 있는 actor 가 10초 뒤에 자기 값(작은 값)으로 덮어썼다.** 6,507 → 1,098 로 줄어들었다.

원인: actor 는 시작할 때 파일을 한 번만 읽어 `_elapsed_base` 로 삼고, 이후 그 값 + 자기 경과를 쓴다.
시작 이후에 파일이 더 큰 값으로 바뀌어도 모른다.

**수정.** `_save_elapsed()` 가 저장 직전에 파일을 다시 읽어 `max()` 를 취한다.
경과 시간은 줄어들 수 없는 값이므로 max 가 항상 옳다. 보정도, 다른 프로세스가 앞서 쓴 값도 안 지워진다.

검증: 5000 저장 → 5000 · 100 저장 → 5000 유지 · 9000 저장 → 9000.

### 7.56 프로세스 점검 결과 — 중복 실험은 없었다

사용자가 "다른 프로세스가 있어서 그런가" 물어 전수 점검했다:

| PID | 시작 | 무엇 |
|---|---|---|
| 4053575 | 07:46 | learner — `03_exp_b_assist` (한 번도 안 죽음) |
| 509040 | 09:43 | actor — `03_exp_b_assist` (재시작됨) |
| 502560 | 09:25 | 대시보드 8088 |
| 505843 | 09:35 | 대시보드 8899 (내가 띄운 여분) |

**실험은 하나뿐이었다.** `new` 를 두 번 한 것도, 설정이 덮어써진 것도 아니다
(`configs/train_config_03_exp_b_assist.json` 수정 시각 07:45, 그 뒤로 안 바뀜).

화면 시간이 0 에 가까웠던 진짜 이유는 **타임라인**이다:
```
07:45  설정 생성 (new 한 번)
07:51  실험 첫 에피소드
09:39  <- elapsed.json 기능을 넣은 시각
09:43  actor 재시작 -> 그때 파일에 55초만 있었음
```
그 전 1시간 48분은 기록할 코드 자체가 없었다. 로그에서 역산해 채워 넣었다(2.04시간).
**돌고 있는 actor 의 HUD 에는 다음 시작 때부터 반영된다.**

### 7.57 learner 와 actor 가 서로 다른 실험 설정으로 돌던 문제 + 안전장치

**발견.** 사용자가 "대시보드에 22/295 인데 밤새 돌았으니 더 나왔어야 하지 않나" 물어 점검했더니:

| PID | 무엇 | 설정 |
|---|---|---|
| 4053575 | learner | `03_exp_b_assist` (07:46부터) |
| 546572 | actor | **`panda_place_gpu`** ← 다름 |
| 550440 | 대시보드 | 기본값 = `panda_place_gpu` ← 다름 |

actor 를 `CFG=` 없이 띄워 기본 설정이 쓰였다. gRPC 는 `localhost:50051` 로 붙으므로
**학습 자체는 정상 진행됐지만**, actor 쪽 산출물(HUD 카운터·`elapsed.json`·actor 로그)이
엉뚱한 폴더에 쌓이고 대시보드도 멈춘 런을 보고 있었다.

| 런 | 에피소드 | 마지막 기록 |
|---|---|---|
| `panda_place_gpu` | 266 / 성공 22 | 05:56 (멈춤) |
| `03_exp_b_assist` | **386 / 성공 30** | 진행 중 |

**안전장치 1 — `train.sh actor` 가 learner 설정과 대조한다.**
`running_learner_cfg()` 로 돌고 있는 learner 의 `--config_path` 를 읽어 `$CFG` 와 비교하고,
다르면 무엇이 어긋났는지 보여주고 **중단한다**(`HIL_FORCE=1` 로 강행 가능).

구현 주의: `pgrep -f` 는 **자기 명령줄도 잡는다**(실제로 `[^` 가 출력되는 오작동이 났다).
`/proc/<pid>/comm` 이 `python*` 인 것만 보고, 인자를 `tr '\0' '\n'` 으로 줄 단위로 쪼갠 뒤
`sed -n 's/^--config_path=//p'` 로 뽑도록 고쳤다.

**안전장치 2 — `dashboard.sh` 가 돌고 있는 learner 를 따라간다.**
`--run` 을 직접 주지 않으면 learner 의 설정에서 `output_dir` 을 읽어 자동으로 붙인다.
```
[대시보드] 돌고 있는 learner 를 따라간다: outputs/train/03_exp_b_assist
```

검증: 설정이 어긋난 상태에서 `train.sh actor` → 경고 후 중단 확인.
대시보드 재시작 → `보는 런: outputs/train/03_exp_b_assist  에피소드 386  성공 30`.

### 7.58 순수 롤아웃 성공은 0건이었다

`03_exp_b_assist` 386판 중 성공 30판 — **30판 전부 개입이 있었다. 개입률 0% 인 성공은 0건.**
168판 이후 160판 넘게 연속 0%. 앞 실험 `panda_place_gpu` 는 266판 중 순수 성공 1건(49번째)뿐.

즉 **두 실험 모두, 개입 없이 정책 혼자 성공한 사례가 사실상 없다.**

### 7.59 경과 시간 자가복구 · 안내 문구에 명령 포함 · 문서 갱신

**경과 시간이 양쪽 다 틀려 있었다.**

| 런 | `elapsed.json` | 로그로 추정 | 원인 |
|---|---|---|---|
| `03_exp_b_assist` | 2.09h | **5.28h** | actor 를 다른 이름으로 띄워 시계가 멈춰 있었다 |
| `panda_place_gpu` | 4.45h | **3.12h** | 그 잘못 띄운 actor 가 이쪽 시계를 부풀렸다 |

**수정.** `_elapsed_from_log(run_dir)` 를 추가해 `learner_metrics.jsonl` 의 에피소드 기록 시각
간격(600초 미만만 합산)으로 실제 가동 시간을 추정한다. actor 가 시작할 때
`max(elapsed.json, 로그추정)` 을 쓰고, 로그 쪽이 크면 파일도 갱신한다.
learner 가 남기는 기록이 런의 원본이므로 **어긋나도 다음 시작 때 저절로 맞춰진다.**
두 런의 값을 로그 기준으로 보정했다(03: 5.28h, panda: 3.12h).

**안내 문구에 명령을 넣었다.**
- `train.sh actor` 경고: 양쪽 설정과 출력 폴더를 나란히 보여주고,
  `export CFG=<learner 설정>` + `bash scripts/train.sh actor --window` 를 그대로 칠 수 있게 출력
- `dashboard.sh`: 따라가는 설정·런을 찍고, learner 가 없으면 `--run` 사용법을 안내

**런북 갱신 (§07c).**
"실험 이름을 바꾸면 CFG 를 칸마다 지정해야 한다" 절 신설 — 칸별 필요 여부 표,
안전장치 설명, 경과 시간 자가복구 설명. §00 명령 블록에도 `export CFG` 주석 추가.

### 7.60 `replay_checkpoints.sh` 구간 선택 (FROM / TO / EVERY)

"400~500번째 판 근처에서 혼자 성공했는데 그것만 다시 보고 싶다" 는 요청.
**에피소드 자체는 되살릴 수 없다** — 영상으로 녹화하지 않았고, 온라인 버퍼(40,000 전이 = 53판)는
링버퍼라 그 시점 데이터가 이미 밀려났다. **하지만 그 시기의 정책(체크포인트)은 남아 있다.**

기록에서 역산한 매핑:

| 혼자 성공한 판 | 시각 | 최적화 스텝 |
|---|---|---|
| 450 | 09-29 01:24 | ~178,010 |
| 483 | 01:43 | ~181,600 |
| 486 · 494 | 01:53 | ~182,010 |
| 558 | 02:47 | ~190,310 |

체크포인트가 2,000 스텝마다 있어 `178000 ~ 190000` 이 그대로 남아 있다.

**추가한 환경변수:**
```
FROM=178000 TO=190000 bash scripts/replay_checkpoints.sh 5   # 그 구간만, 판당 5회
EVERY=20000 bash scripts/replay_checkpoints.sh               # 2만 스텝 간격으로 건너뛰며
```
시작 배너에 구간을 찍고, "영상으로 남기려면 record_clips.py 를 먼저 띄우라" 는 안내도 넣었다.

검증: `FROM=178000 TO=190000` → 7개(178000~190000), `EVERY=40000` → 5개(002000·042000·082000·122000·162000).

---

## 8. UR5e 전환 (2026-10-23, `docs/ur5e-refactoring-prd.md` Phase 0~3)

같은 태스크를 UR5e(6축)로도 돌릴 수 있게 했다. 단계별 근거와 결정은 PRD §8 진행 기록에 있다.

| 무엇 | 어디 |
| --- | --- |
| 팔 관절 수 자동 탐지 (`joint1..N`). Panda 동작은 비트 단위로 동일 | gym-hil 패치 `0005` |
| UR5e 자산 — Menagerie 팔 + `panda.xml` 의 2F-85 서브트리 접붙이기, 토크 `<motor>` | gym-hil 패치 `0006` |
| UR5e 메시 (패치 밖, Menagerie 고정 커밋) | `scripts/server22/fetch_ur5e_assets.sh` |
| `UR5ePlaceCube*-v0` 환경과 등록, 로봇별 opspace 인자 `opspace_kwargs` | gym-hil 패치 `0007` |
| UR5e teleop 설정, `teleop.sh` 가 `CFG=` 를 받음 | `configs/env_config_ur5e_place_teleop.json`, `scripts/teleop.sh` |
| 패치가 같은 파일에 겹쳐도 `02_setup_project.sh` 재실행이 멈추지 않게 | `scripts/server22/02_setup_project.sh` 의 `series_applied` |
