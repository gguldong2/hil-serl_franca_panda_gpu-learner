# 실제 발생한 오류와 해결 기록

작업 순서대로 기록한다. "성공한 최종 절차"는 `README.md`, 변경 요약은 `docs/changes.md` 에 있다.

## 1. ROS Jazzy `PYTHONPATH` 가 모든 셸에 주입됨

- **현상**: `~/.bashrc` 가 `/mnt/c/.../lerobot_moveit2_mujoco/scripts/env.sh` 를 source 하여 `PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages` 가 항상 설정됨. venv 안에서도 ROS 의 numpy 등이 먼저 import 될 수 있음.
- **해결**: `scripts/env.sh` 에서 `unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH` 후 venv 활성화. `.bashrc` 는 수정하지 않음(기존 ROS 프로젝트 보존).

## 2. NVIDIA GPU 없음

- **현상**: `nvidia-smi` 없음, Windows GPU 는 Intel Arc 내장. `torch.cuda.is_available()==False`.
- **해결**: CPU 전용 torch 휠(`torch==2.11.0+cpu`) 설치, 설정 `policy.device=cpu`. CUDA 학습 검증은 **차단됨(GPU 미보유)** 으로 보고.

## 3. 오프라인 버퍼 이미지 스케일 불일치 (lerobot v0.6.1 버그)

- **현상**: `scripts/check_dataset.py` 실행 시 `ReplayBuffer.from_lerobot_dataset` 로 만든 배치의 이미지 `max=255.0`. 온라인 관측은 0~1.
- **원인**: `make_dataset()` 이 `return_uint8=True` 로 데이터셋을 만들고, 버퍼는 `torch.empty`(float32) 에 uint8 값을 그대로 복사.
- **해결**: `docs/patches/0001-buffer-uint8-image-scale.patch` (uint8 → float/255). 재실행 시 `max=1.0`.
- **로그**: `outputs/check_dataset.log`

## 4. 환경 1스텝 460 ms (10 fps 예산 100 ms 초과)

- **현상**: `scripts/check_env.py` 에서 평균 스텝 460 ms.
- **원인**: 프로파일 결과 물리 13.8 ms, 카메라 2장 렌더 343 ms (llvmpipe 소프트웨어 GL).
- **시도**: `LP_NUM_THREADS=16` 효과 없음. `MUJOCO_GL=egl` 도 llvmpipe(`/dev/dri` 권한 없음). `GALLIUM_DRIVER=d3d12` → 렌더 27 ms, 스텝 38 ms 로 해결되지만 아래 5번 문제.
- **결론**: 뷰어를 쓰는 actor/teleop 은 llvmpipe 로 실행(벽시계 기준 약 2~2.5 스텝/s). 시뮬레이션 시간은 `control_dt=0.1 s` 로 고정이므로 동역학·데이터는 동일하고 **벽시계 속도만** 느리다. 뷰어 없는 평가는 d3d12 사용.

## 5. `GALLIUM_DRIVER=d3d12` + 뷰어 창 → 크래시

- **현상**: `PandaPickCubeKeyboard-v0`(PassiveViewerWrapper 포함) 생성 직후 `D3D12: Removing Device.` 출력, 이어서 `Fatal glibc error: pthread_mutex_lock.c:94 ... assertion failed` 또는 무응답. 2회 재현(`outputs/gl_isolation.log`, `outputs/check_keyboard_inject.log`).
- **원인 추정**: 뷰어 스레드의 GL 컨텍스트와 메인 스레드의 오프스크린 렌더러가 d3d12 드라이버를 동시에 사용.
- **해결**: `scripts/env.sh` 기본값을 Mesa 기본(llvmpipe)으로 두고, d3d12 는 headless 실행에서만 사용자가 명시.

## 6. learner 시작 직후 `TypeError: typing.Any cannot be used with isinstance()`

- **현상**: `python -m lerobot.rl.learner` 가 `logging.info(pformat(cfg.to_dict()))` 에서 draccus 인코딩 오류로 종료. 체크포인트 저장(`cfg.save_pretrained`)도 같은 경로를 쓰므로 반드시 필요.
- **원인**: `HILSerl ResetConfig.fixed_reset_joint_positions: Any | None`. draccus 0.11.6 이 `Any` 를 인코딩하지 못함. upstream `main` 은 `list[float] | None` 으로 수정됨.
- **해결**: `docs/patches/0003-envs-configs-reset-joint-positions-type.patch`.

## 7. actor 가 learner 에 연결 불가 (`UNAVAILABLE ... tcp handshaker shutdown`, `DEADLINE_EXCEEDED`)

- **현상**: learner 가 `[LEARNER] gRPC server started` 를 출력하고 `ss -ltnp` 에 `[::ffff:127.0.0.1]:50051 LISTEN` 이 보이는데, actor 의 `Ready` 가 30회 실패 후 종료. 별도 클라이언트도 20 s `DEADLINE_EXCEEDED`.
- **잘못된 가설(기각)**: learner 대기 루프가 CPU 100% 로 GIL 을 점유해 gRPC 스레드가 응답 못한다고 추정 → `concurrency.learner=processes` 로 서버를 별 프로세스로 분리해도 동일하게 실패 → 기각, 공식값(threads)으로 복원.
- **실제 원인**: **이 WSL 에서 IPv4 루프백 TCP 연결이 되지 않음.** 순수 Python 소켓으로 `127.0.0.1:50071` 서버에 connect → `timed out`, `::1` 은 1 ms 에 성공. `.wslconfig` 의 `networkingMode=mirrored` 환경에서 발생(Windows 쪽 50051 리스너는 없음). 최소 gRPC 서버도 `127.0.0.1`/`0.0.0.0`/`[::]` 바인딩은 실패, `localhost` 바인딩(`[::1]` + `[::ffff:127.0.0.1]`)만 성공.
- **해결**: 설정 `policy.actor_learner_config.learner_host="localhost"` (learner 는 localhost 에 바인딩, actor 는 localhost 로 접속 → IPv6 ::1 경로). 코드 변경 없음.
- **비고**: WSL 재시작(`wsl --shutdown`)이나 `.wslconfig` 의 `networkingMode=NAT` 로 바꾸면 IPv4 루프백이 복구될 수 있으나, 기존 ROS 프로젝트가 mirrored 모드에 의존할 수 있어 변경하지 않았다.

## 8. actor 가 learner 와 같은 `output_dir` 로 시작 불가

- **현상**: `TrainPipelineConfig.validate()`: `FileExistsError: Output directory outputs/train/smoke already exists and resume is False` (dry-run 재현). learner 가 먼저 `output_dir/logs` 를 만들기 때문.
- **해결**: `scripts/run_actor.sh` 가 `--output_dir=<output_dir>_actor` 를 붙여 실행. actor 로그는 `outputs/train/<job>_actor/logs/`.

## 9. 합성 키 입력이 2회씩 도착 (검증 방법의 한계)

- **현상**: `pynput.Controller` 로 주입한 키가 XRecord 리스너에 press/release 각 2회 도착 → Space(토글)이 원위치. 홀드 키·Enter·Esc 는 정상.
- **영향**: 자동 검증으로는 개입 토글을 확정할 수 없음. **실제 키보드로 사용자 확인 필요** (`scripts/check_keyboard.py`). 물리 키도 2회 도착한다면 gym_hil `KeyboardController.on_press` 의 Space 처리에 디바운스가 필요(현재 미적용).

## 10. 종료되지 않은 learner 가 포트를 점유

- **현상**: 실패한 실행 뒤 learner(또는 spawn 된 gRPC 자식 프로세스)가 남아 50051 을 점유 → 다음 learner 가 바인딩 못함.
- **해결**: `ss -ltnp | grep 50051` 로 pid 확인 후 `kill`. `pkill -f "lerobot.rl"` 은 현재 셸 명령줄에도 매칭되어 셸을 죽일 수 있으니 `pgrep -f "[l]erobot.rl"` 패턴을 사용.

## 11. 공식 `lerobot.rl.eval_policy` 실행 불가

- **현상**: `env_cfg.pretrained_policy_name_or_path` 속성 없음, `make_robot_env` 반환값(tuple) 오용, 전처리 없이 raw obs 전달.
- **해결**: `scripts/eval_policy.py` 추가 구현 (`docs/changes.md` §4).
