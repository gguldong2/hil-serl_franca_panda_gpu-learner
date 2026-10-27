# Ubuntu 22.04 + NVIDIA GPU 서버(컨테이너)에서 실행하기

이 문서는 24.04 WSL에서 검증한 프로젝트를 **Ubuntu 22.04, NVIDIA GPU, 네이티브 리눅스(클라우드 컨테이너 포함)** 로 옮겨 실행하는 절차다. 스크립트는 `scripts/server22/` 에 있고 번호 순서대로 실행한다.

**검증 범위 표시.** 이 문서와 `scripts/server22/*` 는 2026-09-17 에 24.04 WSL PC에서 작성했다. 22.04 GPU 서버에서 실제로 실행한 것은 아니다. 이 PC에서 확인한 것: 스크립트 문법, `00_check_server.sh` 와 `03_verify.sh cpu` 실행, 번들 생성, 휠의 glibc 요구(전부 manylinux_2_28 이하 → 22.04 glibc 2.35 OK), torchcodec 이 FFmpeg 4~8 을 지원(22.04 apt ffmpeg 4.4 OK), 뷰어(GLFW)와 EGL 오프스크린 렌더러가 한 프로세스에서 공존. 서버에서 처음 실행할 때는 각 단계의 출력을 `outputs/` 에 남기고 실패 지점을 기록한다.

> **2026-09-17 갱신 — 실제로 실행했다.** A100 80GB ×2 컨테이너에서 §1~§4 전 과정을 돌렸고 §7 체크리스트 10개를 모두 채웠다. 서버에서만 드러난 문제 5가지(그 중 2개는 코드 수정이 필요했다)와 측정값은 **[§8](#8-이-서버에서-실제로-실행한-결과-2026-09-17)**. 단계 로그는 `outputs/server_*.log`.

## 0. 24.04 WSL 과 무엇이 다른가

| 항목 | 24.04 WSL (검증됨) | 22.04 GPU 서버 (이 문서) |
| --- | --- | --- |
| Python | 시스템 3.12 | 시스템 3.10 → **3.12 별도 설치** (deadsnakes PPA 또는 uv). lerobot v0.6.1 은 `>=3.12` 필수 |
| torch | 2.11.0+cpu | **2.11.0+cu128** (드라이버 ≥570.86. 미달이면 cu126) |
| 설정 | `train_config_panda_keyboard_cpu.json` (batch 64, buffer 5000/2000) | **`train_config_panda_keyboard_gpu.json`** (device cuda, batch 256·torch.compile 공식값 복원, buffer 20000/5000, online_steps 100000, save_freq 2000) |
| 렌더 | llvmpipe(CPU). d3d12 는 뷰어와 충돌 | 카메라 렌더 `MUJOCO_GL=egl` → NVIDIA EGL(GPU). 뷰어는 GLFW 로 X 디스플레이에 표시 |
| 화면·키보드 | WSLg 창 | 컨테이너에는 디스플레이가 없음 → **VNC(TigerVNC+noVNC) 를 띄워 브라우저로 조작** 또는 Xvfb(자동 검증만, 개입 불가) |
| gRPC 주소 | `localhost` (WSL IPv4 루프백 불통 우회) | `localhost` 그대로 (네이티브에서는 127.0.0.1 도 됨) |
| 코드·패치·데이터 | lerobot v0.6.1 + 패치 3개, gym-hil 0.1.14, 데이터 rev f80a680 | **동일** |
| 제거되는 WSL 전용 처리 | `GALLIUM_DRIVER`, ROS `PYTHONPATH`, `/mnt/c` | 없음 (`scripts/env.sh` 가 WSL 여부를 감지해 자동 분기) |

컨테이너에서 추가로 확인할 것: (1) root 인지 sudo 인지, (2) PID 1 이 systemd 가 아니면 VNC 를 수동 실행, (3) `NVIDIA_DRIVER_CAPABILITIES` 에 `graphics`(또는 `all`)가 없으면 EGL 이 GPU 를 못 쓰고 Mesa 소프트웨어로 폴백(느리지만 동작), (4) `/dev/shm` 이 작으면(64 MB) 큐/데이터로더에서 문제 소지 → 컨테이너 시작 시 `--shm-size=8g` 권장, (5) 포트 6080(noVNC)·50051(gRPC, 로컬 전용)은 외부 노출 불필요(SSH 터널).

## 1. 프로젝트 파일을 서버에 올리기

**기본: git clone.** 저장소에는 `configs/`, `scripts/`, `docs/`(패치 포함),
시연 데이터, 성공 정책 샘플이 들어 있다. `third_party/`·`.venv/`·`nvidia-gl/` 는
없고 2-3 단계에서 그 머신에 맞게 만들어진다 (`.gitignore` 주석 참고).

```bash
# [서버 | ~ | -]
git clone <이 저장소> ~/project/hil-serl-mujoco-franka
cd ~/project/hil-serl-mujoco-franka && mkdir -p outputs
```

git 을 못 쓰는 망이면 번들로 옮긴다:

```bash
# [보내는 쪽 | 프로젝트 루트 | -]
bash scripts/server22/make_bundle.sh          # → hil-serl-bundle.tgz (README, configs, scripts, docs/패치)
scp hil-serl-bundle.tgz <user>@<server>:~/
```

## 2. 서버에서: 순서대로 실행

모든 명령은 `[서버 | ~/project/hil-serl | -]` 에서 시작한다. 각 단계 로그를 `outputs/` 에 남긴다.

```bash
# 2-0. 풀기
mkdir -p ~/project/hil-serl && tar xzf ~/hil-serl-bundle.tgz -C ~/project/hil-serl && cd ~/project/hil-serl && mkdir -p outputs

# 2-1. 사전 점검 (읽기 전용). OS/glibc, root·sudo, GPU 드라이버·EGL 라이브러리, DISPLAY, python3.12 유무, 루프백
bash scripts/server22/00_check_server.sh | tee outputs/server_check.log
```

점검 결과로 결정할 것 세 가지:

| 결정 | 기준 | 선택 |
| --- | --- | --- |
| torch 종류 | `nvidia-smi` 드라이버 ≥570.86 | `--cu128` (기본). 그 미만이면 `--cu126`. GPU 없으면 `--cpu` |
| Python 3.12 방법 | `add-apt-repository` 가능(root/sudo) | `--python deadsnakes` (기본). PPA 불가·sudo 없음이면 `--python uv` |
| 디스플레이 | 사람 개입을 할 것인가 | 예 → `--with-vnc` 로 설치 후 `04_start_vnc.sh`. 아니오(자동 학습·평가만) → Xvfb 로 충분 |

```bash
# 2-2. 시스템 패키지 + Python 3.12   (root 또는 sudo)
bash scripts/server22/01_install_system.sh --with-vnc 2>&1 | tee outputs/server_install_system.log
#   sudo 가 없고 root 도 아니면: 관리자에게 이 스크립트의 apt 줄을 요청하고, Python 은  --python uv  로 대체

# 2-2b. (컨테이너에 NVIDIA 그래픽 라이브러리가 없을 때만) EGL 용 드라이버 라이브러리 준비
#   판단 기준: 00_check_server.sh 의 "libEGL_nvidia.so.0  없음" 또는 03_verify.sh 2번의 GL_RENDERER 가 llvmpipe
bash scripts/server22/05_nvidia_gl.sh 2>&1 | tee outputs/server_nvidia_gl.log
#   커널 드라이버와 같은 버전 런파일을 받아 압축만 푼다(설치 아님, 약 340 MB 다운로드 → nvidia-gl/ 127 MB).
#   호스트 드라이버가 바뀌면 다시 실행해야 한다.

# 2-3. 저장소(고정 커밋) + venv + torch + gym-hil + lerobot[hilserl] + 패치 13개 + UR5e 메시 + 버전 기록
#   torch 종류(cu126/cu128)는 GPU 의 compute capability 를 보고 알아서 고른다.
#   고른 뒤 실제 GPU 커널을 돌려 확인하고, 실패하면 반대쪽으로 한 번 더 설치한다.
#   둘 다 안 되면 멈춘다(예전에는 cuda=False 여도 "완료"로 끝났다). 자세한 것은 docs/gpu.md §6-1.
bash scripts/server22/02_setup_project.sh 2>&1 | tee outputs/server_setup_project.log
#   마지막 줄이 "[02] OK torch=2.11.0+cu126 cuda=True mujoco=3.8.1 lerobot=0.6.1" 이어야 한다.
#   패치는 lerobot 6개 + gym-hil 7개, 그리고 패치 밖의 UR5e 메시(Menagerie 고정 커밋).
#   적용 결과를 대조하는 방법은 docs/patches/README.md.

# 2-4. 실행 검증: CUDA 계산 → EGL 렌더러(GL_RENDERER 에 NVIDIA 가 찍혀야 GPU) → 데이터셋 → 환경 → 뷰어 환경(Xvfb) → 설정 파싱
bash scripts/server22/03_verify.sh gpu 2>&1 | tee outputs/server_verify.log
```

`03_verify.sh` 2번 단계에서 `GL_RENDERER: llvmpipe` 가 나오면 EGL 이 GPU 를 못 잡은 것이다. 학습은 되지만 카메라 렌더가 스텝당 약 340 ms 로 느려진다. 원인은 대개 컨테이너의 `NVIDIA_DRIVER_CAPABILITIES` 에 `graphics` 가 없거나 `libEGL_nvidia.so.0`/`/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 이 없는 것이다.

## 3. 서버에서: 화면과 키보드 (사람 개입용)

```bash
# 2-5. VNC 디스플레이 :1 + noVNC 6080 (컨테이너 안, 수동 실행)
bash scripts/server22/04_start_vnc.sh          # 처음 한 번 VNC 비밀번호 설정
export DISPLAY=:1                              # actor/teleop 을 띄울 모든 셸에서
```

로컬 PC에서 `ssh -L 6080:localhost:6080 <user>@<server>` 로 터널을 열고 브라우저에서 `http://localhost:6080/vnc.html` 에 접속한다. 바탕화면(fluxbox)이 보이면 준비 끝.

키보드 확인(README §5.4 와 동일, 창은 브라우저 안 VNC 화면에 뜬다):

```bash
# [서버 | ~/project/hil-serl | venv, DISPLAY=:1]
source scripts/env.sh
MUJOCO_GL=egl python scripts/check_keyboard.py --seconds 120
```

브라우저의 VNC 화면을 클릭해 포커스를 주고 Space 를 한 번 누르면 터미널에 `is_intervention=True` 가 찍혀야 한다. WSLg 에서 있었던 "합성 키 2회 도착" 문제는 Xvnc 에서는 나타나지 않을 것으로 예상하지만 확인이 필요하다. 조작표는 README §5.3.

## 4. 서버에서: 학습·평가

```bash
# 자동 스모크 (learner → Ready → actor 500스텝 → 체크포인트 50스텝마다 → 종료). DISPLAY 가 있으면 VNC 에 창이 뜨고, 없으면 xvfb-run 으로 감싼다.
source scripts/env.sh && export MUJOCO_GL=egl
scripts/run_smoke_test.sh configs/train_config_smoke_gpu.json            # DISPLAY=:1 인 셸
#   DISPLAY 없이:  xvfb-run -a -s "-screen 0 1280x800x24" scripts/run_smoke_test.sh configs/train_config_smoke_gpu.json

# 본 학습 (터미널 2개, 둘 다 source scripts/env.sh && export MUJOCO_GL=egl)
scripts/run_learner.sh configs/train_config_panda_keyboard_gpu.json      # 터미널 1
DISPLAY=:1 scripts/run_actor.sh configs/train_config_panda_keyboard_gpu.json   # 터미널 2 ("Starting learner thread" 이후)

# 평가 (개입 없음, 가중치만 로딩)
MUJOCO_GL=egl python scripts/eval_policy.py --checkpoint outputs/train/panda_keyboard_gpu/checkpoints/last/pretrained_model --n_episodes 20
# 재개
python -m lerobot.rl.learner --config_path=outputs/train/panda_keyboard_gpu/checkpoints/last/pretrained_model/train_config.json --resume=true
```

개입 방법, 지표 파일(`logs/learner_metrics.jsonl`), 체크포인트 구조, 재개 복원 범위는 README §8~§10 과 `docs/verification.md` §3 이 그대로 적용된다.

## 5. GPU 설정값과 근거

> GPU 가 관여하는 네 지점과 측정값 전체는 **[`docs/gpu.md`](gpu.md)** 에 따로 정리했다.
> 여기는 설정 파일 값만 둔다.

본 학습 설정 `configs/train_config_03_exp_b_assist.json` (위 결과를 낸 것):

| 키 | CPU 설정 | **이 서버(A100 80GB)** | 근거 |
| --- | --- | --- | --- |
| `policy.device` | cpu | **cuda** | SAC 갱신 |
| `policy.storage_device` | cpu | **cuda** | ★ 배치 샘플링 41.2 ms → **1.0 ms**. 단일 변경으로 가장 큰 개선 |
| `batch_size` | 64 | **256** | 공식값 |
| `online_ratio` | 0.5 | 0.5 | 온라인 128 + 시연 128 |
| `online_buffer_capacity` / `offline_buffer_capacity` | 5000 / 2000 | **40000 / 40000** | 전이당 ≈400 KB. 두 버퍼 + 모델 **실측 36 GB / 80 GB**. 40,000 = 최근 50판 분량 |
| `policy.online_steps` | 20000 | **1000000** | 실행 상한. 실제로는 802,000 스텝에서 수동 종료 |
| `online_step_before_learning` | 100 | 100 | 온라인 버퍼가 이만큼 차야 학습 시작 |
| `save_freq` | 200 | **2000** | ≈ 12분마다. 체크포인트 1개 81 MB |
| `env.fps` | 10 | **20** | |
| `processor.reset.control_time_s` | — | **40.0** | 에피소드 상한 40초. 20초면 성공한 판의 62% 가 잘린다(실측) |
| `actor_learner_config.policy_parameters_push_frequency` | 4 | **50** | **초 단위.** 가중치를 actor 로 보내는 주기 |
| `learner_host` | localhost | localhost | 변경 없음 |

GPU 메모리가 80 GB 보다 작으면 **버퍼 용량부터** 줄인다 (40 GB→20000/20000,
24 GB→10000/10000, 16 GB 이하→`storage_device: cpu`). 표는 `docs/gpu.md` §3.

### 실측 성능 (이 설정, 40.9시간 연속)

| 항목 | 값 |
| --- | --- |
| learner 최적화 | **2.7 step/s** |
| actor 환경 스텝 | **20 step/s** (= fps 상한) |
| 카메라 렌더 | 22.9 ms/step (창 없음) · 28.2 ms (창 있음) |
| GPU 사용률 / 메모리 | **11~31% / 36 GB** |
| 체크포인트 | 2,000 스텝 ≈ 12분마다 |

**병목은 GPU 가 아니라 환경의 20 fps 다.** A100 1장으로 충분하고, 두 장을 나눠 쓸
필요도 없었다(`HIL_LEARNER_GPU` / `HIL_ACTOR_GPU` 로 분리 가능).

## 6. 예상 문제와 대처 (서버에서 처음 만날 수 있는 것)

| 증상 | 원인 | 대처 |
| --- | --- | --- |
| `add-apt-repository: command not found` | `software-properties-common` 없음 | `01_install_system.sh` 가 먼저 설치함. 그래도 PPA 가 막힌 네트워크면 `--python uv` |
| `pip install torch ... cu128` 후 `cuda=False` | 드라이버가 570.86 미만 | `02_setup_project.sh --cu126` 로 재설치(venv 삭제 후) |
| `03_verify.sh` 6번에서 GPU 설정인데 `device=cpu` 로 출력 | lerobot 이 CUDA 를 못 찾으면 설정 파싱 시 device 를 조용히 cpu 로 내림 (이 PC에서 그렇게 동작함) | 1번 단계의 `cuda available` 을 먼저 확인. False 면 torch 설치(flavor)·드라이버 문제 |
| `GL_RENDERER: llvmpipe` (03_verify 2번) | 컨테이너에 graphics 캡/NVIDIA EGL 없음 | 컨테이너 재시작 옵션 `-e NVIDIA_DRIVER_CAPABILITIES=all` 또는 `--gpus all` + `graphics`. 불가하면 느린 상태로 진행(동작은 함) |
| `Could not initialize GLFW` / `GLX` 오류 (actor) | DISPLAY 없음 또는 VNC 미기동 | `04_start_vnc.sh` 후 `export DISPLAY=:1`, 또는 `xvfb-run -a` |
| pynput `Xlib.error.DisplayConnectionError` | 같은 원인 | 같은 대처 |
| actor `FileExistsError: Output directory ...` | learner 와 같은 output_dir | `scripts/run_actor.sh` 사용(자동으로 `_actor` 붙임) |
| actor `DEADLINE_EXCEEDED` / learner 미응답 | 루프백 문제 또는 이전 learner 잔존 | `ss -ltnp \| grep 50051`, `pgrep -f "[l]erobot.rl"` 로 정리. `00_check_server.sh` 루프백 결과 확인 |
| `GLIBCXX_3.4.30 not found` | 오래된 libstdc++ | 22.04 기본(libstdc++6 12.x)에는 있음. 없다면 `apt install libstdc++6` 업데이트 |
| learner OOM(RAM) | buffer 용량 과대 | `online_buffer_capacity` 를 RAM/393 KB 의 절반 이하로 |
| torch.compile 오류 | GPU/드라이버·Triton 조합 | `algorithm.use_torch_compile=false` |
| 멀티프로세스 큐 오류·`/dev/shm` 부족 | 컨테이너 shm 64 MB | `--shm-size=8g` 로 컨테이너 재시작 |
| **(실제 발생)** `GL_RENDERER: llvmpipe` 인데 컨테이너 재시작 옵션을 바꿀 수 없음(관리형 클라우드) | 이미지에 compute 라이브러리(`libcuda`, `libnvidia-ml`)만 있고 그래픽(`libEGL_nvidia`)이 없음 | `bash scripts/server22/05_nvidia_gl.sh` — 커널과 **같은 버전**의 드라이버 런파일을 받아 압축만 풀어 `nvidia-gl/` 에 두고 `scripts/env.sh` 가 EGL ICD 를 그쪽으로 돌린다(시스템 설치 아님). 호스트 드라이버가 바뀌면 다시 실행 |
| **(실제 발생)** 뷰어를 띄운 프로세스가 **작업을 다 끝낸 뒤**(`env closed OK` 출력 후) `Segmentation fault (core dumped)` | pynput 키보드 리스너 스레드가 아직 X 연결을 쓰는 중에 `glfw.terminate()` 가 돌아 경합 | `docs/patches/gym-hil/0001-keyboard-listener-join-on-stop.patch` (`02_setup_project.sh` 가 자동 적용). 미적용 시 약 16%(3/19), 적용 후 0/27 |
| **(실제 발생)** `check_keyboard.py --inject` 에서 SPACE·방향키가 안 먹고 ENTER/ESC 만 먹음 | **검증 스크립트의 주입 방식 문제**. pynput `Controller` 는 같은 프로세스 리스너에 키를 2번 넣고, xdotool 에 키심 이름(`Shift_R`)을 주면 왼쪽 modifier 가 같이 눌린다 | 기본값이 `--inject-backend xtest`(키코드 직접 주입)로 바뀌었다. **사람이 VNC 화면에서 실제로 누르는 키는 해당 없음** |
| **(실제 발생)** `eval_policy.py` 가 `Expected all tensors to be on the same device` 로 죽음 | `select_action()` 은 cuda 텐서를, `post.process_action()` 은 env 로 넘기려고 cpu 텐서를 돌려주는데 그대로 `torch.cat` | 수정 완료(두 조각의 device 를 맞춘 뒤 cat). CPU 전용 WSL 에서는 둘 다 cpu 라 드러나지 않던 **GPU 전용** 문제 |
| **(실제 발생)** noVNC 가 `0.0.0.0:6080` 으로 열림 | `04_start_vnc.sh` 의 websockify 인자에 바인드 주소가 없었음 | `localhost:6080` 으로 수정 완료. 접속은 SSH 터널로 |

## 7. 완료 기준 체크리스트 (2026-09-17 서버에서 채움 — 10/10)

| 항목 | 확인 명령 / 파일 | 결과 |
| --- | --- | --- |
| 환경 설치 | `02_setup_project.sh` 마지막 줄, `docs/requirements-lock.server22.txt` | ✅ `[02] OK torch=2.11.0+cu126 cuda=True mujoco=3.8.1 lerobot=0.6.1`, gpu: NVIDIA A100 80GB PCIe. 패치 3개 적용, `pip check` 통과 |
| GPU 사용 | `03_verify.sh` 1번: `cuda available: True`, matmul ms | ✅ `True`, matmul 4096² **74.7 ms**, resnet18 fwd/bwd OK |
| GPU 렌더 | `03_verify.sh` 2번: `GL_RENDERER` 에 NVIDIA | ✅ `NVIDIA A100 80GB PCIe/PCIe/SSE2`, 2-카메라 **2.4 ms/step**. 단 `05_nvidia_gl.sh` 선행 필요(§8) |
| 데이터 로딩 | `outputs/check_dataset.log` 이미지 스케일 "일치" | ✅ 버퍼 458 프레임, 오프라인/온라인 모두 0~1 **일치** |
| 환경 동작·정책 관측 | `outputs/check_env_headless.log`, `outputs/check_env/*.png` | ✅ **34.7 ms/step** (10 FPS 예산 100 ms), 관측 3키 shape/범위 정상 |
| 실시간 화면 | VNC 화면에 "MuJoCo : Scene" 창 | ✅ `0xa00007 "MuJoCo : Scene" 1066x600`, 캡처 `outputs/vnc_viewer_running.png`. 뷰어 포함 **33.5 ms/step** |
| 키보드 조작 | `check_keyboard.py` 에서 Space → `is_intervention=True` | ✅ 8개 항목 전부: SPACE 토글, 방향키 ±xy, Shift_R/L ±z, Ctrl 그리퍼 0↔255, ENTER reward 1.0, ESC done. `outputs/check_keyboard_inject.log` |
| 학습 루프·저장 | `run_smoke_test.sh configs/train_config_smoke_gpu.json` → `outputs/train/smoke_gpu/checkpoints/` | ✅ learner Ready 9초, actor 500스텝 exit=0, 체크포인트 50/100/150/200, `learner_metrics.jsonl` 45줄, learner exit=0 |
| 재개 | `--resume=true` 로그 `Resuming from step` | ✅ `Resuming from step 200, interaction step 0` → 이어서 205 스텝. `outputs/resume_check.log` |
| 자율 평가 | `outputs/train/<job>/eval_<step>.json` | ✅ `eval_000200.json` (5 에피소드, 성공 0/5 — 200스텝 스모크 가중치라 정상). 단 GPU device 버그 수정 필요했음(§8) |

## 8. 이 서버에서 실제로 실행한 결과 (2026-09-17)

**대상 서버.** Ubuntu 22.04.5 (glibc 2.35, 커널 5.15.0), NVIDIA **A100 80GB PCIe ×2**, 드라이버 **535.183.06**, 32 core, RAM 384 GB, `/dev/shm` 256 GB, 디스크 여유 85 GB. 컨테이너(PID 1 = `init`, systemd 아님), 비-root 사용자 + 무암호 sudo, `DISPLAY` 없음. 프로젝트 루트 `~/project/hil-serl-mujoco-franka`.

**§2 의 세 가지 결정 — 이 서버에서 고른 값.**

| 결정 | 근거 | 선택 |
| --- | --- | --- |
| torch 종류 | 드라이버 535.183.06 **< 570.86** | **`--cu126`** (기본값 cu128 아님) |
| Python 3.12 | 이미 uv 로 설치된 3.12.13 이 있음 | **`--python uv`** (deadsnakes PPA 불필요) |
| 디스플레이 | 사람 개입을 함 | **`--with-vnc`** → `04_start_vnc.sh` (`:1`, 1600x900, noVNC `127.0.0.1:6080`) |

실행한 순서: `00_check_server.sh` → `01_install_system.sh --with-vnc --python uv` → **`05_nvidia_gl.sh`(신규)** → `02_setup_project.sh --cu126` → `04_start_vnc.sh` → `03_verify.sh gpu` → `run_smoke_test.sh` → 재개 → `eval_policy.py`.

**패키지는 WSL 기준과 동일.** `docs/requirements-lock.server22.txt` 를 `docs/requirements-lock.txt` 와 비교하면 **torch/torchvision 의 `+cu126` vs `+cpu` 외에 차이가 없다** (mujoco 3.8.1, torchcodec 0.11.1, numpy 2.2.6, grpcio 1.84.0, transformers 5.5.4, datasets 4.8.5 모두 일치).

### 8.1 서버에서만 드러난 문제 5가지

**(1) EGL 이 GPU 를 못 잡는다 — 컨테이너에 그래픽 라이브러리가 없다.** `ldconfig` 에 `libcuda`/`libnvidia-ml` 같은 compute 라이브러리만 있고 `libEGL_nvidia.so.0` 이 없다(`NVIDIA_DRIVER_CAPABILITIES` unset). 문서 §6 이 권하는 "컨테이너를 `-e NVIDIA_DRIVER_CAPABILITIES=all` 로 재시작"은 관리형 클라우드라 불가능했다. 그래서 커널 모듈과 **같은 버전**(535.183.06)의 드라이버 런파일을 받아 **설치 없이 압축만 풀어** GL 라이브러리 6개를 `nvidia-gl/` 에 두고, `scripts/env.sh` 가 `LD_LIBRARY_PATH` 와 `__EGL_VENDOR_LIBRARY_FILENAMES` 를 그쪽으로 돌리게 했다. 이 작업은 `scripts/server22/05_nvidia_gl.sh` 로 자동화했다(드라이버 버전을 `nvidia-smi` 에서 읽어 맞춘다).

> **버전이 정확히 맞아야 한다.** 같은 서버의 이전 프로젝트(`~/project/ser/nvidia-gl/`)에는 580.173.02 라이브러리가 들어 있었는데, 호스트 드라이버가 535.183.06 으로 바뀌면서 못 쓰게 됐다. 호스트 드라이버가 바뀔 때마다 `05_nvidia_gl.sh` 를 다시 실행해야 한다.

효과: 2-카메라 렌더 **340 ms → 2.4 ms/step** (약 140배).

**(2) 뷰어를 쓴 프로세스가 종료할 때 간헐적으로 SIGSEGV.** `03_verify.sh` 5번에서 처음 나왔다. faulthandler 로 보면 메인 스레드는 `glfw.terminate()`, 별도 스레드는 pynput 의 X RECORD 루프(`Xlib ... send_and_recv`)에 있었다. `InputsControlWrapper.close()` 가 리스너를 `stop()` 만 하고 join 하지 않아서, 스레드가 아직 X 연결을 쓰는 사이 GLFW 가 정리되며 겹친다. **작업이 다 끝난 뒤**(`env closed OK` 출력 후) 터지므로 결과·체크포인트에는 영향이 없지만 종료 코드가 139 가 된다. `stop()` 뒤에 `join(timeout=2.0)` 을 넣어 해결했다 — 패치 전 3/19(약 16%), 패치 후 **0/27**.

**(3) 키보드 자동 검증이 거짓 실패.** `check_keyboard.py --inject` 에서 SPACE·방향키·Ctrl 은 안 먹고 ENTER/ESC 만 먹었다. 원인은 환경이 아니라 **키 주입 방식**이었고, 실측으로 갈랐다:

| 주입 방법 | 리스너가 받은 이벤트 (SPACE 1회) | 결과 |
| --- | --- | --- |
| pynput `Controller` | press **2** / release **2** | SPACE 토글이 켜졌다 바로 꺼짐. ENTER/ESC 는 상태를 세팅만 해서 두 번 와도 정상으로 보임 |
| `xdotool keydown Shift_R` (키심 이름) | `Key.shift` + `Key.shift_r` 동시 | +z 와 -z 가 상쇄 |
| **XTEST 키코드 직접**(`keycode 62`) | press 1 / release 1 | ✅ 사람이 VNC 에서 누른 것과 동일 |

`check_keyboard.py` 의 기본 주입을 XTEST 키코드로 바꿨다(`--inject-backend xtest`, 옛 동작은 `pynput`). **사람이 실제로 누르는 키는 원래부터 정상**이었다. 참고로 pynput 은 `Key.shift is Key.shift_l`, `Key.ctrl is Key.ctrl_l` (별칭)이라 왼쪽 modifier 는 일반형으로 오고, gym_hil 핸들러가 이미 그 전제로 쓰여 있다. 문서 §3 이 걱정한 WSLg 의 "합성 키 2회 도착"은 **Xvnc 의 문제가 아니라 pynput Controller 의 성질**이었다.

**(4) `eval_policy.py` 가 GPU 에서 죽는다.** `select_action()` 은 cuda 텐서를 주고 `post.process_action()` 은 env 로 넘기려고 cpu 텐서를 주는데, 그대로 `torch.cat` 해서 `Expected all tensors to be on the same device` 가 났다. 이어붙이기 전에 device 를 맞춰 해결했다. CPU 전용 WSL 에서는 둘 다 cpu 라 드러날 수 없던 문제다.

**(5) noVNC 가 모든 인터페이스에 열렸다.** `04_start_vnc.sh` 의 websockify 호출에 바인드 주소가 없어 `0.0.0.0:6080` 으로 떴다. VNC 자체(5901)는 `-localhost yes` 였지만 noVNC 는 외부에서 닿을 수 있었다. `localhost:6080` 으로 고쳤다(문서 §0 의 "외부 노출 불필요" 의도와 일치).

### 8.2 측정값

| 항목 | 값 |
| --- | --- |
| matmul 4096² (cuda) | 74.7 ms |
| MuJoCo 2-카메라 렌더 (128×128, EGL) | **2.4 ms/step** (llvmpipe ≈ 340 ms) |
| env 스텝 (headless / 뷰어 포함) | 34.7 ms / 33.5 ms (10 FPS 예산 100 ms → 약 3배 여유) |
| learner 최적화 루프 | **6.1 ~ 6.7 Hz** (batch 256, `use_torch_compile=false`) |
| learner gRPC Ready 까지 | 9 초 |
| GPU 메모리 (learner, 스모크) | 약 5.1 GB / 80 GB |
| 오프라인 데모 버퍼 | 458 프레임 |
| 정책 파라미터 | 5,797,641 |

### 8.3 본 학습 전에 정할 것

- `save_freq`: 스모크에서 최적화 루프가 6.1~6.7 Hz 였다. 설정값 2000 스텝이면 체크포인트 간격이 약 5분이다. 그대로 써도 되고, 더 촘촘히 보려면 1000 으로 줄인다.
- `online_buffer_capacity=20000`: 전이당 약 393 KB → 최대 약 9.8 GB. 이 서버 RAM 384 GB 이므로 여유가 크다. 오히려 늘려도 된다.
- `use_torch_compile`: 스모크(`train_config_smoke_gpu.json`)는 false 였다. 본 설정은 true 이므로 **첫 실행 때 컴파일 지연과 오류 가능성을 확인**해야 한다. 문제가 나면 false 로.
- 두 번째 GPU 가 놀고 있다. `CUDA_VISIBLE_DEVICES` 로 learner 와 actor 를 나누면(예: learner=1, actor=0) 렌더와 학습이 서로 방해하지 않는다.
