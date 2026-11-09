# GPU 를 어디에 어떻게 쓰는가

이 학습에서 GPU 가 관여하는 지점은 **네 군데**다. 서로 독립적이라 하나가 CPU 로 떨어져도
나머지는 GPU 로 돈다 — 그래서 "GPU 서버인데 왜 느리지" 가 생긴다. 네 군데를 따로 확인해야 한다.

| # | 어디 | 켜는 방법 | 꺼졌을 때 증상 |
| --- | --- | --- | --- |
| 1 | **MuJoCo 카메라 렌더** (actor) | `MUJOCO_GL=egl` + NVIDIA EGL 라이브러리 | 학습은 되는데 스텝이 **30배** 느려짐 |
| 2 | **SAC 학습** (learner) | `policy.device: cuda` | 최적화 스텝이 느려짐 |
| 3 | **리플레이 버퍼** (learner) | `policy.storage_device: cuda` | 배치 샘플링 **41배** 느려짐 |
| 4 | **정책 추론** (actor) | `policy.device: cuda` (`CUDA_VISIBLE_DEVICES` 로 분리) | actor fps 하락 |

이 서버의 실측값: **2× A100 80GB, 드라이버 535.183.06, torch 2.11.0+cu126, MuJoCo 3.8.1.**

---

## 1. MuJoCo 카메라 렌더 — 가장 많이 틀어지는 곳

정책 입력은 128×128 카메라 두 대(`front`, `handcam_rgb`)다. 이 렌더를 GPU 가 하느냐 CPU 가
하느냐로 **전체 속도가 결정된다.** 학습 계산이 아니라 여기가 병목이다.

### 측정값

| 렌더 백엔드 | 스텝당 | 비고 |
| --- | --- | --- |
| `MUJOCO_GL=glfw` (VNC 의 GLX = llvmpipe, CPU) | **~690 ms** | 창을 안 띄워도 느리다 |
| `MUJOCO_GL=egl` (NVIDIA) · 창 없음 | **22.9 ms** | |
| `MUJOCO_GL=egl` (NVIDIA) · MuJoCo 창 있음 | **28.2 ms** | 창이 느린 게 아니다 |

> **자주 하는 오해**: "창을 띄워서 느리다"가 아니다. 느린 원인은 `MUJOCO_GL=glfw` 하나다.
> `MUJOCO_GL=egl` 이면 창을 띄워도 28 ms 로 정상이다.

`scripts/env.sh` 는 `DISPLAY` 가 있으면 `glfw` 를 고른다 — VNC 를 쓰면 이 조건에 걸린다.
그래서 `run_actor.sh`, `teleop.sh`, `record.sh`, `train.sh actor` 는 **`DISPLAY` 는 유지하고
`MUJOCO_GL` 만 `egl` 로 되돌린다**(키 입력 pynput 에는 `DISPLAY` 가 필요하기 때문).
덮어쓰려면 `HIL_ACTOR_GL=glfw`.

### 확인 방법

```bash
source scripts/env.sh && export MUJOCO_GL=egl
python -c "
import mujoco, os
mujoco.MjModel.from_xml_string('<mujoco/>')
from mujoco import gl_context
import OpenGL.GL as gl
" 2>/dev/null
bash scripts/server22/03_verify.sh gpu     # 2번 단계가 GL_RENDERER 를 찍는다
```

`GL_RENDERER` 에 **NVIDIA** 가 나오면 GPU, **llvmpipe** 면 CPU 다.

### llvmpipe 로 떨어졌을 때 — 컨테이너 환경의 함정

관리형 클라우드/컨테이너 이미지는 보통 **compute 라이브러리만** 넣는다
(`libcuda`, `libnvidia-ml`은 있고 `libEGL_nvidia`, `libnvidia-eglcore` 는 없음).
`NVIDIA_DRIVER_CAPABILITIES` 에 `graphics` 가 빠진 경우다.

- **컨테이너를 다시 띄울 수 있으면**: `-e NVIDIA_DRIVER_CAPABILITIES=all` (또는 `compute,utility,graphics`)
- **다시 띄울 수 없으면** (이 서버가 그랬다):

```bash
bash scripts/server22/05_nvidia_gl.sh
```

커널 모듈과 **같은 버전**의 드라이버 런파일을 받아 **압축만 푼다**(시스템 설치 아님).
GL 라이브러리를 `nvidia-gl/` 에 두고 `scripts/env.sh` 가 EGL ICD 를 그쪽으로 돌린다:

```bash
export LD_LIBRARY_PATH="$HIL_ROOT/nvidia-gl:$LD_LIBRARY_PATH"
export __EGL_VENDOR_LIBRARY_FILENAMES="$HIL_ROOT/nvidia-gl/nvidia_egl_icd.json"
```

> ⚠️ **버전이 정확히 맞아야 한다.** 호스트 드라이버가 바뀌면(`nvidia-smi` 버전이 바뀌면)
> `05_nvidia_gl.sh --force` 로 다시 받아야 한다. 어긋나면 EGL 이 장치를 **0개**로 보고
> `RuntimeError: MUJOCO_EGL_DEVICE_ID ... between 0 and -1` 로 죽는다.
> (운이 나쁘면 죽지 않고 조용히 llvmpipe 로 폴백해 30배 느려진다.)
>
> **이건 한 번 하고 끝나는 설정이 아니다.** 관리형 컨테이너는 다시 뜰 때마다 다른 호스트에
> 올라갈 수 있고 드라이버 버전이 따라 바뀐다. 실제 기록: `535.183.06`(9/17) →
> `535.247.01`(10/14) → `535.183.06`(10/23). 그때마다 어긋났다.
>
> 그래서 **`train.sh learner`/`actor`, `record.sh`, `teleop.sh` 가 시작할 때
> `preflight.sh --quick` 을 자동으로 돌린다**(0.05초). 버전이 어긋나 있으면
> 학습을 띄우지 않고 멈춘다 — 조용히 CPU 로 떨어지거나 알 수 없는 EGL 에러로
> 죽는 것을 막는다.
>
> ```bash
> bash scripts/preflight.sh --fix   # 막혔을 때. 커널과 같은 버전을 다시 받아 nvidia-gl/ 을 교체
> bash scripts/preflight.sh         # 전체 점검 (EGL 실렌더까지)
> ```
>
> `nvidia-gl/` 은 **git 에 올리지 않는다**(드라이버 버전에 묶인 바이너리, 127 MB).
> 새 머신에서는 반드시 그 머신의 드라이버 버전으로 다시 만든다.

---

## 2. SAC 학습 — `policy.device: cuda`

learner 가 크리틱 앙상블(2개) + discrete critic + 액터 + 온도 α 를 갱신한다.
`configs/train_config_03_exp_b_assist.json`:

```json
"policy": { "device": "cuda", "storage_device": "cuda" }
```

| 항목 | 값 |
| --- | --- |
| 최적화 스텝 속도 | **2.7 step/s** (batch 256) |
| 1 스텝 구성 | 크리틱 → discrete critic → 액터 → α → 타깃 soft update(τ=0.005) |
| `utd_ratio` | 1 (배치 1개/스텝) |

> lerobot 은 CUDA 를 못 찾으면 설정 파싱 때 `device` 를 **조용히 cpu 로 내린다.**
> 그래서 `03_verify.sh` 1번의 `cuda available: True` 를 먼저 확인해야 한다.

---

## 3. 리플레이 버퍼 — `storage_device: cuda` (가장 큰 단일 개선)

버퍼 전이를 RAM 에 둘지 GPU 메모리에 둘지다. 이미지가 들어 있어 전송량이 크다.

| `storage_device` | 배치 256 샘플링 | 이론상 최대 배치/초 |
| --- | --- | --- |
| `cpu` (상류 기본값) | **41.2 ms** | 24 |
| **`cuda`** | **1.0 ms** | **971** |

**41배.** learner 가 2.7 step/s 로 도는데 `cpu` 면 샘플링만으로 24 step/s 상한에 걸린다 —
여유가 있어 보이지만 다른 비용과 합쳐지면 실제로 느려진다.

### 메모리 비용

전이 1개 ≈ **400 KB** (128×128×3 float32 이미지 2장 + 상태 + next_state).

| 버퍼 | 용량 | 대략 메모리 |
| --- | --- | --- |
| 온라인 | 40,000 | ~16 GB |
| 시연(offline) | 40,000 | ~16 GB |
| 합계 + 모델 + CUDA 컨텍스트 | | **실측 36 GB / 80 GB** |

**GPU 메모리가 작으면 여기부터 줄인다:**

| GPU 메모리 | 권장 `online/offline_buffer_capacity` |
| --- | --- |
| 80 GB | 40000 / 40000 (이 설정) |
| 40 GB | 30000 / 30000 |
| 32 GB (RTX 5090) | 20000 / 30000 |
| 24 GB | 10000 / 20000 |
| 16 GB 이하 | `storage_device: cpu` 로 되돌리고 RAM 으로 |

> **오프라인 칸은 시연 프레임 수보다 커야 한다.** 시연은 시작할 때 오프라인 버퍼에 통째로 싣는데,
> 칸이 모자라면 learner 가 시작하자마자 `ValueError` 로 죽는다(`buffer.py:443`).
> Panda 시연 43판 = 19,612 프레임, UR5e 29판 = 7,253. 그래서 메모리를 줄일 때는 **온라인부터** 줄인다.
> (2026-11-06 정정: 이 표의 24 GB 행이 10000/10000 이었는데, Panda 시연으로는 돌지 않는다.)
> 40 GB 이하 행은 계산값이다: 필요 VRAM ≈ 7 GiB + (온라인 + 오프라인) × 384 KiB + actor 0.65 GiB.
> 7 GiB 는 버퍼 밖의 몫(모델 · 옵티마이저 · CUDA · torch.compile)으로, 실측 36.2 GiB − 버퍼 29.3 GiB 다.
> GPU 별 근거와 VRAM 말고 맞아야 하는 것(드라이버 · 그래픽 등)은 `docs/04_reproduction-handbook.html` §03 · §04.

버퍼 용량은 **기억 범위**이기도 하다. 40,000 ÷ (40초 × 20fps) ≈ **최근 50판**.
줄이면 그만큼 빨리 잊는다. 자세한 것은 `docs/02_place-cube-hil-serl-QA.html` Q11–Q12.

되돌리기: 실행할 때 `--policy.storage_device=cpu` 를 붙인다.

---

## 4. actor 정책 추론과 GPU 분리

actor 도 정책을 들고 매 스텝 `select_action()` 을 돌린다. 기본은 learner 와 같은 GPU 다.
`scripts/train.sh` 가 환경변수로 분리해 준다 (내부적으로 `CUDA_VISIBLE_DEVICES`):

```bash
HIL_LEARNER_GPU=0 bash scripts/train.sh learner    # learner 를 0번에
HIL_ACTOR_GPU=1   bash scripts/train.sh actor      # actor 를 1번에
HIL_ACTOR_GPU=cpu bash scripts/train.sh actor      # actor 추론만 CPU 로
```

이 서버는 **분리하지 않고 둘 다 0번**에서 돌렸다. GPU 사용률이 11~31% 라 나눌 이유가 없었다.
GPU 가 1장뿐인 서버도 그대로 된다.

---

## 5. 병목은 GPU 가 아니다

| 구성요소 | 속도 |
| --- | --- |
| actor env step | **20 step/s** (= fps 20, 시뮬레이션 상한) |
| learner 최적화 | 2.7 step/s |
| GPU 사용률 | **11~31%** |
| GPU 메모리 | 36 / 80 GB |

환경이 20 fps 로 고정이라 **GPU 를 더 투입해도 데이터가 그 속도로 들어오지 않는다.**
병렬 환경(Isaac Lab 류)이 아니라 단일 MuJoCo 환경 + 사람 개입 구조이기 때문이다.
A100 1장으로 충분하고, 더 작은 GPU 로도 버퍼만 줄이면 된다.

---

## 6. 새 서버로 옮길 때 점검 순서

```bash
# 1) GPU 확인 (고를 필요 없다. 스크립트가 본다)
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv

# 2) 설치 — torch 종류를 알아서 고르고, GPU 계산까지 확인하고, 안 되면 반대쪽으로 재시도
bash scripts/server22/02_setup_project.sh

# 3) EGL 이 GPU 를 잡는지 (가장 중요)
bash scripts/server22/03_verify.sh gpu
#   2번 단계 GL_RENDERER 에 NVIDIA → OK
#   llvmpipe → bash scripts/server22/05_nvidia_gl.sh 후 다시 3)

# 4) GPU 메모리에 맞춰 버퍼 용량 조정 (§3 표)

# 5) 스모크
scripts/run_smoke_test.sh configs/train_config_place_smoke_gpu.json
```

체크리스트:

- [ ] `torch.cuda.is_available()` 이 `True`
- [ ] `GL_RENDERER` 에 `NVIDIA` (llvmpipe 아님)
- [ ] `nvidia-gl/` 버전 == `nvidia-smi` 드라이버 버전 (쓰는 경우)
- [ ] `policy.device` 가 실행 로그에서 `cuda` (조용히 cpu 로 내려가지 않았는지)
- [ ] `policy.storage_device: cuda` 이고 GPU 메모리가 버틴다
- [ ] actor 로그 첫 줄 `[actor] MUJOCO_GL=egl`
- [ ] 스텝당 20~30 ms (690 ms 면 렌더가 CPU 다)

---

## 6-1. CUDA 버전은 어떻게 정해지나 — 직접 맞출 필요 없다

`02_setup_project.sh` 가 `compute_cap` 을 보고 고른다. 고른 뒤 **실제로 GPU 커널을 돌려 보고**,
실패하면 반대쪽(cu126 ↔ cu128)으로 한 번 더 설치한다. 둘 다 안 되면 0 이 아닌 코드로 멈춘다.

```
compute capability 9.x 이하  (RTX 20/30/40, A100, H100)  -> cu126
compute capability 10.0 이상 (RTX 50xx, B200 = Blackwell) -> cu128
GPU 없음                                                  -> cpu
```

### 왜 cu126 이 기본인가

cu126 휠에 들어 있는 아키텍처: `sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90`.
sm_89(RTX 4080/4090)는 전용 바이너리가 없지만 **같은 major 8 이라 sm_86 것으로 돈다**
(CUDA 는 같은 major 안에서 낮은 minor → 높은 minor 바이너리 호환을 보장한다).
Blackwell(sm_100/120)만 cu126 에 없어서 cu128 이 필요하다.

### 드라이버 버전은 생각보다 느슨하다

이 서버가 그 증거다:

```
nvidia-smi  ->  Driver 535.183.06,  CUDA Version: 12.2
torch       ->  2.11.0+cu126  (CUDA 12.6 런타임),  정상 동작
```

드라이버가 "CUDA 12.2" 라고 말해도 **12.6 런타임이 돈다.** CUDA minor version
compatibility 때문이다 — 12.x 런타임은 **드라이버 525.60.13 이상**이면 돈다.
그래서 "드라이버 ≥570.86 이어야 cu128" 같은 규칙은 필요 이상으로 엄격하다.
실제로 중요한 기준은 **525.60.13 이상인가** 하나뿐이다.

### 그래도 안 될 때

스크립트가 멈추면서 확인할 것을 찍는다:

```
- nvidia-smi 가 도는가 (드라이버가 있는가)
- 드라이버가 525.60.13 이상인가
- 컨테이너라면 --gpus all / NVIDIA_DRIVER_CAPABILITIES 가 있는가
```

예전 동작은 `cuda=False` 를 찍고도 "완료" 로 끝났다. 그래서 **CPU 로 학습이 도는 줄
모르고** 돌아가는 사고가 가능했다. 지금은 거기서 멈춘다.

---

## 7. GPU 가 아예 없을 때

동작은 한다. 느릴 뿐이다.

```bash
bash scripts/server22/02_setup_project.sh --cpu
# configs/train_config_panda_keyboard_cpu.json 참고:
#   policy.device: cpu, storage_device: cpu, batch_size 64, buffer 5000/2000
```

`MUJOCO_GL=egl` 은 GPU 가 없어도 동작한다(Mesa 소프트웨어 렌더로 폴백).
CPU 실측: learner 최적화 1스텝 ≈ 4.5 s(batch 32), actor 2~3 step/s.
**실용적인 학습은 어렵고 동작 검증용이다.**
