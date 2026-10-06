# HIL-SERL · MuJoCo Franka 큐브 놓기

Franka Panda 팔이 블록을 집어 바닥의 목표 패드에 내려놓도록, 사람이 중간중간 개입하며
강화학습으로 가르친다. LeRobot 의 HIL-SERL 구현(actor–learner 분리 + SAC/RLPD)을
MuJoCo 시뮬레이션과 **NVIDIA GPU 서버(Ubuntu 22.04 컨테이너)** 에서 돌린 기록이자,
다른 GPU 서버에서 **그대로 다시 돌릴 수 있게** 만든 저장소다.

## 결과

| | |
| --- | --- |
| 태스크 | `PandaPlaceCubeKeyboard-v0` — 집어서 목표 패드에 놓기 |
| 시연 | 43판 (19,612 프레임, 20 fps) |
| 학습 | **33,123 에피소드 · 802,000 최적화 스텝 · 40.9시간** |
| 최종 성적 | **마지막 200판 200/200 성공, 사람 개입 0%** |
| 하드웨어 | A100 80GB × 2 (1장으로 충분, GPU 사용률 11~31%) |

개입은 초반에 집중했고(~1,200판까지 6~12%), 그 뒤로는 거의 0% 로 두었는데도
계속 좋아졌다. 왜 그렇게 되는지는 [`docs/02_place-cube-hil-serl-QA.html`](docs/02_place-cube-hil-serl-QA.html) 에 정리했다.

```
lerobot v0.6.1 (7e241bd6) · gym-hil 0.1.14 (f89b3e55) · MuJoCo 3.8.1
torch 2.11.0+cu126 · Python 3.12 · NVIDIA 드라이버 535.183.06
```

---

## 1. 빠른 시작 (GPU 서버)

```bash
git clone <이 저장소> ~/project/hil-serl-mujoco-franka
cd ~/project/hil-serl-mujoco-franka

# 1) 사전 점검 (읽기 전용) — OS, GPU 드라이버, EGL 라이브러리, python3.12, 루프백
bash scripts/server22/00_check_server.sh | tee outputs/server_check.log

# 2) 시스템 패키지 + Python 3.12   (root 또는 sudo 필요)
bash scripts/server22/01_install_system.sh --with-vnc 2>&1 | tee outputs/server_install_system.log

# 3) (03_verify 의 GL_RENDERER 가 llvmpipe 일 때만) NVIDIA EGL 라이브러리 준비
bash scripts/server22/05_nvidia_gl.sh 2>&1 | tee outputs/server_nvidia_gl.log

# 4) 저장소(고정 커밋) + venv + torch + 패치 10개
#    torch 종류(cu126/cu128)는 GPU 를 보고 알아서 고른다. 고른 뒤 실제 GPU 계산까지
#    해 보고, 안 되면 반대쪽으로 한 번 더 설치한다. 그래도 안 되면 멈춘다.
bash scripts/server22/02_setup_project.sh 2>&1 | tee outputs/server_setup_project.log

# 5) 실행 검증 — CUDA · EGL 렌더러 · 데이터셋 · 환경 · 뷰어 · 설정 파싱
bash scripts/server22/03_verify.sh gpu 2>&1 | tee outputs/server_verify.log

# 6) 패치가 실제 코드를 만들어 내는지 대조 (docs/patches/README.md 의 스크립트)
```

**5) 에서 `GL_RENDERER` 에 `NVIDIA` 가 보여야 한다.** `llvmpipe` 면 카메라 렌더가
CPU 로 떨어져 스텝이 30배 느려진다 → 3) 으로 돌아간다. 자세한 것은 [`docs/gpu.md`](docs/gpu.md).

사람이 개입하려면 화면이 필요하다:

```bash
bash scripts/server22/04_start_vnc.sh      # VNC :1 + noVNC 6080 (localhost 바인드)
export DISPLAY=:1
```

---

## 2. 저장소에 무엇이 있고 무엇이 없는가

재현에 필요한 것은 전부 있고, **다시 만들 수 있는 것은 넣지 않았다.**

| | 상태 | 비고 |
| --- | --- | --- |
| `configs/`, `scripts/`, `docs/` | ✅ 있음 | |
| `docs/patches/` | ✅ 있음 | 상류 코드 변경 전부. 이게 `third_party/` 를 대신한다 |
| `data/panda_place_cube_demo/` | ✅ 있음 (40 MB) | 시연 43판. **학습 입력**이라 넣었다 |
| `outputs/releases/.../pretrained_model/` | ✅ 있음 (23 MB) | 성공한 정책 가중치 샘플 하나 |
| `third_party/lerobot`, `gym-hil` | ❌ 없음 | `02_setup_project.sh` 가 고정 커밋 clone + 패치 |
| `.venv/` | ❌ 없음 (7.5 GB) | `02_setup_project.sh` |
| `nvidia-gl/` | ❌ 없음 (127 MB) | **드라이버 버전에 묶인 바이너리.** `05_nvidia_gl.sh` 가 그 머신 버전으로 다시 받는다 |
| `cache/` | ❌ 없음 (5.5 GB) | HF 캐시. 첫 실행 때 자동 |
| `outputs/train/` | ❌ 없음 (40 GB) | 학습 산출물 |

이유는 [`.gitignore`](.gitignore) 에 주석으로 적어 두었다.

> **상류 소스를 복사해 넣지 않은 이유**: 복사본은 패치와 조용히 어긋난다.
> clone + 패치로 두면 어긋나는 순간 `git apply` 가 실패해서 알 수 있다.
> 실제로 이 저장소를 정리하면서 gym-hil 패치 2개가 어긋나 있던 것을 그렇게 찾았다.

---

## 3. GPU 를 쓰는 네 군데

| # | 어디 | 켜는 방법 | 꺼지면 |
| --- | --- | --- | --- |
| 1 | MuJoCo 카메라 렌더 (actor) | `MUJOCO_GL=egl` + NVIDIA EGL | 스텝 **690 ms** (정상 23 ms) |
| 2 | SAC 학습 (learner) | `policy.device: cuda` | 최적화가 느려짐 |
| 3 | 리플레이 버퍼 (learner) | `policy.storage_device: cuda` | 샘플링 **41.2 ms** (정상 1.0 ms) |
| 4 | 정책 추론 (actor) | `policy.device: cuda` | actor fps 하락 |

네 군데가 독립이라 **하나만 CPU 로 떨어져도 에러 없이 느려진다.**
확인 방법·측정값·GPU 메모리에 맞춘 버퍼 용량 조정은 → [`docs/gpu.md`](docs/gpu.md)

병목은 GPU 가 아니라 환경의 20 fps 다. GPU 사용률이 11~31% 라 A100 1장으로 충분하다.

### 어떤 GPU 에서 되나

CUDA 버전은 `02_setup_project.sh` 가 알아서 맞춘다. **실제로 신경 쓸 것은 메모리뿐이다.**

| GPU | sm | torch | 버퍼 설정 (`online`/`offline_buffer_capacity`) |
| --- | --- | --- | --- |
| RTX 2080 Ti (11 GB) | 75 | cu126 | `storage_device: cpu` 로 (GPU 메모리 부족) |
| RTX 3090 / 4090 (24 GB) | 86 / 89 | cu126 | **10000 / 10000** |
| RTX 4080 (16 GB) | 89 | cu126 | **5000 / 5000** 또는 `storage_device: cpu` |
| A100 40 GB | 80 | cu126 | **20000 / 20000** |
| **A100 80 GB (이 서버)** | 80 | cu126 | **40000 / 40000** ← 검증된 설정 |
| H100 | 90 | cu126 | 40000 / 40000 |
| RTX 50xx / B200 | 120 / 100 | **cu128** | 메모리에 맞춰 |

sm_89(4080/4090)는 cu126 휠에 전용 바이너리가 없지만 같은 major 8 이라 sm_86 것으로 돈다.
Blackwell(sm_100/120)만 cu128 이 필요하고, 스크립트가 `compute_cap` 을 보고 자동으로 고른다.

버퍼 용량을 줄이면 **기억 범위**가 줄어든다(40000 ≈ 최근 50판). 학습은 되지만 느려질 수 있다.

---

## 4. 학습 돌리기 — learner · actor · dashboard

터미널 세 개. 모두 프로젝트 루트에서 시작한다.

```bash
# 칸 A — learner (먼저. gRPC localhost:50051 바인딩 후 시연 데이터 로딩)
bash scripts/train.sh learner

# 칸 B — actor (learner 가 준비되면)
bash scripts/train.sh actor
bash scripts/train.sh actor --window     # MuJoCo 창도 같이 띄우려면

# 칸 C — 대시보드 (브라우저 http://127.0.0.1:8088/)
bash scripts/dashboard.sh
```

- 설정 파일은 `CFG` 로 바꾼다: `CFG=configs/train_config_place_smoke_gpu.json bash scripts/train.sh learner`
- **이어서 하기는 같은 명령을 다시 치면 된다.** `checkpoints/last` 가 있으면 알아서 `--resume=true` 를 붙인다.
- actor 는 learner 와 다른 `output_dir` 을 써야 한다 (스크립트가 `_actor` 를 붙여 처리).

진행 확인과 조작:

```bash
bash scripts/train.sh status     # 지금 어디까지 왔나
bash scripts/train.sh pause      # 시뮬레이터만 일시정지 (학습·연결은 살아 있음)
bash scripts/train.sh resume
bash scripts/train.sh save       # 지금 체크포인트를 남긴다
bash scripts/train.sh stop       # 체크포인트를 남기고 learner 를 정상 종료
bash scripts/train.sh new <이름> # 다른 이름으로 새 학습

python3 scripts/view.py          # 시뮬레이션 전용 뷰어 (8089, 카메라 2대 배치 선택)
python3 scripts/report.py        # 런 하나를 HTML 보고서로
```

조작 키(개입)는 `configs/keymap.json` 에서 바꾼다. 기본 배치와 전체 절차는
→ [`docs/01_place-cube-runbook.html`](docs/01_place-cube-runbook.html)

### 시연을 새로 모으려면

`data/panda_place_cube_demo/` 를 그대로 쓰면 바로 학습할 수 있다. 직접 모으려면:

```bash
bash scripts/record.sh 01                        # 세션 하나 녹화
python3 scripts/merge_demos.py --success-only    # 성공판만 모아 병합
```

---

## 5. 정상 동작 확인 체크리스트

새 서버에서 **학습이 제대로 도는지**를 끝까지 확인하는 순서다.
산출물을 똑같이 재현하는 것이 아니라, 구성요소가 전부 살아 있는지 보는 것이다.

| # | 확인 | 통과 기준 |
| --- | --- | --- |
| 1 | `03_verify.sh gpu` | 6단계 모두 통과, `cuda available: True` |
| 2 | `GL_RENDERER` | `NVIDIA` 포함 (llvmpipe 아님) |
| 3 | 패치 대조 (`docs/patches/README.md`) | 차이 0건 |
| 4 | 스모크 | `scripts/run_smoke_test.sh configs/train_config_place_smoke_gpu.json` 가 체크포인트까지 |
| 5 | learner | `[LEARNER] gRPC server started` → `Starting learner thread` |
| 6 | actor | `[ACTOR] Connection with Learner established`, 첫 줄에 `MUJOCO_GL=egl` |
| 7 | 연결 | learner 로그에 `[LEARNER] Actor episode: {...}` 가 쌓임 |
| 8 | 학습 진행 | `Optimization step` 증가, `logs/learner_metrics.jsonl` 에 손실 기록 |
| 9 | 속도 | 최적화 ~2.7 step/s, actor 20 step/s |
| 10 | 대시보드 | `http://127.0.0.1:8088/` 에 화면·그래프 |
| 11 | 개입 | VNC 화면에서 Space → HUD 에 `● 개입 중` |
| 12 | 평가 | `scripts/eval_policy.py --checkpoint outputs/releases/03_exp_b_assist_step242000/pretrained_model` |

12번은 동봉한 **성공한 정책**으로 바로 해볼 수 있다. 학습을 처음부터 돌리지 않고도
환경·렌더·정책 로딩이 정상인지 확인된다.

---

## 6. 파일 지도

```
.
├── README.md                     ← 이 문서
├── .gitignore                    무엇을 안 올리는지 + 이유
├── configs/
│   ├── train_config_03_exp_b_assist.json    본 학습 (위 결과를 낸 설정)
│   ├── train_config_panda_place_gpu.json    같은 태스크, online_steps 10만
│   ├── train_config_place_smoke_gpu.json    스모크 (3000스텝)
│   ├── train_config_panda_keyboard_cpu.json GPU 없는 환경용
│   ├── env_config_panda_place_record.json   시연 녹화
│   └── keymap.json                          개입 키 배치
├── scripts/
│   ├── env.sh                    venv·HF 캐시·MUJOCO_GL·EGL ICD (모든 스크립트가 source)
│   ├── train.sh                  learner/actor/status/pause/resume/save/stop/new
│   ├── dashboard.sh              진행 대시보드 (8088)
│   ├── view.py                   시뮬레이션 전용 뷰어 (8089)
│   ├── record.sh, merge_demos.py 시연 수집과 병합
│   ├── report.py                 런 하나 → HTML 보고서
│   ├── eval_policy.py            개입 없이 정책만 평가
│   ├── run_smoke_test.sh         자동 스모크
│   ├── record_clips.py           성공 장면만 mp4 로
│   ├── stop_after.py             N 에피소드에서 자동 중단
│   └── server22/                 00_check → 01_install → 02_setup → 03_verify → 04_vnc → 05_nvidia_gl
├── docs/
│   ├── gpu.md                    ★ GPU 를 어디에 어떻게 쓰는가
│   ├── server-ubuntu22.04-gpu.md 서버 설치 전체 기록과 문제 해결
│   ├── patches/README.md         ★ 패치 목록·순서·재현 점검
│   ├── 01_place-cube-runbook.html 조작 절차 (어떻게 하는가)
│   ├── 02_place-cube-hil-serl-QA.html 학습 원리 13문답 (왜 그런가)
│   ├── changes.md                상류 대비 변경과 그 이유 전부
│   ├── troubleshooting.md, verification.md, environment.md
│   └── requirements-lock.server22.txt  이 서버에서 고정된 패키지 목록
├── data/panda_place_cube_demo/   시연 43판 (학습 입력)
└── outputs/releases/03_exp_b_assist_step242000/
    ├── README.md                 이 시점의 성적
    └── pretrained_model/         성공한 정책 (추론에 필요한 전부)
```

---

## 7. 문서 읽는 순서

| 무엇을 알고 싶은가 | 어디 |
| --- | --- |
| 새 서버에 설치하기 | 이 문서 §1 → [`docs/server-ubuntu22.04-gpu.md`](docs/server-ubuntu22.04-gpu.md) |
| **GPU 를 어떻게 쓰나** | [`docs/gpu.md`](docs/gpu.md) |
| 상류 코드를 뭘 고쳤나 | [`docs/patches/README.md`](docs/patches/README.md) → [`docs/changes.md`](docs/changes.md) |
| 실제 조작 절차 | [`docs/01_place-cube-runbook.html`](docs/01_place-cube-runbook.html) |
| 학습이 왜 그렇게 움직이나 | [`docs/02_place-cube-hil-serl-QA.html`](docs/02_place-cube-hil-serl-QA.html) |
| 에러가 났다 | [`docs/troubleshooting.md`](docs/troubleshooting.md), `server-ubuntu22.04-gpu.md` §6 |

---

## 8. 출처

- [LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0) · [gym-hil](https://github.com/huggingface/gym-hil) (Apache-2.0)
- [LeRobot HIL-SERL 시뮬레이션 튜토리얼](https://huggingface.co/docs/lerobot/hilserl_sim)
- [Precise and Dexterous Robotic Manipulation via Human-in-the-Loop RL](https://arxiv.org/abs/2410.21845) (HIL-SERL 논문)

`third_party/` 의 코드 저작권은 각 상류 프로젝트에 있다.
이 저장소가 더한 것은 `configs/`, `scripts/`, `docs/`, 그리고 `docs/patches/` 의 변경분이다.
