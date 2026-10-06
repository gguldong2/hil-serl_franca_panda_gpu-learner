# 환경 명세 (실제 확인값, 2026-09-16)

PRD §4 "환경 점검" 항목을 실제 명령으로 확인한 결과다. 모든 값은 이 PC의 WSL 안에서 직접 실행해 얻었다.

## 1. 실행 위치·OS

| 항목 | 확인값 | 확인 명령 |
| --- | --- | --- |
| 실행 위치 | 사용자 PC의 WSL2 (컨테이너·원격 아님, `WSL_INTEROP` 존재) | `echo $WSL_DISTRO_NAME $WSL_INTEROP` |
| Windows | Windows 11 Home 10.0.26200 (build 26200) | `powershell.exe Get-CimInstance Win32_OperatingSystem` |
| WSL | WSL 2.4.11.0, 커널 5.15.167.4-microsoft-standard-WSL2, WSLg 1.0.65, Direct3D 1.611.1 | `wsl.exe --version`, `uname -r` |
| 배포판 | Ubuntu 24.04.2 LTS (Noble) | `cat /etc/os-release` |
| `.wslconfig` | `[wsl2] networkingMode=mirrored` 만 설정 (메모리 제한 없음) | `cat /mnt/c/Users/liqej/.wslconfig` |

## 2. 하드웨어

| 항목 | 확인값 |
| --- | --- |
| CPU | Intel Core Ultra 5 125H, WSL에 18 논리 스레드 노출 |
| RAM | 호스트 16 GB, WSL에 7.5 GiB 할당 + swap 2 GiB (WSL 기본 50%) |
| 디스크 | WSL ext4(`/`) 1007 GB 중 949 GB 여유. Windows `C:` 457 GB 중 **14 GB만 여유** |
| GPU | **Intel Arc Graphics (내장, 드라이버 32.0.101.5768)**. NVIDIA GPU 없음 |

## 3. GPU 런타임

| 항목 | 확인값 |
| --- | --- |
| `nvidia-smi` | 없음 (WSL, Windows 모두). `/usr/lib/wsl/lib`에는 `libd3d12.so`, `libdxcore.so`만 존재 |
| PyTorch | `torch 2.11.0+cpu`, `torch.cuda.is_available() == False` |
| CPU 연산 검증 | 2048×2048 행렬곱 172 ms, resnet18 128×128 4장 forward/backward 정상 |
| 결론 | **CUDA 학습 불가 → CPU 학습 경로로 진행** (PRD §4 "NVIDIA GPU가 없다면" 항목 적용) |

## 4. GUI

| 항목 | 확인값 |
| --- | --- |
| WSLg | `DISPLAY=:0`, `WAYLAND_DISPLAY=wayland-0`, `/mnt/wslg`, `/tmp/.X11-unix/X0` 존재 |
| X 서버 | `xdpyinfo`: vendor Microsoft Corporation, 2560×1600 |
| OpenGL(기본) | `glxinfo -B`: **llvmpipe** (LLVM 20.1.2) 소프트웨어 렌더, GL 4.5 core, Accelerated: no |
| OpenGL(d3d12) | `GALLIUM_DRIVER=d3d12 glxinfo -B`: renderer `D3D12 (Intel(R) Arc(TM) Graphics)`, Accelerated: yes |
| `/dev/dri` | `card0`(video 그룹), `renderD128`(geoclue 그룹) — 현재 사용자는 두 그룹 미소속 → EGL 은 llvmpipe 로 폴백 (`libEGL warning: failed to open /dev/dri/card0: Permission denied`) |
| MuJoCo 오프스크린 | `MUJOCO_GL=glfw`/`egl` 모두 128×128 2카메라 렌더 성공. `osmesa` 는 라이브러리 없음(실패) |
| MuJoCo 뷰어 창 | `PassiveViewerWrapper`(glfw) 로 "MuJoCo : Scene" 1706×1066 X 창 생성 확인 (`xwininfo -root -tree`) |

렌더 시간(환경 1스텝, 카메라 2장 + 물리 50 substep):

| 설정 | 물리 | 렌더(2캠) | 전체 스텝 |
| --- | --- | --- | --- |
| llvmpipe (기본) | 13.8 ms | 343.5 ms | 374 ms |
| `GALLIUM_DRIVER=d3d12` | 15.6 ms | 27.5 ms | 38 ms |

**d3d12 제한**: 뷰어 창(별도 스레드)과 오프스크린 렌더러를 한 프로세스에서 같이 쓰면 `D3D12: Removing Device.` 후 glibc mutex assertion 으로 크래시/정지가 재현됐다(2회). 따라서 **뷰어를 쓰는 actor/teleop 은 llvmpipe(느림)**, 뷰어 없는 headless 평가만 d3d12 를 쓴다. 자세한 내용은 `docs/troubleshooting.md`.

## 5. 입력

| 항목 | 확인값 |
| --- | --- |
| 입력 라이브러리 | gym_hil `KeyboardController` → `pynput 1.8.2` (`pynput.keyboard._xorg`, XRecord) |
| 리스너 기동 | `keyboard.Listener.start()` 정상, `is_alive() True` |
| 합성 키 주입 | `pynput.Controller`(XTest)로 주입한 키는 리스너에 **각 이벤트가 2회씩** 도착(WSLg Xwayland 특성으로 추정). 홀드 키(방향키·Shift·Ctrl)와 Enter/Esc 는 정상 반영, **Space(토글)은 2회 토글되어 무효** |
| 실제 키보드 | **사용자 확인 필요** — 물리 키보드 이벤트가 1회만 도착하는지 `scripts/check_keyboard.py` 로 확인 (README §5) |

## 6. Python

| 항목 | 확인값 |
| --- | --- |
| 기존 환경 관리자 | conda/uv/pyenv 없음. 시스템 `python3.12.3`, `~/venvs/mujoco`(mujoco 3.12.0, ROS용) 존재 — 건드리지 않음 |
| ROS 간섭 | `~/.bashrc`가 ROS Jazzy `env.sh`를 source → 모든 셸에 `PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages` 주입. `scripts/env.sh`에서 `unset PYTHONPATH` 처리 |
| 프로젝트 venv | `~/project/hil-serl/.venv`, Python 3.12.3 (lerobot 0.6.1 이 `requires-python >=3.12`) |
| sudo | 비밀번호 필요 → apt 설치 불가. 필요한 시스템 라이브러리(libglfw3, libgl1, libegl1, ffmpeg 6.1.1)는 이미 설치되어 있어 추가 설치 없이 진행 |

## 7. 고정한 버전

| 구성 요소 | 버전 / 커밋 | 선택 근거 |
| --- | --- | --- |
| lerobot | **v0.6.1** (`7e241bd630a3719a56157a497ce5d08f244784f1`, 2026-08-03) | 최신 릴리스 태그. `main`(89236ea)은 매일 변하므로 태그 고정. hilserl_sim 문서와 `lerobot.rl.{actor,learner,gym_manipulator}` 모듈이 이 태그에 존재 |
| gym-hil | **0.1.14** (`f89b3e5556798c44ef95083d92b670afa89d110a`) | lerobot v0.6.1 의 `hilserl` extra 가 `gym-hil>=0.1.14,<0.2.0` 을 요구. PyPI 0.1.14 = 저장소 HEAD 와 동일 |
| torch / torchvision | 2.11.0+cpu / 0.26.0+cpu | lerobot 제약 `torch>=2.7,<2.12`, `torchvision<0.27` 안의 최신 CPU 휠. CUDA 휠 불필요 |
| torchcodec | 0.11.1+cpu | torch 2.11 과 짝 |
| mujoco | 3.8.1 | gym-hil 제약 `mujoco>=2.3.7,<3.9.0` |
| numpy | 2.2.6 | lerobot 제약 `<2.3` |
| 데이터셋 | `lilkm/pick_cube_franka_panda_30` revision `f80a680a117d0c8573ab453e3fd9e792e308dfa4` (2026-05-09) | 공식 train_config.json 이 참조. codebase v3.0 |
| 공식 설정 출처 | `lerobot/config_examples` sha `4803f5f379568ddc77763f5a4dd7cd56973d6a10` (2026-05-10), `rl/gym_hil/{train_config,env_config}.json` | |

전체 pip 목록: `docs/requirements-lock.txt` (`pip freeze --exclude-editable`).

## 8. 저장 위치

| 내용 | 실제 경로 (모두 WSL ext4) |
| --- | --- |
| 코드 | `~/project/hil-serl/third_party/{lerobot,gym-hil}` (editable install) |
| venv | `~/project/hil-serl/.venv` (약 1.4 GB) |
| HF 캐시 | `~/project/hil-serl/cache/huggingface` (`HF_HOME`), lerobot 데이터셋은 `cache/huggingface/lerobot/hub/...` (`HF_LEROBOT_HOME`) |
| 로그·체크포인트 | `~/project/hil-serl/outputs/train/<job>/` |
| 직접 수집 데이터 | `~/project/hil-serl/data/` |

`/mnt/c` 를 경유하는 파일은 PRD 문서 원본만이다. 프로젝트 파일은 `/mnt/c` 로 심볼릭 링크하지 않았다.
