#!/usr/bin/env bash
# [프로젝트 설치] 저장소 clone(고정 커밋) → venv(3.12) → torch → gym-hil → lerobot[hilserl] → 호환성 패치 → 버전 기록
# 번들(configs/ scripts/ docs/ README.md)이 풀린 프로젝트 루트에서 실행한다. 기본 루트: ~/project/hil-serl
# 사용법: bash scripts/server22/02_setup_project.sh [--cu126|--cu128|--cpu]
#
# 인자를 생략하면 GPU 를 보고 알아서 고른다. **고른 뒤 실제로 GPU 계산을 해 보고,
# 안 되면 다른 쪽으로 한 번 더 설치한다.** 그래도 안 되면 0 이 아닌 코드로 멈춘다.
# (예전에는 cuda=False 여도 "완료" 를 찍고 끝나서, 학습이 CPU 로 도는 줄 모를 수 있었다.)
#
#   compute capability 9.x 이하 (RTX 20/30/40, A100, H100 ...) -> cu126
#     cu126 휠에는 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90 이 들어 있다.
#     sm_89(RTX 4080/4090)는 같은 major 8 이라 sm_86 바이너리로 돈다.
#     CUDA minor version compatibility 덕분에 드라이버는 525.60.13 이상이면 된다.
#     (이 서버는 드라이버가 "CUDA 12.2" 라고 말하는데 12.6 런타임이 정상 동작한다.)
#   compute capability 10.0 이상 (Blackwell, RTX 50xx / B200) -> cu128
#     sm_100 / sm_120 은 cu126 휠에 없다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
FLAVOR=auto
case "${1:-}" in --cu128) FLAVOR=cu128 ;; --cu126) FLAVOR=cu126 ;; --cpu) FLAVOR=cpu ;; "") ;; *) echo "옵션: --cu126|--cu128|--cpu"; exit 2 ;; esac
if [ "$FLAVOR" = auto ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d " ")"
    CC_MAJOR="${CC%%.*}"
    if [ -n "$CC_MAJOR" ] && [ "$CC_MAJOR" -ge 10 ] 2>/dev/null; then FLAVOR=cu128; else FLAVOR=cu126; fi
    echo "[02] GPU compute capability=${CC:-알수없음} -> $FLAVOR"
  else
    FLAVOR=cpu
  fi
fi
LEROBOT_TAG=v0.6.1          # 7e241bd630a3719a56157a497ce5d08f244784f1
GYMHIL_SHA=f89b3e5556798c44ef95083d92b670afa89d110a   # = PyPI gym-hil 0.1.14
TORCH_VER=2.11.0; TV_VER=0.26.0
echo "[02] root=$ROOT torch flavor=$FLAVOR"

mkdir -p configs scripts docs outputs data cache third_party
unset PYTHONPATH

echo "[02] 저장소 (고정 버전)"
[ -d third_party/lerobot/.git ] || git clone --depth 1 --branch "$LEROBOT_TAG" https://github.com/huggingface/lerobot.git third_party/lerobot
if [ ! -d third_party/gym-hil/.git ]; then
  git clone https://github.com/huggingface/gym-hil.git third_party/gym-hil
  git -C third_party/gym-hil checkout -q "$GYMHIL_SHA"
fi
echo "  lerobot: $(git -C third_party/lerobot rev-parse HEAD)"; echo "  gym-hil: $(git -C third_party/gym-hil rev-parse HEAD)"

echo "[02] venv"
if [ ! -x .venv/bin/python ]; then
  if command -v python3.12 >/dev/null 2>&1; then python3.12 -m venv .venv
  elif command -v uv >/dev/null 2>&1 || [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; uv venv --python 3.12 --seed .venv
  else echo "python3.12 도 uv 도 없음. 01_install_system.sh 먼저"; exit 1; fi
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python --version | grep -q "3.12" || { echo "venv 가 3.12 가 아님"; exit 1; }
pip install -q --upgrade pip wheel setuptools

install_torch() {   # $1 = cu126 | cu128 | cpu
  echo "[02] torch $TORCH_VER ($1)"
  pip install "torch==$TORCH_VER" "torchvision==$TV_VER" --index-url "https://download.pytorch.org/whl/$1"
}
gpu_works() {   # is_available() 만으로는 모자라다. 실제 커널을 돌려 본다.
  python - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print("  torch import 실패:", e); sys.exit(1)
if not torch.cuda.is_available():
    print("  torch.cuda.is_available() == False"); sys.exit(1)
try:
    x = torch.randn(512, 512, device="cuda")
    v = float((x @ x).sum())
    if v != v:
        print("  GPU 계산 결과가 NaN"); sys.exit(1)
except Exception as e:
    print("  GPU 커널 실행 실패:", type(e).__name__, e); sys.exit(1)
cc = "".join(map(str, torch.cuda.get_device_capability(0)))
print(f"  OK  torch={torch.__version__}  cuda={torch.version.cuda}  "
      f"gpu={torch.cuda.get_device_name(0)}  sm_{cc}")
PY
}

install_torch "$FLAVOR"
if [ "$FLAVOR" != cpu ]; then
  if ! gpu_works; then
    case "$FLAVOR" in cu126) ALT=cu128 ;; cu128) ALT=cu126 ;; *) ALT="" ;; esac
    if [ -n "$ALT" ]; then
      echo "[02] !! $FLAVOR 로는 GPU 계산이 안 된다. $ALT 로 다시 설치해 본다."
      install_torch "$ALT"
      if gpu_works; then
        FLAVOR="$ALT"
      else
        echo "[02] !! cu126 / cu128 둘 다 GPU 계산에 실패했다. 확인할 것:"
        echo "       - nvidia-smi 가 도는가 (드라이버가 있는가)"
        echo "       - 드라이버가 525.60.13 이상인가:  nvidia-smi --query-gpu=driver_version --format=csv"
        echo "       - 컨테이너라면 --gpus all / NVIDIA_DRIVER_CAPABILITIES 가 있는가"
        echo "       GPU 없이 동작만 확인하려면:  bash $0 --cpu"
        exit 1
      fi
    fi
  fi
  echo "[02] GPU 계산 확인됨 (flavor=$FLAVOR)"
fi
echo "[02] gym-hil (editable, $GYMHIL_SHA)"
pip install -e third_party/gym-hil
echo "[02] lerobot[hilserl] (editable, $LEROBOT_TAG)"
pip install -e "third_party/lerobot[hilserl]"
pip check

echo "[02] 호환성 패치 (docs/patches)"
series_applied() {   # $1=저장소  $2..=패치(절대경로). 고정 커밋에 전부 적용한 결과와 지금 파일이 같은가
  local repo="$1" T f p rc=0; shift
  T=$(mktemp -d)
  git clone -q --shared --no-checkout "$repo" "$T/r" && git -C "$T/r" checkout -q "$(git -C "$repo" rev-parse HEAD)" || rc=1
  for p in "$@"; do [ $rc = 0 ] && { git -C "$T/r" apply "$p" 2>/dev/null || rc=1; }; done
  if [ $rc = 0 ]; then
    for f in $(sed -n 's|^+++ b/||p' "$@" | sort -u); do cmp -s "$T/r/$f" "$repo/$f" || { rc=1; break; }; done
  fi
  rm -rf "$T"; return $rc
}
apply_patches() {   # $1=저장소 디렉터리  $2=패치 glob
  local repo="$1" glob="$2" p all=()
  for p in $glob; do [ -e "$p" ] && all+=("$ROOT/$p"); done
  for p in $glob; do
    [ -e "$p" ] || continue
    if git -C "$repo" apply --check "$ROOT/$p" 2>/dev/null; then git -C "$repo" apply "$ROOT/$p"; echo "  적용: $p"
    elif git -C "$repo" apply --reverse --check "$ROOT/$p" 2>/dev/null; then echo "  이미 적용됨: $p"
    # 뒤 패치가 같은 파일을 또 고치면(gym-hil 0007 이 0002·0005 위에 얹힌다) 앞 패치는 혼자서는
    # 역적용 검사가 안 된다. 그때는 패치 전체를 고정 커밋에 적용한 결과와 지금 파일을 비교한다.
    elif series_applied "$repo" "${all[@]}"; then echo "  이미 적용됨: $p 부터 끝까지 (패치 전체 결과와 파일이 같다)"; return 0
    else echo "  !! 적용 실패: $p ($repo 버전이 고정 커밋이 아닌지 확인)"; exit 1; fi
  done
}
apply_patches third_party/lerobot "docs/patches/000*.patch"
# gym-hil 패치: 헤드리스 X11(VNC/Xvfb) 에서 종료 시 SIGSEGV 를 막는다. server22 에서 발견.
apply_patches third_party/gym-hil "docs/patches/gym-hil/*.patch"
# UR5e 메시(Menagerie 고정 커밋). 패치 0006 의 assets/ur5e.xml 이 참조한다.
bash scripts/server22/fetch_ur5e_assets.sh

# 설치 목록 기록. docs/requirements-lock.server22.txt 는 기준 서버(server22) 기록이다.
# RunPod 에서는 git 에 안 올라가는 outputs/ 에 쓴다 — pod 가 기준 기록을 덮어써 11-18 · 11-19 두 번 커밋됐다.
# (SSH 접속에는 RUNPOD_* 환경변수가 없을 수 있어 /workspace 경로로도 알아본다)
case "${RUNPOD_POD_ID:+runpod}:$ROOT" in
  runpod:*|*:/workspace/*) LOCK=outputs/requirements-lock.runpod.txt ;;
  *)                       LOCK=docs/requirements-lock.server22.txt ;;
esac
pip freeze --exclude-editable > "$LOCK"
echo "[02] 설치 목록: $LOCK"
EXPECT_GPU=0; [ "$FLAVOR" != cpu ] && EXPECT_GPU=1
EXPECT_GPU=$EXPECT_GPU python - <<'PY'
import os, sys
import torch, mujoco, gym_hil, lerobot, lerobot.rl.actor, lerobot.rl.learner
avail = torch.cuda.is_available()
print(f"[02] OK torch={torch.__version__} cuda={avail} mujoco={mujoco.__version__} lerobot={lerobot.__version__}")
if avail:
    print("     gpu:", torch.cuda.get_device_name(0))
# GPU 로 설치했는데 CUDA 가 안 보이면 여기서 멈춘다. 넘어가면 학습이 조용히 CPU 로 돈다.
if os.environ.get("EXPECT_GPU") == "1" and not avail:
    print("[02] !! GPU 설치인데 torch 가 CUDA 를 못 본다. 위 메시지 확인.")
    sys.exit(1)
PY
echo "[02] 완료 (torch flavor=$FLAVOR). 다음: bash scripts/server22/03_verify.sh"
