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
#
# 템플릿 torch 빌려 쓰기 (인자를 생략했을 때만): 시스템 python3.12 에 아래 REUSE_STACKS 의 torch ·
# torchvision 이 이미 있고 GPU 계산이 되면, venv 를 --system-site-packages 로 만들어 그 torch 를 쓴다.
# RunPod 은 /workspace 가 네트워크 디스크라 torch 를 venv 에 깔면 설치 ~10분, import 108초가 든다
# (템플릿 torch 는 로컬 디스크라 import 4초 — 11-19 L40S pod 실측). 조합이 안 맞으면 지금처럼 venv 에 깐다.
# 빌려 쓴 venv 에는 .venv/hil-template-torch 가 남고, preflight.sh 가 템플릿이 바뀌었는지 본다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
FLAVOR=auto
case "${1:-}" in --cu128) FLAVOR=cu128 ;; --cu126) FLAVOR=cu126 ;; --cpu) FLAVOR=cpu ;; "") ;; *) echo "옵션: --cu126|--cu128|--cpu"; exit 2 ;; esac
if [ "$FLAVOR" = auto ]; then AUTO=1; else AUTO=0; fi
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
TORCH_VER=2.11.0; TV_VER=0.26.0; TC_VER=0.11   # venv 에 직접 까는 기본 조합. torchcodec 은 torch 와 짝(0.11 ↔ 2.11)
# 템플릿(시스템 python3.12)의 torch 를 빌려 쓸 수 있는 조합 — 검증한 것만 둔다. torch · torchvision 이 맞으면
# 빌려 쓰고, torchcodec 은 셋째 값으로 고정한다(템플릿 것이 안 맞으면 venv 에 맞는 것을 깐다).
#   torch  torchvision  torchcodec
REUSE_STACKS=(
  "2.11.0 0.26.0 0.11"   # 기본 조합 그대로 (RunPod 이미지에는 없다 — 11-19 Docker Hub runpod/pytorch 태그)
  "2.9.1 0.24.1 0.9"     # RunPod PyTorch 2.9.1 템플릿 (runpod/pytorch:*-cu1281-torch291-ubuntu2404)
)
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

install_torch() {   # $1 = cu126 | cu128 | cpu
  echo "[02] torch $TORCH_VER ($1)"
  pip install "torch==$TORCH_VER" "torchvision==$TV_VER" --index-url "https://download.pytorch.org/whl/$1"
}
sys_torch_stack() {   # 시스템 python 의 "torch torchvision torchcodec 위치" (버전의 +cu 꼬리표는 뗀다, 없으면 -)
  "$SYS_PY" - <<'PY' 2>/dev/null
import importlib.metadata as md
v = {}
for n in ("torch", "torchvision", "torchcodec"):
    try: v[n] = md.version(n).split("+")[0]
    except md.PackageNotFoundError: v[n] = "-"
try: site = str(md.distribution("torch").locate_file("")).rstrip("/")
except md.PackageNotFoundError: site = "-"
print(v["torch"], v["torchvision"], v["torchcodec"], site)
PY
}
REUSE=""; SYS_SITE_DIR=""
pick_reuse() {   # 템플릿 torch 를 빌려 쓸 수 있으면 REUSE="torch tv tc" · SYS_SITE_DIR 를 채운다
  [ "$AUTO" = 1 ] && [ "$FLAVOR" != cpu ] && [ -n "$SYS_PY" ] || return 0
  local t tv tc site s
  read -r t tv tc site < <(sys_torch_stack) || return 0
  [ "$t" != "-" ] || return 0
  for s in "${REUSE_STACKS[@]}"; do
    case "$s" in "$t $tv "*) REUSE="$s" ;; esac
  done
  if [ -z "$REUSE" ]; then
    echo "[02] 템플릿 torch $t · torchvision $tv — 검증한 조합(REUSE_STACKS)이 아니라 쓰지 않는다. venv 에 torch $TORCH_VER 을 깐다."
    return 0
  fi
  echo "[02] 템플릿 torch $t · torchvision $tv · torchcodec $tc ($site) — GPU 계산 확인"
  if gpu_works "$SYS_PY"; then SYS_SITE_DIR="$site"
  else REUSE=""; echo "[02] !! 템플릿 torch 로 GPU 계산이 안 된다. venv 에 따로 깐다."; fi
}
gpu_works() {   # is_available() 만으로는 모자라다. 실제 커널을 돌려 본다.  $1 = python (기본: venv 의 python)
  "${1:-python}" - <<'PY'
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

echo "[02] venv"
# 시스템 python3.12. 이 셸에 venv 가 켜져 있어도 그 바탕 python 을 쓴다(그래야 템플릿 torch 를 제대로 본다).
SYS_PY="$(command -v python3.12 || true)"
[ -z "$SYS_PY" ] || SYS_PY="$("$SYS_PY" -c 'import sys; print(getattr(sys, "_base_executable", sys.executable))')"
if [ ! -x .venv/bin/python ]; then
  pick_reuse
  if [ -n "$REUSE" ]; then "$SYS_PY" -m venv --system-site-packages .venv
  elif [ -n "$SYS_PY" ]; then "$SYS_PY" -m venv .venv
  elif command -v uv >/dev/null 2>&1 || [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; uv venv --python 3.12 --seed .venv
  else echo "python3.12 도 uv 도 없음. 01_install_system.sh 먼저"; exit 1; fi
elif grep -qiE '^include-system-site-packages *= *true' .venv/pyvenv.cfg 2>/dev/null \
     && ! compgen -G ".venv/lib/python3.12/site-packages/torch-*.dist-info" >/dev/null; then
  # 템플릿 torch 를 빌려 쓰던 venv. pod 를 다른 템플릿으로 만들었을 수 있어 다시 본다.
  pick_reuse
  [ -n "$REUSE" ] || [ "$AUTO" = 0 ] || echo "[02] !! 이 venv 는 템플릿 torch 를 빌려 썼는데 지금 템플릿과 맞지 않는다 → venv 에 torch $TORCH_VER 을 깐다."
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python --version | grep -q "3.12" || { echo "venv 가 3.12 가 아님"; exit 1; }
if [ -n "$REUSE" ]; then read -r TORCH_VER TV_VER TC_VER <<<"$REUSE"; FLAVOR=template; fi
# torch 짝을 고정한다 — 이 뒤의 어떤 pip install 도 torch · torchvision · torchcodec 을 다른 버전으로 바꾸지 못한다.
# torchcodec 은 torch 를 의존성으로 적지 않아서, 고정하지 않으면 pip 가 torch 와 안 맞는 최신(<0.12)을 고른다.
printf 'torch==%s\ntorchvision==%s\ntorchcodec==%s.*\n' "$TORCH_VER" "$TV_VER" "$TC_VER" > .venv/torch-constraints.txt
export PIP_CONSTRAINT="$ROOT/.venv/torch-constraints.txt"
pip install -q --upgrade pip wheel setuptools

if [ -n "$REUSE" ]; then
  echo "[02] torch: 템플릿 것을 쓴다 — torch $TORCH_VER · torchvision $TV_VER ($SYS_SITE_DIR). 내려받지 않는다."
  gpu_works || { echo "[02] !! venv 안에서 템플릿 torch 로 GPU 계산이 안 된다."; exit 1; }
  echo "$TORCH_VER $TV_VER $TC_VER $SYS_SITE_DIR" > .venv/hil-template-torch   # preflight.sh 가 템플릿이 바뀌었는지 본다
else
  rm -f .venv/hil-template-torch
  install_torch "$FLAVOR"
fi
if [ "$FLAVOR" != cpu ] && [ "$FLAVOR" != template ]; then
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
if grep -qiE '^include-system-site-packages *= *true' .venv/pyvenv.cfg; then
  # 템플릿 패키지(Jupyter 등)도 보이는 venv 라, 템플릿 패키지끼리 · 템플릿 패키지와 venv 사이의 충돌도 나온다.
  # 프로젝트가 쓰는 것(venv 에 깐 것 + torch 짝)만 막고 나머지는 알리기만 한다.
  python - <<'PY'
import importlib.metadata as md, re, subprocess, sys, sysconfig
r = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
if r.returncode == 0:
    print(r.stdout.strip()); sys.exit(0)
venv_site = sysconfig.get_paths()["purelib"]
ours = {"torch", "torchvision", "torchcodec"}
bad = []
for line in r.stdout.splitlines():
    if not line.strip(): continue
    name = line.split()[0]
    try: loc = str(md.distribution(name).locate_file(""))
    except Exception: loc = ""
    if loc.startswith(venv_site) or re.sub(r"[-_.]+", "-", name).lower() in ours:
        bad.append(line); print("  !!", line)
    else:
        print("  (템플릿 패키지 — 프로젝트와 무관, 넘어간다)", line)
sys.exit(1 if bad else 0)
PY
else
  pip check
fi

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
  local repo="$1" glob="$2" p all=() i k n
  for p in $glob; do [ -e "$p" ] && all+=("$ROOT/$p"); done
  n=${#all[@]}
  for ((i = 0; i < n; i++)); do
    p="${all[$i]}"
    if git -C "$repo" apply --check "$p" 2>/dev/null; then git -C "$repo" apply "$p"; echo "  적용: ${p#"$ROOT"/}"
    elif git -C "$repo" apply --reverse --check "$p" 2>/dev/null; then echo "  이미 적용됨: ${p#"$ROOT"/}"
    else
      # 뒤 패치가 같은 파일을 또 고치면(gym-hil 0007 이 0002·0005 위에, 0008 이 0006 위에 얹힌다) 앞 패치는
      # 혼자서는 역적용 검사가 안 된다. 그때는 "고정 커밋에 앞에서부터 k 개를 적용한 결과" 가 지금 파일과 같은
      # 가장 긴 k 를 찾는다. 그 k 개는 이미 적용된 것이고, 나머지(나중에 추가된 패치)만 이어서 적용한다.
      # 전에는 전체(n 개)만 비교해서, 패치가 하나 늘면 이미 설치된 곳(RunPod pod 등)에서 02 가 멈췄다(2026-11-20).
      for ((k = n; k > i; k--)); do
        if series_applied "$repo" "${all[@]:0:k}"; then
          if [ "$k" = "$n" ]; then echo "  이미 적용됨: ${p#"$ROOT"/} 부터 끝까지 (패치 전체 결과와 파일이 같다)"; return 0; fi
          echo "  이미 적용됨: ${p#"$ROOT"/} 부터 $(basename "${all[$((k - 1))]}") 까지 (앞 ${k}개를 적용한 결과와 파일이 같다)"
          i=$((k - 1)); continue 2
        fi
      done
      echo "  !! 적용 실패: ${p#"$ROOT"/} ($repo 버전이 고정 커밋이 아닌지 확인)"; exit 1
    fi
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
import os, importlib.metadata as md
print(f"     torch 위치: {os.path.dirname(torch.__file__)}  torchcodec={md.version('torchcodec')}")
# GPU 로 설치했는데 CUDA 가 안 보이면 여기서 멈춘다. 넘어가면 학습이 조용히 CPU 로 돈다.
if os.environ.get("EXPECT_GPU") == "1" and not avail:
    print("[02] !! GPU 설치인데 torch 가 CUDA 를 못 본다. 위 메시지 확인.")
    sys.exit(1)
PY
echo "[02] 완료 (torch flavor=$FLAVOR). 다음: bash scripts/server22/03_verify.sh"
