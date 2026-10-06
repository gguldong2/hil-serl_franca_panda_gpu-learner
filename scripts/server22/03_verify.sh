#!/usr/bin/env bash
# [실행 검증] CUDA 계산 → EGL 렌더러 확인 → 데이터셋 → 환경(headless) → 뷰어 환경(Xvfb 또는 DISPLAY) → 설정 파싱
# 사용법: bash scripts/server22/03_verify.sh [gpu|cpu]   (기본 gpu)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
MODE="${1:-gpu}"; CFG="configs/train_config_panda_keyboard_${MODE}.json"
if [ "$MODE" = gpu ]; then SMOKE_CFG="configs/train_config_smoke_gpu.json"; else SMOKE_CFG="configs/train_config_smoke.json"; fi
# shellcheck disable=SC1091
source scripts/env.sh
mkdir -p outputs; FAIL=0
step() { printf '\n==== %s ====\n' "$1"; }

step "1) PyTorch 장치 계산"
python - <<'PY' || FAIL=1
import torch, time, torchvision
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", dev, "| cuda available:", torch.cuda.is_available())
a = torch.randn(4096, 4096, device=dev); t = time.perf_counter(); c = a @ a
if dev == "cuda": torch.cuda.synchronize()
print(f"matmul 4096x4096 on {dev}: {1000*(time.perf_counter()-t):.1f} ms")
m = torchvision.models.resnet18().to(dev); x = torch.randn(8, 3, 128, 128, device=dev, requires_grad=True); m(x).sum().backward()
print("resnet18 fwd/bwd OK, grad finite:", bool(torch.isfinite(x.grad).all()))
if dev == "cuda": print("gpu:", torch.cuda.get_device_name(0), "| mem alloc MB:", torch.cuda.memory_allocated()//2**20)
PY

step "2) MuJoCo 오프스크린(EGL) 렌더러 — 어떤 GL 장치를 쓰는지"
MUJOCO_GL=egl python - <<'PY' || FAIL=1
import os, time, mujoco, gym_hil, numpy as np
from OpenGL import GL
xml = os.path.join(os.path.dirname(gym_hil.__file__), "assets", "scene.xml")
m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, height=128, width=128)
r.update_scene(d, camera="front"); img = r.render()
print("GL_RENDERER:", GL.glGetString(GL.GL_RENDERER).decode(), "| GL_VENDOR:", GL.glGetString(GL.GL_VENDOR).decode())
t = time.perf_counter()
for _ in range(10):
    for cam in ("front", "handcam_rgb"): r.update_scene(d, camera=cam); r.render()
print(f"2-camera render: {100*(time.perf_counter()-t):.1f} ms/step  (llvmpipe ≈ 340 ms, GPU ≈ 수 ms~30 ms) | frame mean={img.mean():.1f}")
r.close()
PY

step "3) 데이터셋 검증"
python scripts/check_dataset.py --train_config "$CFG" 2>&1 | grep -v -iE "warning|it/s\]" | tee outputs/check_dataset.log | tail -12 || FAIL=1

step "4) 환경 headless (EGL, 뷰어 없음)"
MUJOCO_GL=egl python scripts/check_env.py --task PandaPickCube-v0 --steps 30 2>&1 | grep -v -iE "^warning|deprecat|gym.logger|box.py" | tee outputs/check_env_headless.log | tail -8 || FAIL=1

step "5) 뷰어 포함 환경 (actor 와 같은 PandaPickCubeKeyboard-v0)"
if [ -n "${DISPLAY:-}" ]; then
  echo "DISPLAY=$DISPLAY 에 창을 띄움 (VNC 화면에서 'MuJoCo : Scene' 창 확인)"
  MUJOCO_GL=egl python scripts/check_env.py --task PandaPickCubeKeyboard-v0 --steps 20 2>&1 | grep -v -iE "^warning|deprecat|gym.logger|box.py|^  [A-Z]" | tail -6 || FAIL=1
else
  echo "DISPLAY 없음 → Xvfb 가상 화면에서 실행 (창은 보이지 않지만 actor 경로가 돌아가는지 확인)"
  MUJOCO_GL=egl xvfb-run -a -s "-screen 0 1280x800x24" python scripts/check_env.py --task PandaPickCubeKeyboard-v0 --steps 20 2>&1 | grep -v -iE "^warning|deprecat|gym.logger|box.py|^  [A-Z]" | tail -6 || FAIL=1
fi

step "6) 설정 파싱/검증"
python - "$CFG" "$SMOKE_CFG" <<'PY' || FAIL=1
import sys, draccus
from lerobot.rl.train_rl import TrainRLServerPipelineConfig
for path in sys.argv[1:3]:
    cfg = draccus.parse(TrainRLServerPipelineConfig, path, args=[])
    try:
        cfg.validate(); note = "OK"
    except FileExistsError:
        note = f"OK (파싱 정상. 단 output_dir={cfg.output_dir} 이 이미 있음 → 새 학습 전에 옮기거나 resume=true 로 이어가기)"
    print(f"{path}: device={cfg.policy.device} batch={cfg.batch_size} host={cfg.policy.actor_learner_config.learner_host} task={cfg.env.task} {note}")
PY

echo; if [ $FAIL = 0 ]; then echo "[03] 모두 통과. 다음: (사람 개입용) bash scripts/server22/04_start_vnc.sh  /  (자동 스모크) scripts/run_smoke_test.sh $SMOKE_CFG"; else echo "[03] 실패 항목 있음 — 위 로그 확인"; fi
exit $FAIL
