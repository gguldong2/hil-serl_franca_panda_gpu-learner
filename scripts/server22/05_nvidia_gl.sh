#!/usr/bin/env bash
# [NVIDIA EGL 라이브러리] compute-only 컨테이너에서 MUJOCO_GL=egl 을 GPU 로 돌리기 위한 것.
#
# 증상: 03_verify.sh 2번에서 GL_RENDERER 가 llvmpipe (카메라 렌더 ~340 ms/step).
# 원인: 컨테이너에 libcuda/libnvidia-ml 같은 compute 라이브러리만 있고 그래픽 라이브러리
#       (libEGL_nvidia, libnvidia-eglcore ...)가 없다. NVIDIA_DRIVER_CAPABILITIES 에
#       graphics 가 없는 이미지에서 흔하다. 컨테이너를 다시 띄울 수 없을 때 쓰는 우회책.
# 방법: 커널 모듈과 **같은 버전**의 드라이버 런파일을 받아 압축만 풀고(설치 아님) GL 라이브러리를
#       nvidia-gl/ 에 두고 EGL ICD 를 그쪽으로 돌린다. scripts/env.sh 가 자동으로 잡는다.
# 주의: 호스트 드라이버가 올라가면(= nvidia-smi 버전이 바뀌면) 반드시 다시 실행해야 한다.
#       버전이 어긋나면 EGL 초기화가 실패하고 다시 llvmpipe 로 폴백한다.
#
# 사용법: bash scripts/server22/05_nvidia_gl.sh [--force]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1

command -v nvidia-smi >/dev/null || { echo "[05] nvidia-smi 없음 - GPU 없는 서버에서는 필요 없다"; exit 0; }
VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '[:space:]')"
KVER="$(awk '{print $8}' /proc/driver/nvidia/version 2>/dev/null || echo "$VER")"
[ -n "$KVER" ] && VER="$KVER"
echo "[05] 커널 드라이버 버전: $VER"

if [ -f "nvidia-gl/libEGL_nvidia.so.$VER" ] && [ "$FORCE" = 0 ]; then
  echo "[05] nvidia-gl/ 이 이미 $VER - 할 일 없음 (--force 로 재설치)"; exit 0
fi

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
RUN="NVIDIA-Linux-x86_64-$VER.run"
echo "[05] 런파일 내려받기 (약 340 MB)"
ok=0
for url in "https://us.download.nvidia.com/tesla/$VER/$RUN" \
           "https://us.download.nvidia.com/XFree86/Linux-x86_64/$VER/$RUN"; do
  if curl -fsSL --retry 3 -o "$WORK/$RUN" "$url"; then echo "  from $url"; ok=1; break; fi
done
[ "$ok" = 1 ] || { echo "[05] $VER 런파일을 찾지 못했다. https://www.nvidia.com/en-us/drivers/ 에서 직접 받아 $WORK/$RUN 로 두고 다시 실행"; exit 1; }

echo "[05] 압축 해제 (설치 아님)"
sh "$WORK/$RUN" -x --target "$WORK/nvx" >/dev/null

mkdir -p nvidia-gl && rm -f nvidia-gl/*.so.* nvidia-gl/nvidia_egl_icd.json
for n in libEGL_nvidia libnvidia-eglcore libnvidia-glcore libnvidia-glsi libnvidia-tls libnvidia-glvkspirv libnvidia-gpucomp; do
  f="$WORK/nvx/$n.so.$VER"
  [ -f "$f" ] || { echo "  (없음, 건너뜀) $n"; continue; }
  cp "$f" nvidia-gl/ && ln -sf "$n.so.$VER" "nvidia-gl/$n.so.0"
done
cat > nvidia-gl/nvidia_egl_icd.json <<JSON
{
    "file_format_version" : "1.0.0",
    "ICD" : { "library_path" : "$ROOT/nvidia-gl/libEGL_nvidia.so.0" }
}
JSON
echo "[05] 설치됨: $(du -sh nvidia-gl | cut -f1)  ($VER)"
echo "[05] 확인:  source scripts/env.sh && MUJOCO_GL=egl python -c \"import mujoco,gym_hil,os;from OpenGL import GL;m=mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(gym_hil.__file__),'assets','scene.xml'));r=mujoco.Renderer(m,64,64);print(GL.glGetString(GL.GL_RENDERER).decode())\""
