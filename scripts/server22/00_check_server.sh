#!/usr/bin/env bash
# [서버 사전 점검] Ubuntu 22.04 + NVIDIA GPU 네이티브/컨테이너 가정. 읽기 전용 — 아무것도 설치·변경하지 않는다.
# 사용법: bash scripts/server22/00_check_server.sh | tee outputs/server_check.log
set -uo pipefail
hr() { printf '\n== %s ==\n' "$1"; }

hr "OS"
grep -E "PRETTY_NAME|VERSION_ID" /etc/os-release; uname -r; ldd --version | head -1
if grep -qi microsoft /proc/version 2>/dev/null; then echo "환경: WSL"; else echo "환경: 네이티브 리눅스"; fi
[ -f /.dockerenv ] && echo "컨테이너: /.dockerenv 존재"
grep -qaE 'docker|containerd|kubepods|lxc|podman' /proc/1/cgroup 2>/dev/null && echo "컨테이너: cgroup 에서 감지"
echo "PID1: $(cat /proc/1/comm 2>/dev/null)  (systemd 가 아니면 서비스 자동 시작 불가 → VNC 는 수동 실행)"

hr "권한"
if [ "$(id -u)" = 0 ]; then echo "root 로 실행 중 (sudo 불필요)"; elif sudo -n true 2>/dev/null; then echo "sudo 사용 가능(비밀번호 없음)"; else echo "sudo 비밀번호 필요 또는 없음 → apt 설치 단계는 관리자에게 요청"; fi

hr "CPU / RAM / 디스크 / shm"
nproc; lscpu | grep -E "Model name" | head -1; free -h | head -2
df -h "$HOME" | tail -1; df -h /dev/shm 2>/dev/null | tail -1
echo "참고: replay buffer 는 전이당 약 393 KB(이미지 2장 float32). online 20000 + offline 5000 ≈ 9.8 GB RAM 상한"

hr "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
  echo "cu128 휠 요구 드라이버 ≥ 570.86 (lerobot pyproject 주석). 미달이면 02_setup_project.sh --cu126"
else echo "nvidia-smi 없음 → CPU 경로(--cpu)"; fi
echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-<unset>}   (EGL GPU 렌더에는 graphics 또는 all 필요)"
if ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 >/dev/null 2>&1 || ls /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null 2>&1; then
  echo "NVIDIA EGL 라이브러리 있음 → MUJOCO_GL=egl 로 GPU 렌더 가능(03_verify 에서 실제 확인)"
else echo "NVIDIA EGL 라이브러리 없음 → MUJOCO_GL=egl 은 Mesa 소프트웨어(llvmpipe)로 폴백(느림). 컨테이너 실행 옵션에 graphics 캡 추가 필요"; fi

hr "디스플레이"
echo "DISPLAY=${DISPLAY:-<unset>}  XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-<unset>}"
for c in Xvfb Xvnc vncserver websockify xdpyinfo glxinfo eglinfo xwininfo; do command -v $c >/dev/null 2>&1 && echo "  $c: $(command -v $c)"; done
[ -z "${DISPLAY:-}" ] && echo "DISPLAY 없음 → 사람 개입(뷰어+키보드)에는 04_start_vnc.sh 로 VNC 디스플레이 필요. 자동 검증만이면 xvfb-run 사용"

hr "Python"
for p in python3 python3.12 python3.10 uv conda; do command -v $p >/dev/null 2>&1 && printf "  %-10s %s\n" "$p" "$($p --version 2>&1 | head -1)"; done
command -v python3.12 >/dev/null 2>&1 || echo "python3.12 없음 → 01_install_system.sh 가 deadsnakes 또는 uv 로 설치 (lerobot v0.6.1 은 >=3.12 필수)"

hr "libstdc++ / ffmpeg"
n=$(strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 2>/dev/null | grep -c "GLIBCXX_3.4.30"); echo "GLIBCXX_3.4.30: ${n:-0} (1 이상이면 manylinux_2_28 휠 OK)"
ffmpeg -version 2>/dev/null | head -1 || echo "ffmpeg 없음 (01_install_system.sh 가 설치. torchcodec 은 FFmpeg 4~8 지원)"

hr "루프백 네트워크 (actor↔learner gRPC)"
python3 - <<'PY'
import socket, threading, time
def serve(fam, addr):
    s=socket.socket(fam, socket.SOCK_STREAM); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1); s.bind(addr); s.listen(1); s.settimeout(4)
    try:
        c,_=s.accept(); c.sendall(b"hi"); c.close()
    except Exception: pass
    s.close()
for fam, addr, name in [(socket.AF_INET,("127.0.0.1",50071),"IPv4 127.0.0.1"),(socket.AF_INET6,("::1",50072,0,0),"IPv6 ::1")]:
    try:
        th=threading.Thread(target=serve,args=(fam,addr)); th.start(); time.sleep(0.3)
        c=socket.socket(fam, socket.SOCK_STREAM); c.settimeout(3); c.connect(addr); print(f"  {name:16s} OK"); c.close(); th.join()
    except Exception as e: print(f"  {name:16s} FAIL: {e}  (둘 다 실패하면 learner_host 를 바꿔야 함)")
PY
echo; echo "점검 끝. 다음: bash scripts/server22/01_install_system.sh [--with-vnc]"
