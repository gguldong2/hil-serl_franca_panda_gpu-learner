#!/usr/bin/env bash
# [VNC 디스플레이] TigerVNC(:1) + fluxbox + noVNC(6080). 컨테이너/헤드리스 서버에서 뷰어 창과 키보드 개입을 쓰기 위한 것.
# 사용법: bash scripts/server22/04_start_vnc.sh [:1]     종료: bash scripts/server22/04_start_vnc.sh --stop
# 접속: 로컬 PC 에서  ssh -L 6080:localhost:6080 <user>@<server>  후 브라우저 http://localhost:6080/vnc.html
set -euo pipefail
DISP="${1:-:1}"
if [ "$DISP" = "--stop" ]; then vncserver -kill :1 2>/dev/null || true; pkill -f "websockify.*6080" 2>/dev/null || true; echo "stopped"; exit 0; fi
GEOM="${GEOM:-1600x900}"; NOVNC_PORT="${NOVNC_PORT:-6080}"; VNC_PORT=$((5900 + ${DISP#:}))
command -v vncserver >/dev/null || { echo "tigervnc 없음: 01_install_system.sh --with-vnc"; exit 1; }
mkdir -p ~/.vnc
[ -f ~/.vnc/passwd ] || { echo "VNC 비밀번호를 설정합니다(6자 이상):"; vncpasswd; }
cat > ~/.vnc/xstartup <<'XS'
#!/bin/sh
unset SESSION_MANAGER DBUS_SESSION_BUS_ADDRESS
xsetroot -solid grey 2>/dev/null || true
exec fluxbox
XS
chmod +x ~/.vnc/xstartup
vncserver -kill "$DISP" 2>/dev/null || true
vncserver "$DISP" -geometry "$GEOM" -depth 24 -localhost yes -SecurityTypes VncAuth
pkill -f "websockify.*${NOVNC_PORT}" 2>/dev/null || true
nohup websockify --web=/usr/share/novnc "localhost:$NOVNC_PORT" "localhost:$VNC_PORT" > ~/.vnc/novnc.log 2>&1 &
sleep 1
echo "VNC $DISP (port $VNC_PORT, localhost only) + noVNC http://localhost:$NOVNC_PORT/vnc.html"
echo "이 셸과 actor 를 띄울 셸에서:  export DISPLAY=$DISP"
echo "브라우저 접속 후 xterm 이 필요하면 fluxbox 바탕화면 우클릭 메뉴 → xterm"
