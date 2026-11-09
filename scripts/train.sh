#!/usr/bin/env bash
# 학습 시작 / 이어서 하기 / 진행 확인. 새로 시작인지 이어서인지 알아서 판단한다.
#
#   bash scripts/train.sh status            # 지금 어디까지 왔나 (언제든)
#   bash scripts/train.sh learner           # 칸 A — 먼저 띄운다
#   bash scripts/train.sh actor             # 칸 B — learner 준비되면
#   bash scripts/train.sh actor --window    # MuJoCo 창도 같이
#   bash scripts/train.sh restart           # 처음부터 다시 (지운다. 물어본다)
#
#   bash scripts/train.sh pause             # 시뮬레이터 일시정지 (학습·연결은 살아 있다)
#   bash scripts/train.sh resume            # 재개
#   bash scripts/train.sh save              # 지금 체크포인트를 남긴다
#   bash scripts/train.sh stop              # 체크포인트를 남기고 learner 를 정상 종료
#   bash scripts/train.sh new <이름>        # 다른 이름으로 새 학습 설정을 만든다
#                                          # online_steps 는 1,000,000 으로 새로 잡는다.
#                                          # 바꾸려면 ONLINE_STEPS=200000 을 앞에 붙인다.
#
# 이어서 하기:  그냥 같은 명령을 다시 치면 된다. checkpoints/last 가 있으면 --resume=true 를 붙인다.
#
# 설정 파일은 CFG 로 바꿀 수 있다:
#   CFG=configs/train_config_place_smoke_gpu.json bash scripts/train.sh learner
#
# GPU 고르기 (이 서버에는 A100 이 두 장이다):
#   HIL_LEARNER_GPU=0 bash scripts/train.sh learner    # learner 를 0번에
#   HIL_ACTOR_GPU=1   bash scripts/train.sh actor      # actor 를 1번에
#   HIL_ACTOR_GPU=cpu bash scripts/train.sh actor      # actor 정책 추론을 CPU 로
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh >/dev/null 2>&1

CFG="${CFG:-configs/train_config_panda_place_gpu.json}"
CMD="${1:-status}"; shift || true

OUT="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['output_dir'])" "$CFG" 2>/dev/null)"
if [ -z "$OUT" ]; then
  echo "[train] 설정을 읽지 못했다: $CFG"
  if [ ! -f "$CFG" ]; then
    echo "        그 파일이 없다. CFG= 경로를 확인한다 (tmux 세션에 CFG 를 넣었다면: echo \$CFG)"
  elif ! .venv/bin/python -c '' 2>/dev/null; then
    echo "        venv 의 Python 이 돌지 않는다 — pod/컨테이너가 새로 떠 시스템 Python 3.12 가 지워졌을 때 이렇다."
    echo "        고치기:  bash scripts/preflight.sh --fix   (필요하면 01_install_system.sh 를 다시 돌린다)"
  fi
  exit 1
fi
LAST="$OUT/checkpoints/last"
PAUSE_FILE="${HIL_PAUSE_FILE:-/dev/shm/hil_pause}"

# GPU 선택. 숫자를 주면 그 장만 보이게 하고, cpu 를 주면 정책을 CPU 에 올린다.
#   측정: learner 가 GPU 0 에서 4.2 GB / 80 GB 를 쓴다. 둘을 갈라 놓으면 서로 간섭이 없다.
DEVICE_OVERRIDE=()
apply_gpu() {   # $1 = HIL_LEARNER_GPU 또는 HIL_ACTOR_GPU 값
  local g="${1:-}"
  [ -z "$g" ] && return 0
  if [ "$g" = "cpu" ]; then
    unset CUDA_VISIBLE_DEVICES
    DEVICE_OVERRIDE=(--policy.device=cpu)
    echo "[train] 정책을 CPU 에 올린다."
  else
    export CUDA_VISIBLE_DEVICES="$g"
    echo "[train] CUDA_VISIBLE_DEVICES=$g"
  fi
}
STATUS_FILE="$OUT/STATUS.txt"

have_checkpoint() { [ -e "$LAST/pretrained_model/train_config.json" ]; }

# 지금 돌고 있는 learner 가 어떤 설정을 쓰는지 알아낸다 (없으면 빈 문자열).
running_learner_cfg() {
  # 주의: pgrep -f 는 "이 명령 자체"도 잡는다. 실제 python 프로세스만 보도록
  # /proc/<pid>/comm 이 python 인지 확인하고, 인자를 줄 단위로 쪼개 뽑는다.
  local pid comm args
  for pid in /proc/[0-9]*; do
    pid=${pid#/proc/}
    [ "$pid" = "$$" ] && continue
    comm=$(cat "/proc/$pid/comm" 2>/dev/null) || continue
    case "$comm" in python*) ;; *) continue;; esac
    args=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null) || continue
    case "$args" in
      *lerobot.rl.learner*)
        printf '%s\n' "$args" | sed -n 's/^--config_path=//p' | head -1
        return 0;;
    esac
  done
  return 1
}

write_status() {
  # 학습 폴더가 아직 없으면 만들지 않는다 — 만들어 버리면 learner 가 "이미 있다"며 거부한다.
  local sink=/dev/null
  [ -d "$OUT" ] && sink="$STATUS_FILE"
  python - "$OUT" "$CFG" <<'PYEND' | tee "$sink"
import json, os, sys, time, glob
out, cfg_path = sys.argv[1], sys.argv[2]
def j(p, d=None):
    try:
        with open(p) as f: return json.load(f)
    except Exception: return d

last = os.path.join(out, "checkpoints", "last")
ts = j(os.path.join(last, "training_state", "training_step.json"), {})
cfg = j(cfg_path, {})
# 진행률의 기준은 cfg["steps"] 가 아니라 policy.online_steps 다.
# cfg["steps"] 는 지도학습용 설정에서 물려받은 값이고 RL learner 루프는 그것을 쓰지 않는다
# (learner 는 shutdown 신호가 올 때까지 계속 돈다). 실제로 학습을 끝내는 것은
# actor 의 `for interaction_step in range(cfg.policy.online_steps)` 다.
total = (cfg.get("policy") or {}).get("online_steps")

task = (cfg.get("env") or {}).get("task") or "?"
# 로봇이 둘 이상이 되면 "지금 보고 있는 게 어느 실험이냐" 가 제일 헷갈린다.
# 태스크 id 앞부분이 곧 로봇이다 (PandaPlaceCube... -> panda, UR5ePlaceCube... -> ur5e).
import re as _re
_m = _re.match(r"(UR5e|Panda|[A-Z][a-z0-9]*)", task)
robot = _m.group(1).lower() if _m else "?"
print("=" * 60)
print(f"  로봇 {robot}   ·   {task}")
print(f"  {out}")
print("=" * 60)
ckpt_inter = 0   # 체크포인트가 아직 없을 때(새 학습 첫 2,000 스텝 전)도 아래 inter 계산이 이 이름을 쓴다
if not ts:
    print("  아직 체크포인트가 없다 — 새로 시작하는 상태다.")
else:
    step = ts.get("step", 0)
    print(f"  학습 스텝      : {step:,}   (최적화 횟수. 상한 없음 — 멈출 때까지 돈다)")
    ckpt_inter = ts.get("interaction_step", 0)
    real = os.path.realpath(last)
    print(f"  마지막 체크포인트: {os.path.basename(real)}"
          f"   ({time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(real)))})")


m = os.path.join(out, "logs", "learner_metrics.jsonl")
eps, iv = [], []
live_inter = None
live_on = live_off = None
_inter_base = _prev_inter = _restarts = 0
if os.path.exists(m):
    with open(m, errors="replace") as f:
        for line in f:
            try: r = json.loads(line)
            except Exception: continue
            if r.get("kind") == "interaction":
                v = r.get("Episodic reward")
                if isinstance(v, (int, float)): eps.append(float(v))
                v = r.get("Intervention rate")
                if isinstance(v, (int, float)): iv.append(float(v))
                v = r.get("Interaction step")
                if isinstance(v, (int, float)):
                    # actor 를 재시작하면 이 값이 0 부터 다시 센다. 되돌아간 지점마다
                    # 직전 최대값을 더해 누적으로 만든다.
                    v = int(v)
                    if v < _prev_inter:
                        _inter_base += _prev_inter
                        _restarts += 1
                    _prev_inter = v
                    live_inter = _inter_base + v
            elif r.get("kind") == "train":
                v = r.get("replay_buffer_size")
                if isinstance(v, (int, float)): live_on = int(v)
                v = r.get("offline_replay_buffer_size")
                if isinstance(v, (int, float)): live_off = int(v)
# 체크포인트의 interaction_step 은 저장 시점에 메시지가 없으면 0 으로 남는다(상류 특성).
# 지표 파일의 최신값이 있으면 그쪽을 쓴다.
inter = live_inter if live_inter is not None else ckpt_inter
# 종료 조건은 "지금 도는 actor" 가 online_steps 를 채우는 것이다. 재시작하면 그 카운터는
# 0 부터 다시 시작하므로, 누적치와 현재 세션치를 따로 보여야 오해가 없다.
cur = _prev_inter

# 버퍼 크기는 지표 파일의 최신값이 정확하다. 디스크의 dataset/ 는 체크포인트 시점 스냅샷이고
# 저장 중에 읽으면 덜 쓰인 값이 보인다.
if live_on is not None:
    print(f"  온라인 버퍼    : {live_on:,} 전이")
if live_off is not None:
    print(f"  시연+개입 버퍼 : {live_off:,} 전이")
note = f"  (actor 재시작 {_restarts}회)" if _restarts else ""
print(f"  상호작용 스텝  : {inter:,} 누적{note}")
if total:
    bar = int(30 * min(cur / total, 1.0))
    print(f"  현재 actor     : {cur:,} / {total:,} ({100*cur/total:.1f}%)  "
          f"[{'#'*bar}{'.'*(30-bar)}]")
    left = max(total - cur, 0)
    print(f"                   이 actor 가 끝나기까지 약 {left/20/3600:.1f} 시간")

if eps:
    ok = sum(1 for r in eps if r > 0.5)
    r20 = eps[-20:]; ok20 = sum(1 for r in r20 if r > 0.5)
    print(f"  에피소드       : {len(eps)}개 · 전체 성공 {ok} ({100*ok//len(eps)}%)"
          f" · 최근 {len(r20)}판 {100*ok20//len(r20)}%")
if iv:
    i20 = iv[-20:]
    print(f"  개입률         : 최근 {len(i20)}판 {100*sum(i20)/len(i20):.0f}%")

el = j(os.path.join(out, "elapsed.json"), {}).get("seconds")
if isinstance(el, (int, float)):
    el = int(el)
    print(f"  학습 시간      : {el//3600}시간 {el%3600//60}분  (실제로 돌아간 시간, 누적)")

ck = sorted(glob.glob(os.path.join(out, "checkpoints", "[0-9]*")))
print(f"  체크포인트     : {len(ck)}개")
print()
print("  이어서 하려면 (그냥 같은 명령을 다시 치면 된다):")
print("    bash scripts/train.sh learner")
print("    bash scripts/train.sh actor")
PYEND
  echo
  [ -d "$OUT" ] && echo "  (이 내용은 $STATUS_FILE 에도 저장돼 있다)"
}

case "$CMD" in
  status)
    write_status
    ;;

  learner)
# 띄우기 전 자동 점검 (0.05초). 통과하면 아무것도 찍지 않고, 막히면 이유를 보여주고 멈춘다.
# 이 서버는 컨테이너가 다시 뜰 때 드라이버 버전이 바뀌어 EGL 이 죽는 일이 되풀이된다.
# 건너뛰려면 앞에 HIL_SKIP_PREFLIGHT=1 (예: xvfb-run 으로 VNC 없이 돌릴 때).
if [ "${HIL_SKIP_PREFLIGHT:-}" != "1" ] && [ -x scripts/preflight.sh ]; then
  if ! _pf="$(bash scripts/preflight.sh --quick 2>&1)"; then
    printf '%s\n' "$_pf"
    echo
    echo "[preflight] 환경이 준비되지 않았다.  고치기:  bash scripts/preflight.sh --fix"
    echo "            (점검을 건너뛰려면 앞에 HIL_SKIP_PREFLIGHT=1)"
    exit 1
  fi
fi
    RESUME=()
    if have_checkpoint; then
      RESUME=(--resume=true)
      echo "[train] 체크포인트를 찾았다 -> 이어서 학습한다."
      python -c "
import json;d=json.load(open('$LAST/training_state/training_step.json'))
print(f\"         학습 스텝 {d['step']:,} · 상호작용 스텝 {d['interaction_step']:,} 부터\")" 2>/dev/null
    else
      echo "[train] 체크포인트가 없다 -> 처음부터 시작한다."
      if [ -d "$OUT" ]; then
        echo "[train] 그런데 $OUT 이 이미 있다. 체크포인트 없이 남은 찌꺼기 같다."
        echo "         지우고 시작하려면:  bash scripts/train.sh restart"
        exit 1
      fi
    fi
    apply_gpu "${HIL_LEARNER_GPU:-}"
    echo "[train] 설정: $CFG   출력: $OUT"
    echo
    exec python -u -m lerobot.rl.learner --config_path="$CFG" \
      "${RESUME[@]}" "${DEVICE_OVERRIDE[@]}" "$@"
    ;;

  actor)
# 띄우기 전 자동 점검 (0.05초). 통과하면 아무것도 찍지 않고, 막히면 이유를 보여주고 멈춘다.
# 이 서버는 컨테이너가 다시 뜰 때 드라이버 버전이 바뀌어 EGL 이 죽는 일이 되풀이된다.
# 건너뛰려면 앞에 HIL_SKIP_PREFLIGHT=1 (예: xvfb-run 으로 VNC 없이 돌릴 때).
if [ "${HIL_SKIP_PREFLIGHT:-}" != "1" ] && [ -x scripts/preflight.sh ]; then
  if ! _pf="$(bash scripts/preflight.sh --quick 2>&1)"; then
    printf '%s\n' "$_pf"
    echo
    echo "[preflight] 환경이 준비되지 않았다.  고치기:  bash scripts/preflight.sh --fix"
    echo "            (점검을 건너뛰려면 앞에 HIL_SKIP_PREFLIGHT=1)"
    exit 1
  fi
fi
    WINDOW=()
    if [ "${1:-}" = "--window" ]; then
      shift
      # 창 태스크는 설정의 태스크를 따른다 (UR5ePlaceCubeKeyboard-v0 -> UR5ePlaceCubeKeyboardWindow-v0).
      # 예전에는 Panda 창으로 박혀 있어서 UR5e 설정에 --window 를 주면 Panda 환경이 떴다(상태 18 vs 16).
      _task="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['env']['task'])" "$CFG")" || exit 1
      WINDOW=(--env.task="${_task/Keyboard-v0/KeyboardWindow-v0}")
    fi
    export DISPLAY="${DISPLAY:-:1}"
    export MUJOCO_GL="${HIL_ACTOR_GL:-egl}"
    # 화면 HUD 가 이 런의 기록을 이어받도록 런 폴더를 알려 준다.
    # (actor 프로세스는 자기 스텝을 0 부터 세므로 이게 없으면 "1번째 에피소드" 로 보인다)
    export HIL_RUN_DIR="$OUT"
    apply_gpu "${HIL_ACTOR_GPU:-}"
    # learner 와 설정이 다르면 학습은 gRPC 로 붙어 진행되지만, 화면·경과시간·로그가
    # 엉뚱한 폴더에 쌓여 "어느 실험이 도는지" 를 알 수 없게 된다. 실제로 그렇게 어긋난 적이 있다.
    _lcfg="$(running_learner_cfg 2>/dev/null || true)"
    if [ -n "$_lcfg" ] && [ "$_lcfg" != "$CFG" ]; then
      _lout=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['output_dir'])" "$_lcfg" 2>/dev/null)
      echo "╔══════════════════════════════════════════════════════════════════"
      echo "║  설정이 learner 와 다르다 — 이대로 띄우면 기록이 갈린다"
      echo "║"
      echo "║    돌고 있는 learner : $_lcfg"
      echo "║                       -> $_lout"
      echo "║    띄우려는 actor    : $CFG"
      echo "║                       -> $OUT"
      echo "║"
      echo "║  학습 자체는 gRPC 로 붙어 진행되지만, actor 쪽 기록"
      echo "║  (화면 카운터·경과시간·actor 로그)이 엉뚱한 폴더에 쌓인다."
      echo "║"
      echo "║  ── 이 칸에서 이대로 치면 된다 ──────────────────────────────"
      echo "║"
      echo "║    export CFG=$_lcfg"
      echo "║    bash scripts/train.sh actor --window"
      echo "║"
      echo "║  (확인)  bash scripts/train.sh status"
      echo "║  (화면)  다른 칸에서  bash scripts/dashboard.sh --host 0.0.0.0"
      echo "║"
      echo "║  반대로 이 actor 설정($CFG)으로 가고 싶다면"
      echo "║  learner 를 먼저 그 설정으로 다시 띄워야 한다."
      echo "╚══════════════════════════════════════════════════════════════════"
      if [ "${HIL_FORCE:-}" != "1" ]; then
        echo "[train] 중단했다. 그래도 강행하려면 앞에 HIL_FORCE=1 을 붙여라."
        exit 1
      fi
      echo "[train] HIL_FORCE=1 이라 그대로 진행한다."
    fi

    # actor 에 --resume=true 를 주면 안 된다. lerobot 의 파서가 resume 일 때 --config_path 를
    # "사전학습 정책 폴더"로 해석해서 configs/model.safetensors 를 찾다가 죽는다.
    # actor 는 어차피 가중치를 learner 에게서 gRPC 로 받으므로 resume 이 필요 없다.
    # 출력 폴더가 있으면 거부당하니(로그만 들어 있다) 로그를 따로 보관하고 폴더를 비운다.
    if [ -d "${OUT}_actor" ]; then
      ARCH="${OUT}_actor_prevlogs"
      mkdir -p "$ARCH"
      if [ -d "${OUT}_actor/logs" ]; then
        cp -a "${OUT}_actor/logs/." "$ARCH/" 2>/dev/null || true
      fi
      rm -rf "${OUT}_actor"
      echo "[train] 이전 actor 로그는 $ARCH 에 보관했다. (에피소드 기록의 원본은"
      echo "         learner 쪽 learner_metrics.jsonl 이라 그대로 남는다)"
    fi
    if [ -f "$OUT/logs/learner_metrics.jsonl" ]; then
      _n=$(grep -c '"interaction"' "$OUT/logs/learner_metrics.jsonl" 2>/dev/null || echo 0)
      echo "[train] === 이어서 하는 실험 === 지금까지 ${_n}판. 화면 카운터도 이어서 센다."
    else
      echo "[train] === 새 실험 === 0판부터 시작한다."
    fi
    echo "[train] actor  MUJOCO_GL=$MUJOCO_GL  DISPLAY=$DISPLAY  창=$([ ${#WINDOW[@]} -gt 0 ] && echo 있음 || echo 없음)"
    echo "[train] 화면: bash scripts/dashboard.sh -> http://localhost:8088/"
    echo
    exec python -u -m lerobot.rl.actor --config_path="$CFG" --output_dir="${OUT}_actor" \
      "${WINDOW[@]}" "${DEVICE_OVERRIDE[@]}" "$@"
    ;;

  pause)
    touch "$PAUSE_FILE"
    echo "[train] 일시정지했다.  ($PAUSE_FILE)"
    echo "         화면(8088)에 노란 테두리와 '|| 일시정지' 가 뜬다."
    echo "         재개:  bash scripts/train.sh resume"
    ;;

  resume)
    rm -f "$PAUSE_FILE"
    echo "[train] 재개했다."
    ;;

  save)
    if [ ! -d "$OUT" ]; then echo "[train] $OUT 이 없다. 학습이 돌고 있나?"; exit 1; fi
    touch "$OUT/SAVE_NOW"
    echo "[train] 저장을 요청했다. learner 가 다음 스텝에 체크포인트를 남긴다."
    echo "         확인:  bash scripts/train.sh status"
    ;;

  stop)
    if [ ! -d "$OUT" ]; then echo "[train] $OUT 이 없다. 학습이 돌고 있나?"; exit 1; fi
    touch "$OUT/STOP_NOW"
    echo "[train] 체크포인트를 남기고 learner 를 끝내라고 요청했다."
    echo "         learner 가 스스로 종료하면, actor 칸에서 Ctrl+C 로 정리한다."
    echo "         나중에 이어서:  bash scripts/train.sh learner"
    ;;

  new)
    NAME="${1:-}"
    if [ -z "$NAME" ]; then
      echo "사용법: bash scripts/train.sh new <이름>"
      echo "  예:   bash scripts/train.sh new place_v2"
      exit 1
    fi
    # 베낄 설정의 태스크에서 로봇을 뽑아 이름 앞에 붙인다.
    # 로봇이 둘 이상이 되면 폴더 이름만 보고 어느 로봇인지 알 수 있어야 한다.
    ROBOT="$(python -c "
import json,re,sys
t=(json.load(open(sys.argv[1])).get('env') or {}).get('task') or ''
m=re.match(r'(UR5e|Panda|[A-Z][a-z0-9]*)', t)
print(m.group(1).lower() if m else '')" "$CFG" 2>/dev/null)"
    case "$NAME" in
      "$ROBOT"_*) ;;                                   # 이미 붙어 있으면 그대로
      *) [ -n "$ROBOT" ] && NAME="${ROBOT}_${NAME}" ;;
    esac
    NEWCFG="configs/train_config_${NAME}.json"
    if [ -e "$NEWCFG" ]; then echo "[train] $NEWCFG 이 이미 있다."; exit 1; fi
    python - "$CFG" "$NEWCFG" "$NAME" <<'PYEND'
import json, os, sys
src, dst, name = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(src))
c["output_dir"] = f"outputs/train/{name}"
c["job_name"] = name
c["resume"] = False
# actor 는 online_steps 를 다 채우면 정상 종료한다(actor.py 의 policy loop 범위).
# 베낀 설정의 값을 그대로 물려받으면 — 스모크 설정은 500 이다 — 새 학습이 몇 분 만에
# 끝나 버린다. 그래서 새로 만들 때는 항상 기본값으로 덮어쓴다.
#   다른 값으로:  ONLINE_STEPS=200000 bash scripts/train.sh new <이름>
c["policy"]["online_steps"] = int(os.environ.get("ONLINE_STEPS") or 1000000)
json.dump(c, open(dst, "w"), indent=4)
print(f"[train] {dst} 를 만들었다. output_dir=outputs/train/{name}")
print(f"[train] online_steps={c['policy']['online_steps']:,} "
      f"(actor 가 한 번에 도는 상호작용 스텝. 다 돌면 actor 만 정상 종료한다)")
PYEND
    echo
    echo "  이 설정으로 돌리려면 앞에 CFG= 를 붙인다:"
    echo "    CFG=$NEWCFG bash scripts/train.sh learner"
    echo "    CFG=$NEWCFG bash scripts/train.sh actor --window"
    echo "    CFG=$NEWCFG bash scripts/train.sh status"
    echo
    echo "  기존 학습(outputs/train/$(basename "$OUT"))은 그대로 남아 있다."
    ;;

  restart)
    echo "[train] 다음을 지운다:"
    echo "    $OUT"
    echo "    ${OUT}_actor"
    if have_checkpoint; then
      python -c "
import json;d=json.load(open('$LAST/training_state/training_step.json'))
print(f\"  경고: 학습 스텝 {d['step']:,} 까지의 진행이 사라진다.\")" 2>/dev/null
    fi
    read -r -p "  정말 지울까? (yes 를 입력) " ans
    [ "$ans" = "yes" ] || { echo "  취소했다."; exit 1; }
    rm -rf "$OUT" "${OUT}_actor"
    echo "  지웠다. 이제:  bash scripts/train.sh learner"
    ;;

  *)
    # 머리말 주석을 그대로 사용법으로 쓴다. 줄 번호를 박아 두면 주석을 고칠 때마다
    # 어긋나므로(실제로 어긋난 적이 있다) set 줄 앞까지로 잡는다.
    sed -n '2,/^set -/p' "$0" | sed '$d;s/^# \?//'
    exit 1
    ;;
esac
