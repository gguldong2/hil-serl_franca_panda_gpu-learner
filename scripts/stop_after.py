#!/usr/bin/env python3
"""에피소드 N 개를 채우면 학습을 깔끔하게 멈추고 기록을 남긴다.

lerobot 은 "몇 에피소드까지" 라는 종료 조건이 없다(actor 는 online_steps 라는 환경 스텝
수로만 끝난다). 에피소드 길이가 성공/실패에 따라 달라지므로 실험끼리 조건을 맞추려면
에피소드 수로 끊는 편이 낫다. 이 스크립트가 그 역할을 한다.

  python scripts/stop_after.py --episodes 100
  python scripts/stop_after.py --episodes 100 --run outputs/train/place_v2
  python scripts/stop_after.py --episodes 100 --no-stop     # 세기만 하고 안 멈춘다

N 에 도달하면 순서대로:
  1) report.html 을 갱신한다
  2) <run>/STOP_NOW 를 만든다  -> learner 가 체크포인트를 남기고 스스로 끝난다
  3) actor 에 SIGINT 를 보낸다 -> 시뮬레이터가 정리 종료한다

학습을 시작한 뒤 다른 칸에서 띄워 두면 된다. 학습에는 아무 영향이 없다.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, required=True, help="이만큼 끝나면 멈춘다")
p.add_argument("--run", default="outputs/train/panda_place_gpu")
p.add_argument("--no-stop", dest="stop", action="store_false", help="세기만 하고 멈추지 않는다")
p.add_argument("--poll", type=float, default=3.0, help="몇 초마다 확인할지")
args = p.parse_args()

RUN = Path(args.run)
METRICS = RUN / "logs" / "learner_metrics.jsonl"


def read_episodes():
    """끝난 에피소드들의 (보상, 개입률) 목록. learner_metrics.jsonl 이 원본 기록이다."""
    if not METRICS.exists():
        return []
    out = []
    with METRICS.open(errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") != "interaction":
                continue
            rew = r.get("Episodic reward")
            if isinstance(rew, (int, float)):
                iv = r.get("Intervention rate")
                out.append((float(rew), float(iv) if isinstance(iv, (int, float)) else 0.0))
    return out


def find_actor_pid():
    try:
        r = subprocess.run(
            ["pgrep", "-f", f"lerobot.rl.actor.*{RUN.name}_actor"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [int(x) for x in r.stdout.split() if x.isdigit()]
        return pids[0] if pids else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def bar(done, total, width=30):
    k = int(width * min(done / total, 1.0))
    return "#" * k + "." * (width - k)


def main() -> int:
    print("=" * 62)
    print(f"  에피소드 {args.episodes}개를 채우면 멈춘다")
    print("=" * 62)
    print(f"  런     : {RUN}")
    print(f"  기록   : {METRICS}")
    print(f"  멈추기 : {'예' if args.stop else '아니오 (세기만 한다)'}")
    print()
    if not METRICS.exists():
        print("  아직 기록 파일이 없다. learner 가 뜨면 생긴다. 기다린다...")

    last = -1
    while True:
        eps = read_episodes()
        n = len(eps)
        if n != last:
            last = n
            ok = sum(1 for r, _ in eps if r > 0.5)
            rec = eps[-20:]
            rok = sum(1 for r, _ in rec if r > 0.5)
            riv = sum(i for _, i in rec) / len(rec) if rec else 0.0
            print(
                f"  [{bar(n, args.episodes)}] {n}/{args.episodes}   "
                f"성공 {ok} ({100 * ok // max(n, 1)}%)   "
                f"최근{len(rec)}판 {100 * rok // max(len(rec), 1)}%   "
                f"개입 {100 * riv:.0f}%",
                flush=True,
            )
        if n >= args.episodes:
            break
        time.sleep(args.poll)

    print()
    print(f"  에피소드 {args.episodes}개 도달. 마무리한다.")

    # 1) 리포트
    try:
        subprocess.run([sys.executable, "scripts/report.py", "--run", str(RUN)], timeout=180)
    except (OSError, subprocess.TimeoutExpired) as e:  # noqa: BLE001
        print(f"  [경고] 리포트 생성 실패: {e}")

    if not args.stop:
        print("  --no-stop 이라 멈추지 않는다.")
        return 0

    # 2) learner 에게 저장 후 종료 요청
    (RUN / "STOP_NOW").touch()
    print(f"  learner 에 저장 후 종료를 요청했다 ({RUN}/STOP_NOW)")

    # 3) actor 정리 종료
    pid = find_actor_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGINT)
            print(f"  actor(PID {pid}) 에 종료 신호를 보냈다.")
        except OSError as e:  # noqa: BLE001
            print(f"  [경고] actor 종료 실패: {e}")
    else:
        print("  actor 를 찾지 못했다. 그 칸에서 직접 Ctrl+C 해라.")

    print()
    print("  두 프로세스가 끝나면:")
    print(f"    리포트 : {RUN}/report.html")
    print(f"    체크포인트: {RUN}/checkpoints/")
    print("  다른 조건으로 다음 실험:  bash scripts/train.sh new <이름>")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  종료. (학습은 계속 돌고 있다)")
