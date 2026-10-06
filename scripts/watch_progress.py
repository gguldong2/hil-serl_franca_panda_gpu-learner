#!/usr/bin/env python
"""HIL-SERL 학습이 나아지고 있는지 한 화면에서 본다.

learner 가 남기는 outputs/<run>/logs/learner_metrics.jsonl 을 읽어
  - 에피소드별 성공 여부(Episodic reward > 0)
  - 개입률(Intervention rate)
  - 버퍼 크기(온라인 / 시연)
를 구간별로 묶어 보여준다. 개입률이 내려가면서 성공률이 올라가면 정책이 배우고 있는 것이다.

사용법:
  python scripts/watch_progress.py                          # 기본 run
  python scripts/watch_progress.py --run outputs/train/panda_place_gpu
  python scripts/watch_progress.py --follow                 # 20초마다 갱신 (Ctrl+C 로 종료)
  python scripts/watch_progress.py --bucket 1000            # 구간 크기(상호작용 스텝)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--run", default="outputs/train/panda_place_gpu", help="learner 의 output_dir")
p.add_argument("--bucket", type=int, default=2000, help="구간 크기 (상호작용 스텝)")
p.add_argument("--recent", type=int, default=20, help="'최근' 요약에 쓸 에피소드 수")
p.add_argument("--follow", action="store_true", help="주기적으로 갱신")
p.add_argument("--interval", type=float, default=20.0, help="--follow 갱신 주기(초)")
args = p.parse_args()

METRICS = Path(args.run) / "logs" / "learner_metrics.jsonl"

BAR_FULL = "█"
BAR_EMPTY = "░"


def read_records(path: Path):
    """jsonl 을 읽는다. 학습 중에 잘린 마지막 줄은 버린다."""
    episodes, train = [], []
    if not path.exists():
        return episodes, train
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # 쓰는 중인 마지막 줄
            if rec.get("kind") == "interaction":
                episodes.append(rec)
            elif rec.get("kind") == "train":
                train.append(rec)
    return episodes, train


def bar(frac: float, width: int = 12) -> str:
    n = 0 if frac <= 0 else max(1, round(frac * width))
    n = min(n, width)
    return BAR_FULL * n + BAR_EMPTY * (width - n)


def fmt_pct(x) -> str:
    return "  -  " if x is None else f"{x * 100:4.0f}%"


def render() -> str:
    out = []
    episodes, train = read_records(METRICS)

    out.append(f"\033[1m== 학습 진행 ==\033[0m  {args.run}")

    if not METRICS.exists():
        out.append("")
        out.append(f"  {METRICS} 가 아직 없다.")
        out.append("  learner 를 아직 안 띄웠거나, 첫 에피소드가 끝나지 않았다.")
        out.append("  actor 는 에피소드가 끝날 때만 전이를 보내므로 처음 20~40초는 비어 있는 게 정상이다.")
        return "\n".join(out)

    if train:
        last = train[-1]
        opt = last.get("Optimization step")
        onl = last.get("replay_buffer_size")
        off = last.get("offline_replay_buffer_size")
        line = "  최적화 스텝 " + (f"{opt:,}" if isinstance(opt, (int, float)) else "?")
        if isinstance(onl, (int, float)):
            line += f"   |   온라인 버퍼 {onl:,}"
        if isinstance(off, (int, float)):
            line += f"   |   시연 버퍼 {off:,}"
        out.append(line)
        for key, label in (("loss_critic", "critic"), ("loss_actor", "actor"), ("loss_temperature", "temp")):
            if key in last and isinstance(last[key], (int, float)):
                out.append(f"  {label:7s} loss = {last[key]:.4f}")
                break

    if not episodes:
        out.append("")
        out.append("  아직 끝난 에피소드가 없다. 첫 에피소드(최대 200스텝 = 20초)를 기다리는 중이다.")
        return "\n".join(out)

    total_succ = sum(1 for e in episodes if (e.get("Episodic reward") or 0) > 0)
    out.append("")
    out.append(f"  에피소드 {len(episodes)}개  ·  성공 {total_succ}개  ·  전체 성공률 {total_succ / len(episodes) * 100:.0f}%")

    # ---- 구간별 표 -------------------------------------------------------
    buckets = {}
    for e in episodes:
        step = e.get("Interaction step") or 0
        b = int(step) // args.bucket
        d = buckets.setdefault(b, {"n": 0, "succ": 0, "interv": [], "rew": 0.0})
        d["n"] += 1
        d["succ"] += 1 if (e.get("Episodic reward") or 0) > 0 else 0
        d["rew"] += float(e.get("Episodic reward") or 0.0)
        ir = e.get("Intervention rate")
        if isinstance(ir, (int, float)):
            d["interv"].append(float(ir))

    out.append("")
    out.append("  \033[2m구간(상호작용 스텝)   에피소드   성공   성공률   개입률   추이\033[0m")
    for b in sorted(buckets):
        d = buckets[b]
        lo, hi = b * args.bucket, (b + 1) * args.bucket - 1
        rate = d["succ"] / d["n"]
        irate = sum(d["interv"]) / len(d["interv"]) if d["interv"] else None
        out.append(
            f"  {lo:7,} ~ {hi:7,}   {d['n']:6d}   {d['succ']:4d}   {fmt_pct(rate)}   {fmt_pct(irate)}   {bar(rate)}"
        )

    # ---- 최근 요약 -------------------------------------------------------
    recent = episodes[-args.recent :]
    r_succ = sum(1 for e in recent if (e.get("Episodic reward") or 0) > 0)
    r_iv = [e["Intervention rate"] for e in recent if isinstance(e.get("Intervention rate"), (int, float))]
    out.append("")
    out.append(
        f"  \033[1m최근 {len(recent)} 에피소드: 성공 {r_succ}/{len(recent)} "
        f"({r_succ / len(recent) * 100:.0f}%)"
        + (f"  ·  개입률 {sum(r_iv) / len(r_iv) * 100:.0f}%" if r_iv else "")
        + "\033[0m"
    )

    # ---- 읽는 법 ---------------------------------------------------------
    if len(buckets) >= 2:
        ks = sorted(buckets)
        first = buckets[ks[0]]["succ"] / buckets[ks[0]]["n"]
        lastb = buckets[ks[-1]]["succ"] / buckets[ks[-1]]["n"]
        iv_f = buckets[ks[0]]["interv"]
        iv_l = buckets[ks[-1]]["interv"]
        msg = []
        if lastb > first + 0.05:
            msg.append("성공률이 오르고 있다")
        elif lastb < first - 0.05:
            msg.append("성공률이 떨어졌다")
        else:
            msg.append("성공률이 아직 평평하다")
        if iv_f and iv_l:
            if sum(iv_l) / len(iv_l) < sum(iv_f) / len(iv_f) - 0.05:
                msg.append("개입은 줄고 있다 — 정책이 스스로 하는 비율이 늘었다는 뜻")
            elif sum(iv_l) / len(iv_l) > sum(iv_f) / len(iv_f) + 0.05:
                msg.append("개입이 늘었다")
        out.append(f"  \033[2m→ {' · '.join(msg)}\033[0m")

    return "\n".join(out)


if not args.follow:
    print(render())
    sys.exit(0)

try:
    while True:
        os.system("clear")
        print(render())
        print(f"\n\033[2m{time.strftime('%H:%M:%S')}  ·  {args.interval:.0f}초마다 갱신  ·  Ctrl+C 로 종료\033[0m")
        time.sleep(args.interval)
except KeyboardInterrupt:
    print()
