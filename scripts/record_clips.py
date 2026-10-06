#!/usr/bin/env python3
"""학습 과정을 동영상으로 남긴다. 성공한 순간만, 또는 계속.

화면은 view-stream 래퍼가 이미 JPEG 으로 그려 /dev/shm 에 갱신하고 있다(대시보드가 보는 그 파일).
이 스크립트는 **렌더를 다시 하지 않고** 그 JPEG 을 모아 두었다가 mp4 로 떨군다.
영상 위에는 그 시점의 학습 상황(스텝·성공률·개입률·버퍼·경과시간)을 띠로 얹는다.

  python scripts/record_clips.py                    # 성공한 에피소드만 (기본 25초)
  python scripts/record_clips.py --all              # 실패한 에피소드도
  python scripts/record_clips.py --continuous 120   # 2분마다 통째로 (성공 클립과 함께)
  python scripts/record_clips.py --continuous 120 --no-episode-clips   # 통째로만
  touch /dev/shm/hil_clip_now                       # 지금 이 순간을 남기고 싶을 때

결과: outputs/clips/
  ep_00012340_성공_143052.mp4     에피소드 클립 (스텝 순으로 정렬됨)
  seg_00012340_143052.mp4         연속 녹화 구간
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--run", default=None,
               help="learner 의 output_dir. 생략하면 outputs/train/* 중 최신")
p.add_argument("--stream", default=os.environ.get("HIL_VIEW_STREAM", "/dev/shm/hil_view.jpg"))
p.add_argument("--out", default="outputs/clips")
p.add_argument("--seconds", type=float, default=25.0, help="에피소드 클립 길이")
p.add_argument("--fps", type=int, default=20)
p.add_argument("--all", action="store_true", help="실패한 에피소드도 저장")
p.add_argument("--threshold", type=float, default=0.5, help="이 보상 이상이면 성공")
p.add_argument("--max_clips", type=int, default=200)
p.add_argument("--continuous", type=float, default=0.0,
               help="N초마다 통째로 저장 (0 = 끔). 60~300 을 권한다")
p.add_argument("--no-episode-clips", dest="episode_clips", action="store_false",
               help="에피소드 클립은 만들지 않는다 (--continuous 와 같이 쓴다)")
p.add_argument("--no-overlay", dest="overlay", action="store_false", help="정보 띠를 얹지 않는다")
p.add_argument("--trigger", default="/dev/shm/hil_clip_now")
args = p.parse_args()

REWARD_RE = re.compile(r"Global step (\d+): Episode reward: ([-\d.]+)")
FONTS = ["/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf",
         "/usr/share/fonts/truetype/nanum/NanumGothicCodingBold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
START = time.time()


# --------------------------------------------------------------------------- 지표
def find_run(run: str | None) -> Path | None:
    if run:
        return Path(run)
    cands = [d for d in glob.glob("outputs/train/*") if not d.endswith("_actor")
             and os.path.isdir(os.path.join(d, "logs"))]
    return Path(max(cands, key=os.path.getmtime)) if cands else None


def find_actor_log(run: Path | None) -> Path | None:
    pats = [f"{run}_actor/logs/*.log"] if run else []
    pats.append("outputs/train/*_actor/logs/*.log")
    hits: list[str] = []
    for pat in pats:
        hits += glob.glob(pat)
        if hits:
            break
    return Path(max(hits, key=os.path.getmtime)) if hits else None


class Metrics:
    """learner_metrics.jsonl 을 꼬리부터 읽어 최신 값만 들고 있는다."""

    def __init__(self, run: Path | None):
        self.path = (run / "logs" / "learner_metrics.jsonl") if run else None
        self.pos = 0
        self.opt_step = None
        self.online = None
        self.offline = None
        self.rewards: collections.deque[float] = collections.deque(maxlen=20)
        self.ivs: collections.deque[float] = collections.deque(maxlen=20)

    def poll(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            size = self.path.stat().st_size
            if size < self.pos:
                self.pos = 0
            if size <= self.pos:
                return
            with self.path.open(errors="replace") as f:
                f.seek(self.pos)
                chunk = f.read()
                self.pos = f.tell()
        except OSError:
            return
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "train":
                for k, attr in (("Optimization step", "opt_step"),
                                ("replay_buffer_size", "online"),
                                ("offline_replay_buffer_size", "offline")):
                    v = r.get(k)
                    if isinstance(v, (int, float)):
                        setattr(self, attr, int(v))
            elif r.get("kind") == "interaction":
                v = r.get("Episodic reward")
                if isinstance(v, (int, float)):
                    self.rewards.append(float(v))
                v = r.get("Intervention rate")
                if isinstance(v, (int, float)):
                    self.ivs.append(float(v))

    def lines(self, step: int | None, note: str = "") -> list[str]:
        el = int(time.time() - START)
        elapsed = f"{el // 3600}시간 {el % 3600 // 60}분" if el >= 3600 else f"{el // 60}분 {el % 60}초"
        a = [f"스텝 {step:,}" if step is not None else "스텝 -"]
        if self.opt_step is not None:
            a.append(f"학습 {self.opt_step:,}회")
        a.append(f"경과 {elapsed}")
        if note:
            a.append(note)
        b = []
        if self.rewards:
            ok = sum(1 for r in self.rewards if r > args.threshold)
            b.append(f"최근{len(self.rewards)}판 성공 {100 * ok // len(self.rewards)}%")
        if self.ivs:
            b.append(f"개입 {100 * sum(self.ivs) // len(self.ivs)}%")
        if self.online is not None:
            b.append(f"온라인 {self.online:,}")
        if self.offline is not None:
            b.append(f"시연 {self.offline:,}")
        return [" · ".join(a), " · ".join(b) or "(learner 지표 대기 중)"]


# --------------------------------------------------------------------------- 인코딩
_BANNER_CACHE: dict[tuple, bytes] = {}


def make_banner(width: int, lines: list[str]) -> Path | None:
    """정보 띠를 PNG 한 장으로 그린다. 프레임마다 그리지 않고 ffmpeg 이 얹게 한다."""
    key = (width, tuple(lines))
    out = Path(args.out) / ".banner.png"
    if _BANNER_CACHE.get("key") == key and out.exists():
        return out
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    font = None
    for f in FONTS:
        if os.path.exists(f):
            try:
                font = ImageFont.truetype(f, 19)
                break
            except OSError:
                continue
    if font is None:
        return None
    h = 27 * len(lines) + 14
    img = Image.new("RGBA", (width, h), (0, 0, 0, 190))
    d = ImageDraw.Draw(img)
    for i, t in enumerate(lines):
        d.text((14, 8 + 27 * i), t, font=font, fill=(255, 255, 255, 235))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    _BANNER_CACHE["key"] = key
    return out


def frame_width(jpg: bytes) -> int:
    try:
        from PIL import Image
        import io as _io
        return Image.open(_io.BytesIO(jpg)).width
    except Exception:  # noqa: BLE001
        return 1024


def encode(frames: list[bytes], path: Path, lines: list[str] | None) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    banner = make_banner(frame_width(frames[0]), lines) if (lines and args.overlay) else None
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "image2pipe", "-vcodec", "mjpeg", "-r", str(args.fps), "-i", "-"]
    if banner is not None:
        cmd += ["-i", str(banner), "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto[v]",
                "-map", "[v]"]
    cmd += ["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-crf", "26", "-movflags", "+faststart", str(path)]
    try:
        pr = subprocess.run(cmd, input=b"".join(frames), stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"  [경고] ffmpeg 실패: {e}", flush=True)
        return False
    if pr.returncode != 0:
        print(f"  [경고] ffmpeg 실패: {pr.stderr.decode(errors='replace')[:300]}", flush=True)
        return False
    return True


# --------------------------------------------------------------------------- 본체
def main() -> int:
    stream = Path(args.stream)
    out_dir = Path(args.out)
    run = find_run(args.run)
    log = find_actor_log(run)
    met = Metrics(run)

    print("=" * 64)
    print("  학습 과정 녹화기")
    print("=" * 64)
    print(f"  화면       : {stream}")
    print(f"  학습 폴더  : {run if run else '아직 없음 (학습을 시작하면 자동으로 잡는다)'}")
    print(f"  actor 로그 : {log if log else '아직 없음'}")
    print(f"  저장       : {out_dir}/")
    if args.episode_clips:
        print(f"  에피소드   : {args.seconds:.0f}초 · "
              f"{'전부' if args.all else f'보상 {args.threshold} 이상(성공)만'}")
    else:
        print("  에피소드   : 끔")
    print(f"  연속 녹화  : {f'{args.continuous:.0f}초마다' if args.continuous > 0 else '끔'}")
    print(f"  정보 띠    : {'얹음' if args.overlay else '끔'}")
    print(f"  수동 저장  : touch {args.trigger}")
    print()
    print("  Ctrl+C 로 종료. 학습에는 아무 영향이 없다.")
    print()

    ring: collections.deque[bytes] = collections.deque(maxlen=max(1, int(args.seconds * args.fps)))
    seg: list[bytes] = []
    seg_started = time.time()
    last_mtime: float | None = None
    log_pos = 0
    saved = 0
    step = None
    seen = False

    while True:
        # 화면 한 장
        try:
            mt = stream.stat().st_mtime
            if mt != last_mtime:
                last_mtime = mt
                jpg = stream.read_bytes()
                if jpg[:2] == b"\xff\xd8" and jpg[-2:] == b"\xff\xd9":
                    ring.append(jpg)
                    if args.continuous > 0:
                        seg.append(jpg)
                    if not seen:
                        seen = True
                        print(f"  화면을 받기 시작했다 ({len(jpg) // 1024} KB/프레임)", flush=True)
        except OSError:
            pass

        met.poll()

        # 새 에피소드
        if log is None:
            log = find_actor_log(run)
            if log is not None:
                print(f"  actor 로그를 찾았다: {log}", flush=True)
                log_pos = log.stat().st_size
        if log is not None:
            try:
                size = log.stat().st_size
                if size < log_pos:
                    log_pos = 0
                if size > log_pos:
                    with log.open(errors="replace") as f:
                        f.seek(log_pos)
                        chunk = f.read()
                        log_pos = f.tell()
                    for m in REWARD_RE.finditer(chunk):
                        step, reward = int(m.group(1)), float(m.group(2))
                        ok = reward >= args.threshold
                        if not args.episode_clips:
                            continue
                        if not ok and not args.all:
                            print(f"  스텝 {step:>8,}  보상 {reward:+.2f}  실패", flush=True)
                            continue
                        if saved >= args.max_clips or not ring:
                            continue
                        name = f"ep_{step:08d}_{'성공' if ok else '실패'}_{time.strftime('%H%M%S')}.mp4"
                        fr = list(ring)
                        tag = f"보상 {reward:+.2f} {'성공' if ok else '실패'}"
                        if encode(fr, out_dir / name, met.lines(step, tag)):
                            saved += 1
                            print(f"  스텝 {step:>8,}  보상 {reward:+.2f}  저장 {name} "
                                  f"({len(fr)} 프레임, 누적 {saved})", flush=True)
            except OSError:
                pass

        # 연속 녹화 구간
        if args.continuous > 0 and time.time() - seg_started >= args.continuous:
            if seg:
                name = f"seg_{(step or 0):08d}_{time.strftime('%H%M%S')}.mp4"
                if encode(seg, out_dir / name, met.lines(step)):
                    saved += 1
                    print(f"  연속 구간 저장 {name} ({len(seg)} 프레임, "
                          f"{len(seg) / args.fps:.0f}초)", flush=True)
            seg = []
            seg_started = time.time()

        # 수동
        try:
            if os.path.exists(args.trigger):
                os.unlink(args.trigger)
                if ring:
                    name = f"manual_{time.strftime('%H%M%S')}.mp4"
                    fr = list(ring)
                    if encode(fr, out_dir / name, met.lines(step, "수동 저장")):
                        saved += 1
                        print(f"  수동 저장 {name} ({len(fr)} 프레임)", flush=True)
                else:
                    print("  수동 저장: 화면이 아직 없다", flush=True)
        except OSError:
            pass

        time.sleep(1.0 / (args.fps * 2))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  종료.")
