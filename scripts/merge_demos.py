#!/usr/bin/env python
"""여러 번에 나눠 녹화한 시연 데이터셋을 하나로 합친다.

gym_manipulator 의 record 모드는 LeRobotDataset.create() 를 쓰기 때문에
이미 있는 디렉터리에 이어서 녹화할 수 없다(FileExistsError). 그래서 시연은
data/<이름>_01, _02, ... 처럼 나눠 녹화한 뒤 이 스크립트로 합친다.

사용법:
  python scripts/merge_demos.py \
      --inputs data/panda_place_cube_demo_01 data/panda_place_cube_demo_02 \
      --out data/panda_place_cube_demo --out_repo_id local/panda_place_cube_demo

  # 또는 glob 으로
  python scripts/merge_demos.py --glob 'data/panda_place_cube_demo_*' \
      --out data/panda_place_cube_demo --out_repo_id local/panda_place_cube_demo
"""
import argparse, glob as globmod, json, shutil, sys
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

p = argparse.ArgumentParser()
p.add_argument("--inputs", nargs="*", default=[], help="합칠 데이터셋 디렉터리들")
p.add_argument("--glob", default=None, help="디렉터리 glob 패턴 (예: 'data/foo_*')")
p.add_argument("--out", required=True, help="결과 디렉터리")
p.add_argument("--out_repo_id", default=None, help="결과 repo_id (기본: local/<out 디렉터리 이름>)")
p.add_argument("--force", action="store_true", help="결과 디렉터리가 있으면 지우고 다시 만든다")
p.add_argument("--success-only", dest="success_only", action="store_true",
               help="성공한 에피소드만 남긴다. 실패가 섞이면 정책이 실패를 따라 배운다")
p.add_argument("--threshold", type=float, default=0.5, help="이 보상 이상이면 성공으로 본다")
args = p.parse_args()

roots = [Path(x) for x in args.inputs]
if args.glob:
    roots += [Path(x) for x in sorted(globmod.glob(args.glob))]
roots = [r for r in roots if r.is_dir()]
out = Path(args.out)
roots = [r for r in roots if r.resolve() != out.resolve()]
if len(roots) < 1:
    sys.exit("합칠 데이터셋을 찾지 못했다.")

out_repo_id = args.out_repo_id or f"local/{out.name}"

def filter_success(root: Path, repo_id: str, threshold: float) -> None:
    """성공한 에피소드만 남긴 데이터셋으로 갈아 끼운다.

    성공/실패는 에피소드 마지막 몇 프레임의 보상으로 판정한다. 환경이 성공하는 순간
    terminated 로 끝내므로 마지막 프레임에 보상 1.0 이 붙는다.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # 데이터셋이 스스로 매기는 값들. 다시 저장할 때 넣으면 안 된다.
    SKIP_KEYS = {"index", "episode_index", "frame_index", "task_index", "timestamp"}

    src = LeRobotDataset(repo_id, root=root)
    keep = []
    for e in range(src.num_episodes):
        m = src.meta.episodes[e]
        a, b = m["dataset_from_index"], m["dataset_to_index"]
        if max(float(src[i]["next.reward"]) for i in range(max(a, b - 5), b)) > threshold:
            keep.append(e)
    dropped = src.num_episodes - len(keep)
    print(f"\n성공만 남기기: {src.num_episodes} 중 성공 {len(keep)} · 실패 {dropped} 제외")
    if dropped == 0:
        print("  버릴 것이 없다. 그대로 둔다.")
        return
    if not keep:
        sys.exit("  성공한 에피소드가 하나도 없다. 중단한다.")

    tmp = root.parent / (root.name + "__success")
    if tmp.exists():
        shutil.rmtree(tmp)
    dst = LeRobotDataset.create(
        repo_id=repo_id, fps=src.fps, root=tmp, use_videos=True,
        image_writer_threads=4, image_writer_processes=0, features=src.features,
    )
    task = None
    for n, e in enumerate(keep, 1):
        m = src.meta.episodes[e]
        a, b = m["dataset_from_index"], m["dataset_to_index"]
        for i in range(a, b):
            f = src[i]
            if task is None:
                task = src.meta.tasks.index[int(f["task_index"])] if hasattr(src.meta, "tasks") else "place cube"
            frame = {}
            for k, spec in src.features.items():
                if k in SKIP_KEYS or k not in f:
                    continue
                v = f[k]
                want = tuple(spec.get("shape") or ())
                # LeRobotDataset 은 스칼라를 0차원으로 돌려주는데 add_frame 은 (1,) 을 기대한다.
                if want and hasattr(v, "reshape") and tuple(getattr(v, "shape", ())) != want:
                    try:
                        v = v.reshape(want)
                    except (RuntimeError, ValueError):
                        pass
                frame[k] = v
            frame["task"] = task
            dst.add_frame(frame)
        dst.save_episode()
        print(f"  {n}/{len(keep)}  (원본 에피소드 {e})", flush=True)
    if dst.writer is not None and dst.writer.image_writer is not None:
        dst.writer.image_writer.stop()
    # meta/episodes 의 parquet 은 finalize 에서 기록된다. 빼면 다시 읽을 때
    # "Provided directory does not contain any parquet file" 로 실패한다.
    dst.finalize()

    shutil.rmtree(root)
    tmp.rename(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    print(f"\n최종: {root}  {info['total_episodes']} 에피소드  {info['total_frames']} 프레임 (전부 성공)")


def read_meta(root: Path):
    info = json.loads((root / "meta" / "info.json").read_text())
    return info["total_episodes"], info["total_frames"], info["codebase_version"]

print("합칠 대상:")
total_ep = 0
for r in roots:
    try:
        ep, fr, ver = read_meta(r)
    except FileNotFoundError:
        sys.exit(f"  {r}: meta/info.json 이 없다. 녹화가 끝나지 않은 디렉터리로 보인다.")
    total_ep += ep
    print(f"  {str(r):45s} {ep:3d} 에피소드  {fr:6d} 프레임  {ver}")
print(f"  합계 {total_ep} 에피소드 -> {out}")

if out.exists():
    if not args.force:
        sys.exit(f"\n{out} 이 이미 있다. 지우고 다시 만들려면 --force 를 붙여라.")
    print(f"\n{out} 삭제 후 재생성")
    shutil.rmtree(out)

# repo_id 는 각 소스 디렉터리의 meta 에 들어 있는 값을 쓰지 않고, root 로만 구분한다.
src_repo_ids = [f"local/{r.name}" for r in roots]
aggregate_datasets(
    repo_ids=src_repo_ids,
    aggr_repo_id=out_repo_id,
    roots=roots,
    aggr_root=out,
)

ep, fr, ver = read_meta(out)
print(f"\n병합 완료: {out}  {ep} 에피소드  {fr} 프레임  {ver}  repo_id={out_repo_id}")
if ep != total_ep:
    print(f"!! 경고: 입력 합계({total_ep})와 결과({ep})가 다르다.")

if args.success_only:
    filter_success(out, out_repo_id, args.threshold)
