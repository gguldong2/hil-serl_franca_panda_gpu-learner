#!/usr/bin/env python
"""녹화한 시연 데이터셋의 통계를 학습 설정(train config)의 dataset_stats 에 써 넣는다.

관측 정규화(MIN_MAX)는 데이터 분포에 맞춰야 한다. 다른 태스크에서 만든 min/max 를
그대로 쓰면 정규화가 어긋나 학습이 잘 되지 않는다.

observation.state = qpos(7) + qvel(7) + gripper(1) + tcp(3) = 18

  qpos    : 데이터에서 측정한 min/max 를 그대로 (여유 margin 포함)
  qvel    : ±100 고정. 순간적으로 튀는 값이라 데이터 min/max 를 쓰면 정규화가 불안정하다.
  gripper : 0 ~ 255 고정 (액추에이터 ctrl 범위)
  tcp     : 환경의 cartesian bounds 고정

사용법:
  python scripts/update_dataset_stats.py \
      --repo_id local/panda_place_cube_demo --root data/panda_place_cube_demo \
      --train_config configs/train_config_panda_place_gpu.json
"""
import argparse, json, shutil
from pathlib import Path

import numpy as np
from lerobot.datasets import LeRobotDataset

p = argparse.ArgumentParser()
p.add_argument("--repo_id", required=True)
p.add_argument("--root", default=None)
p.add_argument("--train_config", required=True)
p.add_argument("--qpos_margin", type=float, default=0.05, help="qpos min/max 에 더할 여유")
p.add_argument("--qvel_limit", type=float, default=100.0)
p.add_argument("--tcp_bounds", default="0.2,-0.3,0.0,0.6,0.3,0.5", help="xmin,ymin,zmin,xmax,ymax,zmax")
p.add_argument("--dry_run", action="store_true")
args = p.parse_args()

ds = LeRobotDataset(args.repo_id, root=args.root, return_uint8=True)
meta = ds.meta
print(f"데이터셋: {args.repo_id}  root={ds.root}")
print(f"  codebase_version={meta.info['codebase_version']}  fps={meta.fps}  "
      f"episodes={meta.total_episodes}  frames={meta.total_frames}")

hf = ds.hf_dataset
rew = np.asarray(hf["next.reward"]).reshape(-1)
ep = np.asarray(hf["episode_index"]).reshape(-1)
n_succ = sum(1 for e in np.unique(ep) if rew[ep == e].max() > 0)
print(f"  성공(보상>0) 에피소드: {n_succ}/{meta.total_episodes}")
lens = [int((ep == e).sum()) for e in np.unique(ep)]
print(f"  에피소드 길이: 최소 {min(lens)}  중앙값 {int(np.median(lens))}  최대 {max(lens)}")

state = np.asarray(hf["observation.state"], dtype=np.float64)
assert state.shape[1] == 18, f"observation.state 차원이 18 이 아님: {state.shape}"

lo = state.min(axis=0)
hi = state.max(axis=0)

tb = [float(x) for x in args.tcp_bounds.split(",")]
new_min = np.empty(18); new_max = np.empty(18)
new_min[:7] = lo[:7] - args.qpos_margin
new_max[:7] = hi[:7] + args.qpos_margin
new_min[7:14] = -args.qvel_limit
new_max[7:14] = args.qvel_limit
new_min[14] = 0.0
new_max[14] = 255.0
new_min[15:18] = tb[:3]
new_max[15:18] = tb[3:]

names = [f"qpos{i+1}" for i in range(7)] + [f"qvel{i+1}" for i in range(7)] + ["gripper", "tcp_x", "tcp_y", "tcp_z"]
print("\n  idx 이름      데이터 min     데이터 max  ->  설정 min      설정 max")
for i in range(18):
    print(f"  {i:3d} {names[i]:8s} {lo[i]:12.5f} {hi[i]:12.5f}  ->  {new_min[i]:10.5f} {new_max[i]:10.5f}")

act = np.asarray(hf["action"], dtype=np.float64)
print(f"\n  action shape={act.shape}  연속 3축 min={np.round(act[:, :3].min(axis=0), 3)} "
      f"max={np.round(act[:, :3].max(axis=0), 3)}  이산 그리퍼 값={sorted(set(act[:, 3].tolist()))}")

cfg_path = Path(args.train_config)
cfg = json.loads(cfg_path.read_text())
stats = cfg["policy"]["dataset_stats"]
old_min = stats["observation.state"]["min"]
stats["observation.state"]["min"] = [round(float(v), 8) for v in new_min]
stats["observation.state"]["max"] = [round(float(v), 8) for v in new_max]

if args.dry_run:
    print("\n--dry_run 이라 파일은 쓰지 않았다.")
else:
    shutil.copy(cfg_path, cfg_path.with_suffix(".json.bak"))
    cfg_path.write_text(json.dumps(cfg, indent=4) + "\n")
    print(f"\n갱신 완료: {cfg_path}  (원본은 {cfg_path.with_suffix('.json.bak')})")
    print(f"  이전 qpos min[0:3] = {[round(v,4) for v in old_min[:3]]}")
    print(f"  새   qpos min[0:3] = {[round(v,4) for v in stats['observation.state']['min'][:3]]}")
