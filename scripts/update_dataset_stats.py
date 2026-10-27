#!/usr/bin/env python
"""녹화한 시연 데이터셋의 통계를 학습 설정(train config)의 dataset_stats 에 써 넣는다.

관측 정규화(MIN_MAX)는 데이터 분포에 맞춰야 한다. 다른 태스크에서 만든 min/max 를
그대로 쓰면 정규화가 어긋나 학습이 잘 되지 않는다.

observation.state = qpos(n) + qvel(n) + gripper(1) + tcp(3)
  Panda n=7 -> 18,  UR5e n=6 -> 16.  n 은 상태 차원에서 구한다: n = (차원 - 4) / 2

  qpos    : 데이터에서 측정한 min/max 를 그대로 (여유 margin 포함)
  qvel    : ±100 고정. 순간적으로 튀는 값이라 데이터 min/max 를 쓰면 정규화가 불안정하다.
  gripper : 0 ~ 255 고정 (액추에이터 ctrl 범위)
  tcp     : 환경의 cartesian bounds 고정

사용법:
  python scripts/update_dataset_stats.py \
      --repo_id local/panda_place_cube_demo --root data/panda_place_cube_demo \
      --train_config configs/train_config_panda_place_gpu.json

  # 설정 여러 개에 한꺼번에 (본 학습 + 스모크)
  python scripts/update_dataset_stats.py \
      --repo_id local/ur5e_place_cube_demo --root data/ur5e_place_cube_demo \
      --train_config configs/train_config_ur5e_place.json configs/train_config_ur5e_smoke.json

tcp 범위는 --tcp_bounds 를 안 주면 학습 설정의 env.task 에 해당하는 환경 코드의
_CARTESIAN_BOUNDS 를 그대로 읽는다 (Panda 0.2~, UR5e 0.25~). 숫자를 두 군데 적어 두지 않기 위해서다.
"""
import argparse, json, shutil
from pathlib import Path

import numpy as np
from lerobot.datasets import LeRobotDataset

p = argparse.ArgumentParser()
p.add_argument("--repo_id", required=True)
p.add_argument("--root", default=None)
p.add_argument("--train_config", required=True, nargs="+", help="갱신할 학습 설정 (여러 개 가능)")
p.add_argument("--qpos_margin", type=float, default=0.05, help="qpos min/max 에 더할 여유")
p.add_argument("--qvel_limit", type=float, default=100.0)
p.add_argument("--tcp_bounds", default=None, help="xmin,ymin,zmin,xmax,ymax,zmax (없으면 환경 코드에서 읽는다)")
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
D = state.shape[1]
assert D >= 6 and (D - 4) % 2 == 0, f"observation.state 차원 {D} 은 qpos(n)+qvel(n)+gripper+tcp(3) 꼴이 아니다"
n = (D - 4) // 2
print(f"  상태 {D}차원 -> 관절 {n}개")

# 학습 설정들을 먼저 읽고, 상태 차원이 시연과 같은지 본다.
# (Panda 시연(18)에 UR5e 설정(16)을 쓰는 식의 실수를 여기서 멈춘다)
cfgs = []
for path in args.train_config:
    cp = Path(path)
    c = json.loads(cp.read_text())
    want = c["policy"]["input_features"]["observation.state"]["shape"][0]
    if want != D:
        raise SystemExit(f"\n[멈춤] {cp}: 설정의 상태 차원은 {want}, 시연은 {D}. 다른 로봇의 데이터·설정을 섞은 것 같다.")
    cfgs.append((cp, c))

def tcp_bounds_from_env(task):
    """학습 설정의 태스크에 해당하는 환경 코드의 _CARTESIAN_BOUNDS 를 읽는다."""
    import importlib
    table = [("UR5ePlaceCube", "gym_hil.envs.ur5e_place_gym_env"),
             ("PandaPlaceCube", "gym_hil.envs.panda_place_gym_env"),
             ("PandaPickCube", "gym_hil.envs.panda_pick_gym_env"),
             ("PandaArrangeBoxes", "gym_hil.envs.panda_arrange_boxes_gym_env")]
    for prefix, mod in table:
        if task.startswith(prefix):
            b = np.asarray(importlib.import_module(mod)._CARTESIAN_BOUNDS, dtype=float)
            return [float(v) for v in b[0]] + [float(v) for v in b[1]], mod
    raise SystemExit(f"[멈춤] 태스크 {task} 의 작업 범위를 모른다. --tcp_bounds 로 직접 준다.")

if args.tcp_bounds:
    tb = [float(x) for x in args.tcp_bounds.split(",")]
    print(f"  tcp 범위: --tcp_bounds {tb}")
else:
    tasks = {c["env"]["task"] for _, c in cfgs}
    found = {}
    for t in tasks:
        b, m = tcp_bounds_from_env(t)
        found[tuple(b)] = (b, m)
    if len(found) != 1:
        raise SystemExit(f"[멈춤] 설정들의 작업 범위가 서로 다르다: {tasks}. 설정을 따로따로 갱신한다.")
    tb, mod = next(iter(found.values()))
    print(f"  tcp 범위: {[round(v, 3) for v in tb]}  ({mod}._CARTESIAN_BOUNDS)")

lo = state.min(axis=0)
hi = state.max(axis=0)

new_min = np.empty(D); new_max = np.empty(D)
new_min[:n] = lo[:n] - args.qpos_margin
new_max[:n] = hi[:n] + args.qpos_margin
new_min[n:2 * n] = -args.qvel_limit
new_max[n:2 * n] = args.qvel_limit
new_min[2 * n] = 0.0
new_max[2 * n] = 255.0
new_min[2 * n + 1:] = tb[:3]
new_max[2 * n + 1:] = tb[3:]

names = [f"qpos{i+1}" for i in range(n)] + [f"qvel{i+1}" for i in range(n)] + ["gripper", "tcp_x", "tcp_y", "tcp_z"]
print("\n  idx 이름      데이터 min     데이터 max  ->  설정 min      설정 max")
for i in range(D):
    print(f"  {i:3d} {names[i]:8s} {lo[i]:12.5f} {hi[i]:12.5f}  ->  {new_min[i]:10.5f} {new_max[i]:10.5f}")

act = np.asarray(hf["action"], dtype=np.float64)
print(f"\n  action shape={act.shape}  연속 3축 min={np.round(act[:, :3].min(axis=0), 3)} "
      f"max={np.round(act[:, :3].max(axis=0), 3)}  이산 그리퍼 값={sorted(set(act[:, 3].tolist()))}")

for cfg_path, cfg in cfgs:
    stats = cfg["policy"]["dataset_stats"]
    old_min = stats["observation.state"]["min"]
    stats["observation.state"]["min"] = [round(float(v), 8) for v in new_min]
    stats["observation.state"]["max"] = [round(float(v), 8) for v in new_max]

    if args.dry_run:
        print(f"\n--dry_run 이라 {cfg_path} 는 쓰지 않았다.")
    else:
        shutil.copy(cfg_path, cfg_path.with_suffix(".json.bak"))
        cfg_path.write_text(json.dumps(cfg, indent=4) + "\n")
        print(f"\n갱신 완료: {cfg_path}  (원본은 {cfg_path.with_suffix('.json.bak')})")
        print(f"  이전 qpos min[0:3] = {[round(v,4) for v in old_min[:3]]}")
        print(f"  새   qpos min[0:3] = {[round(v,4) for v in stats['observation.state']['min'][:3]]}")
