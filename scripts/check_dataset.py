#!/usr/bin/env python
"""시연 데이터셋 검증 (PRD §6). 추가 구현: 공식 코드에는 없는 점검 스크립트.

확인 항목: revision/포맷/에피소드·프레임 수, 관측·행동 키와 차원, 보상·done, 정규화 통계와의 일치,
실제 샘플 로딩(영상 디코딩), ReplayBuffer 변환 및 학습 배치 구성, 이미지 스케일(online/offline 일치 여부).
"""
import argparse, json, sys
import numpy as np, torch
from lerobot.datasets import LeRobotDataset
from lerobot.rl.buffer import ReplayBuffer

p = argparse.ArgumentParser()
p.add_argument("--repo_id", default="lilkm/pick_cube_franka_panda_30")
p.add_argument("--revision", default="f80a680a117d0c8573ab453e3fd9e792e308dfa4")
p.add_argument("--root", default=None, help="로컬 데이터셋 디렉터리. 주면 revision 은 무시된다.")
p.add_argument("--train_config", default="configs/train_config_panda_keyboard_cpu.json")
args = p.parse_args()
# 직접 녹화한 로컬 데이터셋은 Hub revision 이 없다.
revision = None if args.root else args.revision

ds = LeRobotDataset(args.repo_id, root=args.root, revision=revision, return_uint8=True)  # make_dataset()과 동일하게 uint8
meta = ds.meta
print("== META ==")
print("root:", ds.root)
print("codebase_version:", meta.info["codebase_version"], "| fps:", meta.fps, "| episodes:", meta.total_episodes, "| frames:", meta.total_frames, "| len(ds):", len(ds))
print("features:")
for k, v in meta.features.items():
    print(f"  {k:45s} {v['dtype']:8s} {str(v['shape']):16s} {v.get('names')}")

print("\n== EPISODES ==")
hf = ds.hf_dataset
ep = np.array(hf["episode_index"]); rew = np.array(hf["next.reward"]).reshape(-1); done = np.array(hf["next.done"]).reshape(-1)
pen = np.array(hf["complementary_info.discrete_penalty"]).reshape(-1)
act = np.array(hf["action"]); st = np.array(hf["observation.state"])
lens = np.bincount(ep)
print("episode lengths:", lens.tolist())
print("reward values:", dict(zip(*np.unique(rew, return_counts=True))))
print("episodes ending with reward 1.0:", int(sum(rew[np.where(ep==e)[0][-1]]==1.0 for e in range(meta.total_episodes))), "/", meta.total_episodes)
print("next.done True count:", int(done.sum()), "| done at last frame of each episode:", all(done[np.where(ep==e)[0][-1]] for e in range(meta.total_episodes)))
print("discrete_penalty values:", dict(zip(*np.unique(pen, return_counts=True))))

print("\n== ACTION (delta_x, delta_y, delta_z, gripper) ==")
print("shape:", act.shape, "min:", act.min(0).round(3).tolist(), "max:", act.max(0).round(3).tolist())
print("gripper column unique values:", np.unique(act[:, 3]).tolist(), "(gym_hil: 0=열기(놓기), 1=유지, 2=닫기(잡기). 코드 변수명 open/close 와 실제 동작이 반대다)")
print("xyz unique values (first 10):", np.unique(act[:, :3])[:10].tolist())

print(f"\n== STATE ({st.shape[1]}) vs train_config dataset_stats ==")
print("state min:", st.min(0).round(3).tolist()); print("state max:", st.max(0).round(3).tolist())
try:
    cfg = json.load(open(args.train_config)); cs = cfg["policy"]["dataset_stats"]["observation.state"]
    inside = np.all(st >= np.array(cs["min"]) - 1e-6) and np.all(st <= np.array(cs["max"]) + 1e-6)
    print("all states inside config min/max:", bool(inside))
    print("config min:", cs["min"]); print("config max:", cs["max"])
    print("config action min/max:", cfg["policy"]["dataset_stats"]["action"], "| policy output shape:", cfg["policy"]["output_features"]["action"]["shape"], "| num_discrete_actions:", cfg["policy"]["num_discrete_actions"])
except FileNotFoundError:
    print("(train_config not found, skip)")
_n = (st.shape[1] - 4) // 2
print(f"state layout (gym_hil FrankaGymEnv.get_robot_state): qpos[{_n}] | qvel[{_n}] | gripper_ctrl[1] (0~255) | tcp_pos[3]")

print("\n== SAMPLE LOADING (video decode) ==")
s = ds[0]
for k in ["observation.images.front", "observation.images.wrist", "observation.state", "action", "next.reward", "next.done", "complementary_info.discrete_penalty"]:
    v = s[k]; print(f"  {k:45s} {str(tuple(v.shape)):14s} {str(v.dtype):14s} min={float(v.min()):.3f} max={float(v.max()):.3f}")
ds_f = LeRobotDataset(args.repo_id, root=args.root, revision=revision, return_uint8=False)
sf = ds_f[0]["observation.images.front"]; print(f"  (return_uint8=False) front: {sf.dtype} min={float(sf.min()):.3f} max={float(sf.max()):.3f}")

print("\n== REPLAY BUFFER (learner offline buffer 경로 재현) ==")
keys = ["observation.images.front", "observation.images.wrist", "observation.state"]
buf = ReplayBuffer.from_lerobot_dataset(ds, device="cpu", state_keys=keys, capacity=len(ds), storage_device="cpu", optimize_memory=True)
print("buffer size:", len(buf), "| action storage shape:", tuple(buf.actions.shape), "| image storage dtype:", buf.states[keys[0]].dtype)
b = buf.sample(8)
print("batch keys:", list(b.keys()))
for k in keys: print(f"  state[{k}] {tuple(b['state'][k].shape)} {b['state'][k].dtype} min={float(b['state'][k].min()):.3f} max={float(b['state'][k].max()):.3f}")
print("  action", tuple(b["action"].shape), "| reward", tuple(b["reward"].shape), "| done", tuple(b["done"].shape), "| complementary_info:", {k: tuple(v.shape) for k, v in (b["complementary_info"] or {}).items()})
img_max = float(b["state"][keys[0]].max())
print("\nOFFLINE image scale in buffer:", "0~255 (uint8 값이 float으로 복사됨)" if img_max > 1.5 else "0~1")
print("ONLINE image scale (VanillaObservationProcessorStep): 0~1  ->", "불일치!" if img_max > 1.5 else "일치")
