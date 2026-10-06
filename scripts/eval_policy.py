#!/usr/bin/env python
"""학습된 정책의 자율 평가 (PRD §8.4). 추가 구현.

공식 lerobot.rl.eval_policy 는 v0.6.1 에서 존재하지 않는 env_cfg.pretrained_policy_name_or_path 를 참조하고
make_robot_env 의 반환값(tuple)을 env 로 쓰는 등 그대로 실행되지 않아, actor 의 추론 경로
(make_policy -> make_pre_post_processors -> select_action -> step_env_and_process_transition)를 그대로 재사용해 작성했다.

- 가중치 고정(eval), optimizer 업데이트 없음, 인간 개입 없음(PandaPickCube-v0 = 입력 래퍼 없음)
- 에피소드별 보상·성공·스텝 수·소요 시간을 JSON 으로 기록
"""
import argparse, json, os, time, faulthandler
faulthandler.enable()
import numpy as np, torch, gymnasium as gym, gym_hil  # noqa: F401
import draccus
from lerobot.rl.train_rl import TrainRLServerPipelineConfig
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.factory import make_policy_config  # noqa: F401
from lerobot.rl.gym_manipulator import make_processors, reset_and_build_transition, step_env_and_process_transition
from lerobot.processor import TransitionKey
from lerobot.utils.random_utils import set_seed

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True, help="checkpoints/<step>/pretrained_model 디렉터리 (또는 last/pretrained_model)")
p.add_argument("--n_episodes", type=int, default=10)
p.add_argument("--seed", type=int, default=1000)
p.add_argument("--viewer", action="store_true", help="MuJoCo 뷰어 창 표시")
p.add_argument("--task", default="PandaPickCube-v0", help="입력 래퍼가 없는 환경 ID (개입 불가)")
p.add_argument("--out", default=None)
args = p.parse_args()

ckpt = os.path.abspath(args.checkpoint)
train_cfg_path = os.path.join(ckpt, "train_config.json")
cfg = draccus.parse(TrainRLServerPipelineConfig, train_cfg_path, args=[])
cfg.policy.pretrained_path = ckpt  # 가중치만 로딩 (optimizer/critic/replay buffer 는 복원하지 않음)
if cfg.algorithm is not None:
    cfg.algorithm.policy_config = cfg.policy
set_seed(args.seed)

env_cfg = cfg.env
env_cfg.task = args.task
env = gym.make(f"gym_hil/{args.task}", image_obs=True, render_mode="human", use_gripper=env_cfg.processor.gripper.use_gripper,
               gripper_penalty=env_cfg.processor.gripper.gripper_penalty, use_viewer=args.viewer)
env_proc, act_proc = make_processors(env, None, env_cfg, cfg.policy.device)

policy = make_policy(cfg=cfg.policy, env_cfg=env_cfg).to(cfg.policy.device).eval()
pre, post = make_pre_post_processors(policy_cfg=cfg.policy, dataset_stats=cfg.policy.dataset_stats)
n_params = sum(p.numel() for p in policy.parameters())
log(f"loaded policy from {ckpt} | params={n_params} | discrete_critic={'yes' if policy.discrete_critic is not None else 'no'}")

results = []
for ep in range(args.n_episodes):
    env.reset(seed=args.seed + ep)
    tr = reset_and_build_transition(env, env_proc, act_proc)
    ep_reward, steps, t0, success = 0.0, 0, time.perf_counter(), False
    while True:
        obs = {k: v for k, v in tr[TransitionKey.OBSERVATION].items() if k in cfg.policy.input_features}
        with torch.no_grad():
            a = policy.select_action(pre.process_observation(obs))
            if cfg.policy.num_discrete_actions is not None:
                # select_action() 은 policy.device(cuda) 텐서를 주고, post.process_action() 은
                # env 로 넘기려고 결과를 cpu 로 내린다. 이어붙이기 전에 두 조각의 device 를 맞춘다.
                # (CPU 전용이던 WSL 에서는 둘 다 cpu 라 드러나지 않던 GPU 전용 문제)
                cont = post.process_action(a[..., :-1])
                disc = a[..., -1:].to(torch.float32).to(cont.device)
                a = torch.cat([cont, disc], dim=-1)
            else:
                a = post.process_action(a)
        tr = step_env_and_process_transition(env, tr, a, env_proc, act_proc)
        ep_reward += float(tr[TransitionKey.REWARD]); steps += 1
        success = success or bool(tr[TransitionKey.INFO].get("succeed", False)) or float(tr[TransitionKey.REWARD]) >= 1.0
        if tr[TransitionKey.DONE] or tr[TransitionKey.TRUNCATED]:
            break
    r = {"episode": ep, "seed": args.seed + ep, "reward_sum": ep_reward, "success": success, "steps": steps, "wall_time_s": round(time.perf_counter() - t0, 2)}
    results.append(r); log(r)
env.close()
summary = {"checkpoint": ckpt, "task": args.task, "n_episodes": args.n_episodes, "seed_base": args.seed,
           "success_count": int(sum(r["success"] for r in results)), "success_rate": float(np.mean([r["success"] for r in results])),
           "mean_reward": float(np.mean([r["reward_sum"] for r in results])), "mean_steps": float(np.mean([r["steps"] for r in results])),
           "mean_wall_time_s": float(np.mean([r["wall_time_s"] for r in results])), "episodes": results,
           "note": "정책은 GaussianActor 의 확률적 샘플링(actor 와 동일). 인간 개입 없음. 가중치만 로딩."}
from pathlib import Path
_ck = Path(ckpt).resolve()  # .../<run>/checkpoints/<step>/pretrained_model (last 심볼릭 링크는 실제 step 으로 해석)
out = args.out or str(_ck.parents[2] / f"eval_{_ck.parent.name}.json")
summary["checkpoint_resolved"] = str(_ck)
os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(summary, open(out, "w"), indent=2, ensure_ascii=False)
log(f"success {summary['success_count']}/{args.n_episodes} | saved {out}")
