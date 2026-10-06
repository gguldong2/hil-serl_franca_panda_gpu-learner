#!/usr/bin/env python
"""Panda 환경 동작·정책 관측 검증 (PRD §7, §10). 추가 구현: 공식 코드에는 없는 점검 스크립트.

--task PandaPickCube-v0          : 창 없이 reset/step, 카메라 관측 생성 확인 (headless)
--task PandaPickCubeKeyboard-v0  : lerobot make_robot_env 경로 그대로(뷰어 창 + 키보드 리스너) 실행
"""
import argparse, time, os
import numpy as np, torch, gymnasium as gym
import gym_hil  # noqa: F401  (환경 등록)

p = argparse.ArgumentParser()
p.add_argument("--task", default="PandaPickCube-v0")
p.add_argument("--steps", type=int, default=30)
p.add_argument("--fps", type=int, default=10)
args = p.parse_args()

print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"), "| DISPLAY =", os.environ.get("DISPLAY"))
# lerobot.rl.gym_manipulator.make_robot_env 와 동일한 인자
env = gym.make(f"gym_hil/{args.task}", image_obs=True, render_mode="human", use_gripper=True, gripper_penalty=-0.02)
print("wrappers:", end=" ")
e = env
while hasattr(e, "env"):
    print(type(e).__name__, end=" > "); e = e.env
print(type(e).__name__)
print("observation_space:", env.observation_space)
print("action_space:", env.action_space)

from lerobot.envs.configs import HILSerlRobotEnvConfig, HILSerlProcessorConfig, GripperConfig, ResetConfig
from lerobot.rl.gym_manipulator import make_processors, reset_and_build_transition, step_env_and_process_transition
from lerobot.processor import TransitionKey
cfg = HILSerlRobotEnvConfig(name="gym_hil", task=args.task, fps=args.fps,
    processor=HILSerlProcessorConfig(control_mode="keyboard", gripper=GripperConfig(True, -0.02),
        reset=ResetConfig(fixed_reset_joint_positions=[0.0,0.195,0.0,-2.43,0.0,2.62,0.785], reset_time_s=1.0, control_time_s=15.0)))
env_proc, act_proc = make_processors(env, None, cfg, "cpu")
print("env processor steps:", [type(s).__name__ for s in env_proc.steps])
print("action processor steps:", [type(s).__name__ for s in act_proc.steps])

t0 = time.perf_counter(); tr = reset_and_build_transition(env, env_proc, act_proc); print(f"reset OK ({1000*(time.perf_counter()-t0):.0f} ms)")
obs = tr[TransitionKey.OBSERVATION]
print("policy observation keys/shapes/dtypes/ranges:")
for k, v in obs.items():
    if isinstance(v, torch.Tensor): print(f"  {k:30s} {str(tuple(v.shape)):18s} {v.dtype} min={float(v.min()):.3f} max={float(v.max()):.3f}")
print("info:", {k: v for k, v in tr[TransitionKey.INFO].items()})

# 정책 대신 스크립트 행동: 아래로(z-) 이동 후 그리퍼 닫기 → 위로 올리기 (성공 여부는 무관, step 동작 확인용)
rewards, steps_t = [], []
for i in range(args.steps):
    if i < 10: a = torch.tensor([0.0, 0.0, -1.0, 1.0])
    elif i < 14: a = torch.tensor([0.0, 0.0, 0.0, 0.0])
    else: a = torch.tensor([0.0, 0.0, 1.0, 1.0])
    t0 = time.perf_counter()
    tr = step_env_and_process_transition(env, tr, a, env_proc, act_proc)
    steps_t.append(time.perf_counter() - t0)
    rewards.append(float(tr[TransitionKey.REWARD]))
    info = tr[TransitionKey.INFO]
    if i % 5 == 0 or tr[TransitionKey.DONE] or tr[TransitionKey.TRUNCATED]:
        st = tr[TransitionKey.OBSERVATION]["observation.state"][0]
        print(f"  step {i:3d} reward={rewards[-1]:.1f} done={tr[TransitionKey.DONE]} trunc={tr[TransitionKey.TRUNCATED]} "
              f"is_intervention={info.get('is_intervention')} teleop_action={np.round(tr[TransitionKey.COMPLEMENTARY_DATA].get('teleop_action', np.zeros(4)),2).tolist()} "
              f"tcp_z={float(st[17]):.3f} gripper={float(st[14]):.0f} penalty={tr[TransitionKey.COMPLEMENTARY_DATA].get('discrete_penalty')}")
    if tr[TransitionKey.DONE] or tr[TransitionKey.TRUNCATED]:
        print("  episode ended -> reset"); tr = reset_and_build_transition(env, env_proc, act_proc)
    time.sleep(max(0, 1/args.fps - steps_t[-1]))
print(f"step time: mean={1000*np.mean(steps_t):.1f} ms, max={1000*np.max(steps_t):.1f} ms (env fps target {args.fps} -> budget {1000/args.fps:.0f} ms)")
print("sum reward:", sum(rewards))
front = tr[TransitionKey.OBSERVATION]["observation.images.front"][0]
import imageio.v3 as iio
os.makedirs("outputs/check_env", exist_ok=True)
for cam in ["front", "wrist"]:
    img = (tr[TransitionKey.OBSERVATION][f"observation.images.{cam}"][0].permute(1,2,0).numpy()*255).astype(np.uint8)
    iio.imwrite(f"outputs/check_env/{args.task}_{cam}.png", img)
print("saved policy camera frames to outputs/check_env/")
env.close(); print("env closed OK")
