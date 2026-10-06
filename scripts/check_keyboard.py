#!/usr/bin/env python
"""키보드 개입 경로 검증 (PRD §7.2). 추가 구현.

PandaPickCubeKeyboard-v0 를 lerobot 과 같은 인자로 만들고(뷰어 창 + pynput 리스너),
--inject 옵션이면 pynput Controller 로 X 서버에 키 이벤트를 주입해 개입 토글/이동/그리퍼가
info["is_intervention"], teleop_action 에 반영되는지 자동 확인한다.
--inject 없이 실행하면 사용자가 실제 키보드로 조작하면서 콘솔에 출력되는 상태를 확인한다.
"""
import argparse, time, subprocess, os
import numpy as np, torch, gymnasium as gym, gym_hil  # noqa: F401
from lerobot.envs.configs import HILSerlRobotEnvConfig, HILSerlProcessorConfig, GripperConfig, ResetConfig
from lerobot.rl.gym_manipulator import make_processors, reset_and_build_transition, step_env_and_process_transition
from lerobot.processor import TransitionKey

p = argparse.ArgumentParser(); p.add_argument("--inject", action="store_true"); p.add_argument("--seconds", type=float, default=60)
p.add_argument("--inject-backend", choices=["xtest", "pynput"], default="xtest",
               help="키 주입 방법. xtest=키코드로 직접 주입(기본, 사람이 누른 것과 같은 이벤트). pynput 은 2중 입력 문제가 있다(아래 주석).")
args = p.parse_args()
env = gym.make("gym_hil/PandaPickCubeKeyboard-v0", image_obs=True, render_mode="human", use_gripper=True, gripper_penalty=-0.02)
cfg = HILSerlRobotEnvConfig(name="gym_hil", task="PandaPickCubeKeyboard-v0", fps=10, processor=HILSerlProcessorConfig(control_mode="keyboard", gripper=GripperConfig(True, -0.02), reset=ResetConfig(reset_time_s=1.0, control_time_s=15.0)))
env_proc, act_proc = make_processors(env, None, cfg, "cpu")
tr = reset_and_build_transition(env, env_proc, act_proc)
time.sleep(1.0)
try:
    wins = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True, timeout=10).stdout
    mj = [l.strip() for l in wins.splitlines() if "MuJoCo" in l or "mujoco" in l.lower()]
    print("X windows containing 'MuJoCo':", mj if mj else "(none found)")
except Exception as ex:
    print("xwininfo failed:", ex)

def step(a):
    global tr
    tr = step_env_and_process_transition(env, tr, torch.tensor(a, dtype=torch.float32), env_proc, act_proc)
    info = tr[TransitionKey.INFO]; cd = tr[TransitionKey.COMPLEMENTARY_DATA]
    st = tr[TransitionKey.OBSERVATION]["observation.state"][0]
    return bool(info.get("is_intervention", False)), np.round(np.asarray(cd.get("teleop_action")), 2).tolist(), float(st[15]), float(st[16]), float(st[17]), float(st[14]), float(tr[TransitionKey.REWARD]), bool(tr[TransitionKey.DONE] or tr[TransitionKey.TRUNCATED])

policy_action = [0.0, 0.0, 0.0, 1.0]  # 정책 대신 "정지" 행동
if args.inject:
    from pynput.keyboard import Controller, Key
    # 주입 방법 주의 — 이 서버(Xvnc)에서 실측한 것:
    #   1) pynput Controller 로 넣으면 같은 프로세스의 Listener 가 한 번의 입력을 **두 번** 받는다
    #      (press 2 / release 2). SPACE 처럼 토글하는 키는 켜졌다 곧바로 꺼져 개입이 시작되지 않는다.
    #      ENTER/ESC 는 상태를 세팅만 해서 두 번 와도 결과가 같아 정상으로 보였다.
    #   2) xdotool 에 **키심 이름**(Shift_R 등)을 주면 왼쪽 modifier 까지 같이 눌려
    #      Key.shift 와 Key.shift_r 이 동시에 와서 +z 와 -z 가 상쇄된다.
    #   3) **키코드로 직접** XTEST 주입하면 키마다 정확히 1회씩 온다 = 사람이 VNC 화면에서 누른 것과 동일.
    # 그래서 기본값은 xtest(키코드) 다.
    # 참고: pynput 은 Key.shift is Key.shift_l, Key.ctrl is Key.ctrl_l (별칭)이라
    #       왼쪽 modifier 는 일반형(Key.shift / Key.ctrl)으로 온다. gym_hil 핸들러가 이를 전제한다.
    _XKEY = {Key.space: "space", Key.up: "Up", Key.down: "Down", Key.left: "Left", Key.right: "Right",
             Key.shift: "Shift_L", Key.shift_r: "Shift_R", Key.ctrl_l: "Control_L", Key.ctrl_r: "Control_R",
             Key.enter: "Return", Key.esc: "Escape"}

    class _XTestController:
        """키코드로 직접 XTEST 이벤트를 넣는다 (python-xlib, pynput 의존성에 이미 포함)."""

        def __init__(self):
            import Xlib.display, Xlib.XK
            from Xlib import X
            from Xlib.ext import xtest
            self._X, self._xtest = X, xtest
            self._d = Xlib.display.Display()
            self._kc = {k: self._d.keysym_to_keycode(Xlib.XK.string_to_keysym(n)) for k, n in _XKEY.items()}
            missing = [n for k, n in _XKEY.items() if not self._kc[k]]
            if missing:
                raise RuntimeError(f"키맵에 없는 키: {missing}")

        def press(self, key):
            self._xtest.fake_input(self._d, self._X.KeyPress, self._kc[key]); self._d.sync()

        def release(self, key):
            self._xtest.fake_input(self._d, self._X.KeyRelease, self._kc[key]); self._d.sync()

    _backend = args.inject_backend
    if _backend == "xtest":
        try:
            kb = _XTestController()
        except Exception as ex:
            print(f"[inject] xtest 사용 불가({ex}) -> pynput 으로 대체"); _backend = "pynput"; kb = Controller()
    else:
        kb = Controller()
    print(f"[inject] backend={_backend}")
    def hold(key, n):
        kb.press(key); out = [step(policy_action) for _ in range(n)]; kb.release(key); time.sleep(0.05); return out
    print("\n[1] 개입 전: policy action 실행"); print("   ", step(policy_action))
    print("[2] SPACE -> 개입 시작 (정책 행동이 무시되고 키 입력이 실행되어야 함)"); kb.press(Key.space); kb.release(Key.space); time.sleep(0.05); print("   ", step(policy_action))
    for name, key, idx in [("UP(+x)", Key.up, 0), ("LEFT(+y)", Key.left, 1), ("SHIFT_R(+z)", Key.shift_r, 2), ("SHIFT_L(-z)", Key.shift, 2)]:
        r = hold(key, 3); print(f"[3] {name:12s} teleop_action={r[-1][1]} tcp(x,y,z)=({r[-1][2]:.3f},{r[-1][3]:.3f},{r[-1][4]:.3f}) is_intervention={r[-1][0]}")
    r = hold(Key.ctrl_r, 4); print(f"[4] CTRL_R('open' 라벨, 행동값 2.0) -> gripper ctrl {r[0][5]:.0f} -> {r[-1][5]:.0f}  (값 증가 = 2f85 손가락 닫힘)")
    r = hold(Key.ctrl_l, 4); print(f"[5] CTRL_L('close' 라벨, 행동값 0.0) -> gripper ctrl {r[0][5]:.0f} -> {r[-1][5]:.0f}  (값 감소 = 손가락 열림)")
    print("[6] SPACE -> 개입 종료 (정책 행동으로 복귀)"); kb.press(Key.space); kb.release(Key.space); time.sleep(0.05); print("   ", step(policy_action))
    print("[7] ENTER -> 성공 표시 (reward 1.0, done)"); kb.press(Key.enter); kb.release(Key.enter); time.sleep(0.05); r = step(policy_action); print("   reward,done =", r[6], r[7])
    tr = reset_and_build_transition(env, env_proc, act_proc)
    print("[8] ESC -> 실패 표시 (done, reward 0)"); kb.press(Key.esc); kb.release(Key.esc); time.sleep(0.05); r = step(policy_action); print("   reward,done =", r[6], r[7])
else:
    print("\n뷰어 창을 클릭해 포커스를 준 뒤 키를 눌러보세요. SPACE=개입 토글, 방향키=XY, Shift/우Shift=Z, 좌/우Ctrl=그리퍼, Enter=성공, Esc=실패. Ctrl+C 로 종료.")
    t0 = time.time(); last = None
    try:
        while time.time() - t0 < args.seconds:
            r = step(policy_action)
            if r[:2] != last: print(f"is_intervention={r[0]} teleop_action={r[1]} tcp=({r[2]:.3f},{r[3]:.3f},{r[4]:.3f}) gripper={r[5]:.0f} reward={r[6]} done={r[7]}"); last = r[:2]
            if r[7]: tr = reset_and_build_transition(env, env_proc, act_proc)
            time.sleep(0.1)
    except KeyboardInterrupt: pass
env.close(); print("closed")
