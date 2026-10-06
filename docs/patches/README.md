# 상류 코드 패치

`third_party/` 의 lerobot·gym-hil 소스는 **저장소에 올리지 않는다.**
대신 고정 커밋을 clone 한 뒤 여기 패치를 순서대로 적용한다.
`scripts/server22/02_setup_project.sh` 가 자동으로 한다.

```
lerobot  v0.6.1   7e241bd630a3719a56157a497ce5d08f244784f1
gym-hil  0.1.14   f89b3e5556798c44ef95083d92b670afa89d110a
```

패치는 **번호 순서대로, 각 저장소 루트에서** `git apply` 한다.
`02_setup_project.sh` 는 이미 적용된 패치를 `--reverse --check` 로 알아내 건너뛰므로
여러 번 돌려도 안전하다.

---

## lerobot — `000*.patch`

| 패치 | 파일 | 무엇을 |
| --- | --- | --- |
| `0001-buffer-uint8-image-scale` | `rl/buffer.py` | 시연 데이터 이미지가 0~255 로 들어와 온라인(0~1)과 스케일이 달랐다. 맞춘다 |
| `0002-learner-local-metrics-jsonl` | `rl/learner.py` | W&B 없이도 지표를 보려고 `logs/learner_metrics.jsonl` 로 남긴다. `SAVE_NOW`/`STOP_NOW` 트리거 파일 처리도 포함 |
| `0003-envs-configs-reset-joint-positions-type` | `envs/configs.py` | draccus 가 설정을 파싱할 때 나던 `TypeError` |
| `0004-gym-manipulator-close-env-and-fps` | `rl/gym_manipulator.py` | 종료 시 env 를 닫지 않던 것과 fps 처리 |
| `0005-actor-clean-shutdown` | `rl/actor.py` | Ctrl+C 때 정리 종료 |
| `0006-record-progress-output` | `rl/gym_manipulator.py` | 녹화 중 진행 표시가 한 줄도 안 나오던 문제(루트 로거가 WARNING 이라 `logging.basicConfig` 가 무효). 에피소드 시간 누적 버그도 함께 |

## gym-hil — `gym-hil/000*.patch`

| 패치 | 파일 | 무엇을 |
| --- | --- | --- |
| `0001-keyboard-listener-join-on-stop` | `wrappers/intervention_utils.py` | 헤드리스 X11(VNC/Xvfb)에서 종료 시 SIGSEGV. pynput 리스너 스레드를 `join` 하고 넘어간다. 미적용 시 3/19(16%), 적용 후 0/27 |
| `0002-panda-place-cube-env` | `__init__.py`, `envs/`, `wrappers/factory.py`, `wrappers/view_stream_wrapper.py`, `assets/place_cube_scene.xml` | **PandaPlaceCube 태스크 전체** (블록을 집어 바닥 목표 패드에 놓기) + 화면 스트림 래퍼(HUD·일시정지·경과시간) |
| `0003-keyboard-configurable-keymap` | `wrappers/intervention_utils.py` | 키 배치를 `configs/keymap.json`(또는 `GYM_HIL_KEYMAP`)으로 바꿀 수 있게. 그리퍼 키가 죽는 문제(X 에서 keyup 유실 시 "no-op" 고착)를 **마지막에 누른 쪽 우선**으로 수정한 것도 포함 |
| `0004-keyboard-config-passthrough` | `wrappers/hil_wrappers.py` | 키맵 설정을 래퍼까지 전달 |

> `0003` 은 `0001` 위에 얹히도록 만들어져 있다. 순서를 지켜야 한다.

---

## 재현 점검 — 패치가 실제 코드를 만들어 내는가

**새 머신에서 설치한 뒤, 또는 패치를 고친 뒤에는 반드시 이걸 돌린다.**
고정 커밋을 깨끗이 clone 해 패치를 적용하고, 지금 쓰고 있는 코드와 바이트 단위로 대조한다.

```bash
cd "$HIL_ROOT"           # 또는 프로젝트 루트
T=$(mktemp -d)
for R in lerobot gym-hil; do
  BASE=$(git -C third_party/$R rev-parse HEAD)
  git clone -q --shared --no-checkout third_party/$R "$T/$R"
  git -C "$T/$R" checkout -q "$BASE"
  PD="$PWD/docs/patches"; [ "$R" = gym-hil ] && PD="$PD/gym-hil"
  echo "=== $R (${BASE:0:8}) ==="
  for p in "$PD"/0*.patch; do
    [ -f "$p" ] || continue
    printf '  %-50s ' "$(basename "$p")"
    git -C "$T/$R" apply "$p" && echo OK || echo "<<< 실패"
  done
done
echo "--- 대조 ---"
diff -rq "$T/lerobot/src/lerobot" third_party/lerobot/src/lerobot | grep -v __pycache__
diff -rq "$T/gym-hil/gym_hil"     third_party/gym-hil/gym_hil     | grep -v __pycache__
echo "위에 아무것도 안 나오면 완전 일치."
rm -rf "$T"
```

마지막 확인: **2026-10-06, lerobot 6개 / gym-hil 4개 모두 적용 OK, 차이 0건.**

---

## 패치를 다시 뜨는 방법

`third_party/` 안의 코드를 고쳤다면 패치를 반드시 다시 떠야 한다.
안 그러면 clone 한 사람은 **다른 코드**를 받는다. (실제로 이 일이 있었다 — `0002` 의
`view_stream_wrapper.py` 가 HUD 작업 이전 버전으로 남아 있었고, `0003` 은 `0001` 을
건너뛴 기준으로 만들어져 충돌했다.)

한 저장소의 전체 변경을 한 패치로:

```bash
git -C third_party/gym-hil diff > docs/patches/gym-hil/00XX-이름.patch
```

파일을 나눠 여러 패치로 유지하려면, 고정 커밋에서 시작해 앞 패치를 적용한 상태를 만들고
해당 파일만 현재 코드로 덮은 뒤 `git diff --cached` 를 뜬다.
뜨고 나면 **위의 재현 점검을 반드시 다시 돌린다.**
