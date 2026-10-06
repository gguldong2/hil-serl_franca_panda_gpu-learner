# UR5e 전환 PRD

Franka Panda 로 돌고 있는 HIL-SERL 큐브 놓기 태스크를 **UR5e 로도 돌릴 수 있게** 한다.
이 문서는 다른 세션에서 이걸 보고 작업할 수 있도록 쓴 것이다.

| | |
| --- | --- |
| 작성 | 2026-10-06 |
| 상태 | **미착수.** 아래 "사전 조사" 는 코드를 직접 읽고 확인한 것이고, Phase 0~5 는 계획이다 |
| 결정 | **모노레포 + 설정 전환.** 브랜치도 별도 저장소도 아니다 |
| 기준선 | Panda `PandaPlaceCubeKeyboard-v0` — 33,123판 / 802,000스텝 / 40.9시간, 마지막 200판 100% 성공 |

---

## 1. 목표와 범위

**목표**: `UR5ePlaceCube*-v0` 환경에서 Panda 와 **같은 태스크**(블록을 집어 바닥 목표 패드에 놓기)를
같은 학습 스택(lerobot actor–learner + SAC/RLPD)으로 돌린다.

**범위 안**
- gym-hil 에 UR5e 로봇 자산과 환경 추가
- 로봇에 묶인 코드를 XML 이름 규약으로 분리
- UR5e 용 train config, 시연 데이터
- (마지막) `--robot` CLI

**범위 밖**
- 태스크 자체를 바꾸는 것 (보상·성공조건·블록/패드 배치는 그대로)
- 학습 알고리즘 변경
- 실물 로봇
- Panda 결과를 UR5e 로 재현하는 것 (**정책 가중치는 호환되지 않는다**)

---

## 2. 사전 조사 — 코드를 읽고 확인한 것

### 2.1 제어가 작업공간(Cartesian)이다 — 이게 전환을 쉽게 만든다

```
행동 (xyz 3차원 델타, [-1,1])
  → mocap body "target" 의 위치를 그만큼 이동  (mujoco_gym_env.py:234-238)
  → opspace(site_id=pinch, dof_ids, pos=mocap_pos, ori=mocap_quat, joint=home)
  → data.ctrl[dof_ids] = tau                     (mujoco_gym_env.py:248-257)
```

- `opspace()` 는 `model.nv`, `len(dof_ids)`, 자코비안으로만 돌아간다.
  **7자유도가 박혀 있지 않다** (`gym_hil/controllers/opspace.py`).
- mocap body `target` 은 **목표 자세를 담아두는 그릇일 뿐 용접(weld)이 아니다.**
  `panda.xml` 의 `<equality>` 는 그리퍼 손가락 연동(tendon)이지 mocap 용접이 아니다.
  → UR5e XML 에 weld 를 새로 만들 필요가 없다. mocap body 하나만 있으면 된다.

### 2.2 자산이 이미 계층으로 갈라져 있다

```
place_cube_scene.xml        태스크 (블록 · 목표패드 · 카메라 front · mocap target)  ... 로봇 무관
  └ <include file="panda.xml"/>
       로봇 (joint1..7, actuator1..7, pinch site, handcam_rgb 카메라)
       └ robotiq_2f85/      그리퍼 (fingers_actuator, ctrlrange 0~255)  ... 별도 자산
```

### 2.3 로봇에 실제로 묶인 코드는 4줄

`third_party/gym-hil/gym_hil/mujoco_gym_env.py:162-165`

```python
self._panda_dof_ids   = np.asarray([self._model.joint(f"joint{i}").id    for i in range(1, 8)])
self._panda_ctrl_ids  = np.asarray([self._model.actuator(f"actuator{i}").id for i in range(1, 8)])
self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
self._pinch_site_id   = self._model.site("pinch").id
```

**`range(1, 8)` 만 동적으로 바꾸면 Python 쪽은 끝난다.** 나머지 셋은 손댈 필요가 없고,
대신 **이름 규약**으로 본다. UR5e XML 이 이것만 제공하면 된다:

| 이름 | 무엇 |
| --- | --- |
| `joint1` … `jointN` | 팔 관절 (연속 번호) |
| `actuator1` … `actuatorN` | 각 관절의 **토크(motor) 액추에이터** |
| `fingers_actuator` | 그리퍼 |
| `pinch` | TCP site (opspace 가 이 site 를 목표 자세로 몬다) |
| `front`, `handcam_rgb` | 카메라 두 대 |
| mocap body 1개 | opspace 목표 자세 홀더 |

### 2.4 이미 파라미터화되어 있는 것 (건드릴 필요 없음)

- `xml_path`, `home_position`, `cartesian_bounds` → **이미 생성자 인자**
  (`mujoco_gym_env.py:121-131`, 태스크 env 가 넘겨준다)
- `agent_dim = self.get_robot_state().shape[0]` → **상태 차원을 이미 동적으로 잡는다**
  (`panda_place_gym_env.py:92`)

### 2.5 상태 차원이 18 → 16 으로 바뀐다

```python
# mujoco_gym_env.py:266-270
qpos(N) + qvel(N) + gripper_pose(1) + tcp_pos(3)
Panda: 7+7+1+3 = 18        UR5e: 6+6+1+3 = 16
```

gym env 쪽은 자동으로 맞춰지지만, **lerobot train config 에는 18 이 박혀 있다**:
- `policy.input_features."observation.state".shape`
- `policy.dataset_stats."observation.state"` 의 min/max 배열 (18개 원소)

### 2.6 액추에이터 타입이 다르다 — 여기가 진짜 일

`panda.xml:357-363` 은 **토크 제어**다:

```xml
<motor class="panda" name="actuator1" joint="joint1" ctrlrange="-87 87"/>
...
<motor class="panda" name="actuator5" joint="joint5" ctrlrange="-12 12"/>
```

`opspace()` 가 토크를 내놓으므로 **반드시 `<motor>` 여야 한다.**
그런데 MuJoCo Menagerie 의 `universal_robots_ur5e` 는 **`position` 액추에이터로 배포된다.**
→ `<motor>` 로 바꾸고 `ctrlrange` 를 UR5e 관절 토크 한계로 넣어야 한다
(대략 base/shoulder/elbow 150 Nm, wrist 3개 28 Nm — Menagerie XML 에서 확인할 것).

---

## 3. 왜 모노레포인가

| 방식 | 판단 |
| --- | --- |
| 브랜치 | ✗ `scripts/`·lerobot 패치·설치 파이프라인을 고칠 때마다 양쪽 체리픽 |
| 별도 저장소 | ✗ 95% 중복. 패치·설치 인프라를 두 벌 유지 |
| **모노레포 + config 전환** | ✅ 공유분 한 벌, 로봇별은 파일 추가로 끝 |

로봇을 바꿀 때 달라지는 것이 **train config JSON 한 파일 안에 모두 모인다**:
`env.task`, `dataset.root`/`repo_id`, `output_dir`/`job_name`,
`policy.input_features."observation.state".shape`, `policy.dataset_stats`.

그래서 전환은 이미 **지금도** 가능하다:

```bash
CFG=configs/train_config_ur5e_place.json bash scripts/train.sh learner
CFG=configs/train_config_ur5e_place.json bash scripts/train.sh actor
```

`train.sh` 가 `output_dir` 을 config 에서 읽으므로 산출물도 자동으로 갈린다.
대시보드·뷰어·보고서는 그대로 붙는다.

---

## 4. 단계별 계획

> 각 단계의 **완료 기준은 실행해서 확인할 수 있는 것**으로 적었다.
> 기준을 통과하지 못하면 다음 단계로 넘어가지 않는다.

### Phase 0 — 기준선 고정 (코드 수정 없음)

Panda 가 지금 정상임을 먼저 기록해 둔다. 이후 어떤 변경이 Panda 를 깨뜨렸는지 즉시 알기 위해서다.

```bash
bash scripts/server22/03_verify.sh gpu
scripts/run_smoke_test.sh configs/train_config_place_smoke_gpu.json
```

**완료 기준**: 스모크가 체크포인트까지 통과. 로그를 `outputs/` 에 남긴다.

---

### Phase 1 — 로봇 중립화 (Panda 동작이 **하나도 안 변하는 것**이 완료 기준)

`mujoco_gym_env.py:162-163` 의 `range(1, 8)` 을 XML 에서 자동 탐지로 바꾼다.

```python
# joint1, joint2, ... 가 끊기는 지점까지가 팔의 자유도다.
n = 0
while True:
    try:
        self._model.joint(f"joint{n + 1}")
    except KeyError:
        break
    n += 1
if n == 0:
    raise RuntimeError("joint1 이 없다 — XML 의 이름 규약을 확인할 것")
self._arm_dof_ids  = np.asarray([self._model.joint(f"joint{i}").id    for i in range(1, n + 1)])
self._arm_ctrl_ids = np.asarray([self._model.actuator(f"actuator{i}").id for i in range(1, n + 1)])
```

- 기존 이름(`_panda_dof_ids`, `_panda_ctrl_ids`)을 쓰는 곳이 더 있으니 같이 바꾸거나 별칭을 남긴다.
- 클래스명 `FrankaGymEnv` 는 **그대로 둔다** (이름만 로봇 냄새가 날 뿐 동작은 중립이다).
  바꾸고 싶으면 `RobotGymEnv` 로 하고 `FrankaGymEnv = RobotGymEnv` 별칭을 남긴다.

**완료 기준**
1. Phase 0 과 **똑같은 스모크가 똑같이 통과**
2. `get_robot_state().shape[0] == 18` 그대로
3. `docs/patches/gym-hil/` 에 패치를 **다시 뜨고**, `docs/patches/README.md` 의 재현 점검
   스크립트로 **차이 0건** 확인

---

### Phase 2 — UR5e 자산 (물리만. 학습 없음)

1. [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 의
   `universal_robots_ur5e` 를 `gym_hil/assets/universal_robots_ur5e/` 로 가져온다.
2. `gym_hil/assets/ur5e.xml` 작성 — `panda.xml` 을 본보기로:
   - 관절을 `joint1`…`joint6` 으로 명명
   - 액추에이터를 **`<motor>`** 로, 이름 `actuator1`…`actuator6`, `ctrlrange` 는 토크 한계
   - `robotiq_2f85` 를 손목 플랜지에 부착 (기존 자산 그대로 재사용)
   - 그리퍼 TCP 에 `<site name="pinch">` — Panda 는 `pos="0 0 0.145"` 였다
   - 손목 카메라 `handcam_rgb` 배치
3. `gym_hil/assets/place_cube_scene_ur5e.xml` — `place_cube_scene.xml` 을 복사하고
   `<include file="panda.xml"/>` → `<include file="ur5e.xml"/>` 만 교체

**완료 기준**

```python
import mujoco
m = mujoco.MjModel.from_xml_path(".../place_cube_scene_ur5e.xml")
[m.joint(f"joint{i}") for i in range(1, 7)]       # 전부 찾아짐
[m.actuator(f"actuator{i}") for i in range(1, 7)]
m.actuator("fingers_actuator"); m.site("pinch")
m.camera("front"); m.camera("handcam_rgb")
```

그리고 뷰어에서 팔이 **중력에 처지지 않고 자세를 유지**해야 한다
(처지면 `ctrlrange` 토크 한계가 모자란 것).

---

### Phase 3 — 제어 성립 (**사람이 조작할 수 있는 것**이 완료 기준)

1. `gym_hil/envs/ur5e_place_gym_env.py` — `panda_place_gym_env.py` 를 복사해
   `_UR5E_HOME` (6개 값), `_CARTESIAN_BOUNDS` (UR5e 작업 범위), `xml_path` 만 바꾼다.
   보상·성공조건·블록/패드 배치는 **건드리지 않는다**.
2. `gym_hil/__init__.py` 에 `UR5ePlaceCube{,Keyboard,...}-v0` 등록
   (Panda 쪽 `register(...)` 블록 그대로 베낀다. `max_episode_steps=2000` 유지)
3. opspace 게인 튜닝 — `pos_gains`, `ori_gains`, `damping_ratio`

**완료 기준**: `scripts/teleop.sh` 로 **사람이 키보드로 블록을 집어 목표 패드에 올릴 수 있다.**

> 이게 안 되면 학습은 의미가 없다. 사람이 못 하는 것을 정책이 배울 수는 없다.
> 여기서 막히면 게인·home 자세·작업 범위를 다시 본다.

---

### Phase 4 — 학습 배관

1. `configs/train_config_ur5e_place.json` — `train_config_panda_place_gpu.json` 복사 후:
   - `env.task` = `UR5ePlaceCubeKeyboard-v0`
   - `job_name` / `output_dir` = `ur5e_place_01` / `outputs/train/ur5e_place_01`
   - `dataset.repo_id` / `root` = UR5e 시연
   - `policy.input_features."observation.state".shape` = `[16]`
   - `policy.dataset_stats."observation.state"` = 16개 원소 (시연 녹화 후 생성)
2. 시연 녹화 → 병합 → 통계

```bash
bash scripts/record.sh ur5e_01                   # 여러 세션
python3 scripts/merge_demos.py --success-only
python3 scripts/update_dataset_stats.py --repo_id ... --root ... --train_config ...
```

3. 스모크 → 본 학습

**완료 기준**
- 스모크가 체크포인트까지
- `train.sh status` 첫 줄에 `로봇 ur5e` 가 찍힘
- learner · actor · dashboard 세 개가 정상 (README §5 체크리스트 12항목)

---

### Phase 5 — `--robot` CLI (선택. **UR5e 가 돌아간 뒤에**)

```
configs/robots/
  franka-panda/ {train,record,teleop}.json
  ur5e/         {train,record,teleop}.json
```

| 명령 | `--robot` |
| --- | --- |
| `train.sh learner` / `actor` / `new`, `record.sh`, `teleop.sh` | **필수** |
| `train.sh status` / `pause` / `resume` / `save` / `stop` | **받지 않는다** — 돌고 있는 것을 따라간다 |
| `dashboard.sh` | 불필요 — 이미 돌고 있는 learner 의 `output_dir` 을 따라간다 |

`CFG=` 는 일회성 설정을 가리키는 탈출구로 남긴다 (`CFG` 가 있으면 `--robot` 불필요).

**기본값을 두지 않고 필수로 하는 이유**: 기본값은 조용한 선택이다. UR5e 를 의도하고
`train.sh learner` 를 쳤는데 Panda 가 떠서 20분 뒤에 알아차리는 것이 가장 흔한 사고다.
40시간짜리 학습을 시작하는 명령에서 15글자 더 치는 비용은 없는 것과 같다.

---

## 5. 올라오면서 정해야 할 것 (지금은 모른다)

| 질문 | 어느 단계에서 | 어떻게 판단 |
| --- | --- | --- |
| `control_time_s` 40초로 충분한가 | Phase 3 (teleop) | 사람이 40초 안에 여유 있게 성공하는가. 빠듯하면 늘린다 |
| `fps` 20 이 맞는가 | Phase 3 | 20 Hz 로 제어가 부드러운가. UR5e 가 더 무거워 떨리면 조정 |
| 그리퍼 ctrl 범위 0~255 | Phase 2 | Robotiq 2f85 를 그대로 쓰면 동일. 다른 걸 달면 `MAX_GRIPPER_COMMAND` 확인 |
| `cartesian_bounds` 를 **config 로 올릴 것인가** | Phase 3 | 아래 설명 |

### "config 가 아니라 env 인자여야 했네" 가 무슨 뜻인가

설정이 들어갈 수 있는 **층이 두 개**다.

| 층 | 어디 | 예 | 성질 |
| --- | --- | --- | --- |
| **(a) lerobot train config** | `configs/*.json` | `env.task`, `fps`, `control_time_s`, `policy.*` | 실행할 때마다 바꿀 수 있다. 학습 스택이 아는 값 |
| **(b) gym env 상수/생성자 인자** | `gym_hil/envs/*_gym_env.py` | `_PANDA_HOME`, `_CARTESIAN_BOUNDS`, `success_radius`, `lift_threshold` | 환경 클래스에 박혀 있다. 어느 gym id 를 등록했느냐로 정해진다 |

`cartesian_bounds` 는 지금 **(b)** 다. UR5e 를 올리다 보면 둘 중 하나가 된다:

- 한 번 정하면 안 바꾼다 → **(b) 로 두면 된다.** 배관 추가 없음
- 자주 튜닝하게 된다 → **(a) 로 올리고 싶어진다.** 그러려면
  lerobot config → `gym.make(**kwargs)` 로 넘기는 배관을 새로 만들어야 한다

**지금 `--robot` 을 먼저 만들면** "로봇별 차이는 전부 train config 에 있다" 를 전제로 설계하게 되는데,
Phase 3 에서 `cartesian_bounds` 가 (a) 로 가야 한다고 밝혀지면 **계획에 없던 배관을 덧대야 한다.**
그래서 UR5e 가 한 번 돌아간 뒤에 `--robot` 을 만든다. 그때는 이음새가 어디인지 다 보인다.

---

## 6. 위험

| 위험 | 왜 | 대처 |
| --- | --- | --- |
| **6축에는 여유자유도가 없다** | Panda 는 7축이라 null space 자세 제어가 되지만 UR5e 6축은 null space 가 거의 비어 있다. `opspace()` 의 `nullspace_stiffness` / `Jnull` 항이 무의미해지고 수치적으로 불안정할 수 있다 | Phase 3 에서 `nullspace_stiffness` 를 0 에 가깝게 낮춰 보고, 거동을 teleop 으로 확인 |
| **특이자세(singularity)** | UR5e 는 어깨·손목 특이점이 잘 걸린다. `Mx_inv` 의 행렬식이 작아지면 `pinv` 로 떨어지며 거동이 튄다 (`opspace.py:268`) | `_CARTESIAN_BOUNDS` 를 특이점을 피하는 범위로 잡는다. home 자세를 팔꿈치가 충분히 굽은 쪽으로 |
| 액추에이터가 position 이면 조용히 이상 동작 | opspace 가 토크를 내는데 position 으로 해석된다 | Phase 2 완료 기준(중력에 처지지 않는가)으로 걸러진다 |
| 패치를 다시 안 뜸 | clone 한 사람이 다른 코드를 받는다. **실제로 전에 이 사고가 있었다** | 각 Phase 완료 기준에 "패치 재현 점검 차이 0건" 포함 |
| 시연 재사용 시도 | 상태 차원(18→16)과 동역학이 달라 **재사용 불가** | UR5e 시연을 새로 녹화. 태스크·조작 키는 같아서 절차는 동일 |
| learner / actor 설정 어긋남 | 상태 차원이 달라 즉시 깨지거나 조용히 이상해진다 | `train.sh actor` 가 돌고 있는 learner 와 config 경로가 다르면 **중단한다** (`HIL_FORCE=1` 로만 강행) |

---

## 7. 참고 — 손댈 파일

| 파일 | Phase | 무엇 |
| --- | --- | --- |
| `third_party/gym-hil/gym_hil/mujoco_gym_env.py:162-165` | 1 | DOF 자동 탐지 |
| `third_party/gym-hil/gym_hil/assets/ur5e.xml` | 2 | **신규** |
| `third_party/gym-hil/gym_hil/assets/universal_robots_ur5e/` | 2 | **신규** (Menagerie) |
| `third_party/gym-hil/gym_hil/assets/place_cube_scene_ur5e.xml` | 2 | **신규** |
| `third_party/gym-hil/gym_hil/envs/ur5e_place_gym_env.py` | 3 | **신규** |
| `third_party/gym-hil/gym_hil/envs/__init__.py` | 3 | export |
| `third_party/gym-hil/gym_hil/__init__.py` | 3 | `register(...)` |
| `third_party/gym-hil/gym_hil/controllers/opspace.py` | 3 | 게인 (필요시) |
| `configs/train_config_ur5e_place.json` | 4 | **신규** |
| `configs/env_config_ur5e_place_record.json` | 4 | **신규** |
| `docs/patches/gym-hil/*.patch` | 1,2,3 | **매번 다시 뜰 것** |
| `scripts/train.sh`, `record.sh`, `teleop.sh` | 5 | `--robot` |

읽어야 할 것: `README.md`, `docs/gpu.md`, `docs/patches/README.md`,
`docs/01_place-cube-runbook.html`(조작), `docs/02_place-cube-hil-serl-QA.html`(학습 원리).
