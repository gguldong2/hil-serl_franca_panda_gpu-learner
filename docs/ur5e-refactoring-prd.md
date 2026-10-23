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

**실제 값** — `data/panda_place_cube_demo` 의 한 프레임을 그대로 찍어 본 것:

```
[ 0] qpos1   =  0.0000    관절 1 각도 (rad)
[ 1] qpos2   =  0.1933    관절 2 각도
 ...                      ... 7개 (Panda 는 7축)
[ 6] qpos7   =  0.7854
[ 7] qvel1   =  0.0000    관절 1 각속도 (rad/s)
 ...                      ... 7개
[13] qvel7   = -0.0000
[14] gripper =  0.0000    그리퍼 열림 정도 (0=열림 ~ 255=닫힘)
[15] tcp_x   =  0.4903    손끝 위치 x (m)
[16] tcp_y   =  0.0000    손끝 위치 y
[17] tcp_z   =  0.0776    손끝 위치 z
```

즉 **"지금 로봇이 어떤 자세이고 얼마나 빨리 움직이고 손끝이 어디 있는가"** 다.
정책은 이 18개 숫자 + 카메라 2장을 보고 다음 행동을 고른다.
UR5e 는 관절이 6개뿐이라 `qpos`·`qvel` 이 각각 7→6 으로 줄어 **16개**가 된다.
(그리퍼 1 + 손끝 3 은 그대로 — 그리퍼도 손끝도 로봇이 바뀌어도 하나/세 개다.)

gym env 쪽은 자동으로 맞춰지지만, **lerobot train config 에는 18 이 박혀 있다**:
- `policy.input_features."observation.state".shape`
- `policy.dataset_stats."observation.state"` 의 min/max 배열 (18개 원소)

> 그래서 **Panda 시연 데이터와 Panda 정책 가중치는 UR5e 에서 쓸 수 없다.**
> 입력 길이부터 다르다.

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

### 2.7 에셋 — 무엇을 쓸 것인가와 그 이유

#### 결정: **Menagerie 의 UR5e 팔 + gym-hil 에 이미 있는 2f85 그리퍼 블록**

| 쓸 것 | 출처 | 라이선스 |
| --- | --- | --- |
| UR5e 팔 (링크·관절·메시·관성) | [MuJoCo Menagerie `universal_robots_ur5e`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e) | BSD-3-Clause |
| Robotiq 2F-85 그리퍼 | **이 저장소의 `gym_hil/assets/panda.xml` + `robotiq_2f85/meshes/`** | gym-hil (Apache-2.0) |

#### 왜 Menagerie 인가

MuJoCo 로봇 모델의 사실상 표준이다(DeepMind 가 큐레이션). UR 공식 배포물은
ROS 용 **URDF/xacro** (`UniversalRobots/Universal_Robots_ROS2_Description`) 라서
MJCF 로 옮기면서 충돌 메시 단순화·관성·접촉 파라미터를 직접 손봐야 한다.
Menagerie 는 그 작업이 끝나 있고 MuJoCo 에서 검증된 상태다.

#### 왜 "Menagerie UR5e + Menagerie 2f85 합치기" 가 아닌가 ← **여기가 핵심**

둘 다 Menagerie 에 있지만 **붙이는 일 자체가 쉽지 않다.** Menagerie 의 2F-85 는
4절 링크를 `connect` equality 로 닫고 tendon 액추에이터 하나로 모는 구조라,
팔에 붙일 때 접촉 제외(`<contact><exclude>`)·equality·tendon 을 전부 올바로 옮겨야 한다
(Menagerie issue #40 이 바로 "UR 팔에 2f85 붙이기" 문제다).

**그런데 그 작업은 이 저장소에 이미 되어 있다.** `panda.xml` 이 Panda + 2F-85 결합본이고,
결합에 필요한 것이 **하나의 서브트리로 깔끔하게 모여 있다**:

```xml
<body name="attachment" pos="0 0 0.107" quat="...">     ← 팔 끝(link7)에 달린 어태치먼트
  <camera name="handcam_depth" .../>
  <camera name="handcam_rgb"  .../>                      ← 손목 카메라
  <site name="attachment_site" .../>
  <body name="base_mount" childclass="2f85">             ← 여기부터 그리퍼 전체
    <body name="base" ...>
      <site name="pinch" pos="0 0 0.145"/>               ← TCP (opspace 가 모는 지점)
      ... right/left driver · coupler · follower · spring_link · pad ...
```

그리고 짝이 되는 `<contact><exclude>` 6줄, `<tendon name="split">`,
`<equality>` 3줄, `<general name="fingers_actuator" ctrlrange="0 255">` 가
파일 아래쪽에 모여 있다.

#### 그래서 `ur5e.xml` 만드는 법은 "합치기" 가 아니라 **"접붙이기(graft)"** 다

`panda.xml` 을 복사한 뒤:

| | 할 일 |
| --- | --- |
| ① | `<default class="panda">` 계열 → UR5e 용으로 교체 |
| ② | `<asset>` 의 panda 메시 → UR5e 메시로 교체 (2f85 메시 7줄은 **그대로**) |
| ③ | `<worldbody>` 의 `link0`…`link7` 체인 → UR5e `base`…`wrist_3_link` 체인으로 교체 |
| ④ | **`<body name="attachment">` 서브트리를 통째로** UR5e 손목(`wrist_3_link`) 밑으로 옮긴다. `pos`/`quat` 만 UR5e 플랜지에 맞게 조정 |
| ⑤ | `<actuator>` 의 panda `<motor>` 7개 → UR5e `<motor>` 6개 (`actuator1`…`actuator6`) |
| ⑥ | `<contact>` / `<tendon>` / `<equality>` / `fingers_actuator` / `pinch` / `handcam_rgb` — **손대지 않는다** |

④ 덕분에 `pinch`, `handcam_rgb`, 그리퍼 전체가 한 덩어리로 따라온다.
**이름 규약(§2.3)이 저절로 충족된다.**

> ⑤ 주의: Menagerie UR5e 는 **`position` 액추에이터로 배포된다**(README 에 명시).
> `opspace()` 는 토크를 내므로 반드시 `<motor>` 로 바꾸고 `ctrlrange` 를
> UR5e 관절 토크 한계로 넣어야 한다. Menagerie XML 의 `forcerange` 를 보고 맞춘다.

#### 참고만 하고 가져다 쓰지는 않을 것

[`ian-chuang/homestri-ur5e-rl`](https://github.com/ian-chuang/homestri-ur5e-rl) 에
UR5e + 2F-85 MJCF 와 operational space 컨트롤러가 있다. **붙이는 방법(플랜지 오프셋,
자세 쿼터니언)을 참고하기에 좋다.** 다만 저자가 "실험적이고 유지보수 안 함" 이라 밝혔고
**라이선스 표기가 없어** 파일을 가져다 쓰지는 않는다. 눈으로만 참고한다.

#### 검토했지만 쓰지 않는 것

| 후보 | 왜 안 쓰나 |
| --- | --- |
| UR 공식 URDF (ROS2 description) | MJCF 변환 + 충돌/관성 튜닝을 직접 해야 함. Menagerie 가 이미 해 둠 |
| robosuite 의 UR5e | 자체 로봇/그리퍼 추상화에 묶여 있어 떼어내면 결국 같은 일 |
| `robot_descriptions` 패키지 | 내부적으로 Menagerie 를 받아올 뿐. 의존성만 늘어남 |

---

### 2.8 시연 데이터 — 가져다 쓸 수 있는 것이 없다. 직접 녹화한다

찾아봤고, **쓸 수 있는 공개 데이터셋은 없다.** 확인한 것:

| 데이터셋 | 왜 못 쓰나 |
| --- | --- |
| `lilkm/pick_cube_franka_panda_30` (이 프로젝트 출발점) | **Panda**, 그리고 place 가 아니라 pick |
| `sk1700/Ur10e_Pickplace` | UR10e 실물, **행동이 7차원 관절 목표**(우리는 3차원 xyz 델타), 30 fps, 다른 장면 |
| `woodr/lerobot_teaching_HS_PicknPlace_1004` | UR5e 실물, 카메라 4대, 다른 장면/행동공간 |
| `IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot` | 실물 UR5, 다른 태스크·관측 |

근본 이유는 **HIL-SERL 데이터셋이 환경과 1:1 로 맞아야** 하기 때문이다:

```
관측 키    observation.images.front · observation.images.wrist · observation.state
상태 차원  16 (UR5e)
행동       [dx, dy, dz, gripper]  — 작업공간 델타. 관절 목표가 아니다
fps        20
장면       블록 크기·목표 패드 위치·카메라 위치까지 동일해야 정규화 통계가 맞는다
```

하나라도 다르면 못 쓴다. **애초에 HIL-SERL 의 설계가 "자기 환경에서 20~30판 직접
녹화한다" 는 전제다**(논문도 그렇게 한다). 이 프로젝트는 43판을 썼고, 녹화에 약 1시간 걸렸다.

녹화 도구는 **이미 다 있다** — 로봇이 바뀌어도 절차와 조작 키가 같다:

```bash
bash scripts/record.sh ur5e_01                   # 세션별로 여러 번
python3 scripts/merge_demos.py --success-only    # 성공판만 병합
python3 scripts/update_dataset_stats.py --repo_id ... --root ... --train_config ...
```

---

### 2.9 카메라 구성 — 지금 2대. 늘릴 때의 비용

| 카메라 | 어디 | 정의된 곳 |
| --- | --- | --- |
| `front` | 장면 전경 (`pos="1.3 0 0.7"`, fovy 45) | `place_cube_scene.xml` — **로봇 무관** |
| `handcam_rgb` | 손목 | `panda.xml` 의 `attachment` 서브트리 — 그리퍼와 함께 따라옴 |

둘 다 128×128 로 정책에 들어간다. UR5e 로 바꿔도 **`front` 는 그대로**,
`handcam_rgb` 는 ④ 접붙이기로 따라온다. **카메라 때문에 추가로 할 일은 없다.**

천장 뷰를 하나 더 다는 것은 가능하지만 공짜가 아니다:

```
카메라 1대 = 128×128×3 float32 = 196 KB / 전이
전이 1개가 400 KB → 600 KB (+50%)
버퍼 40,000 × 2 기준 GPU 메모리 32 GB → 48 GB
```

그리고 `policy.input_features` 에 키를 추가하고 시연을 다시 녹화해야 한다.
**먼저 2대로 UR5e 를 성립시키고, 성능이 모자랄 때 늘리는 것을 권한다**
(Panda 는 2대로 100% 를 달성했다).

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
2. `gym_hil/assets/ur5e.xml` 작성 — **§2.7 의 접붙이기(graft) 절차 ①~⑥ 을 따른다.**
   요점만: `panda.xml` 을 복사해 팔 부분만 UR5e 로 바꾸고,
   `<body name="attachment">` 서브트리(손목카메라 + 그리퍼 + `pinch` site)를
   **통째로** UR5e 손목 밑으로 옮긴다. `<contact>`/`<tendon>`/`<equality>`/
   `fingers_actuator` 는 손대지 않는다.
   액추에이터는 반드시 `<motor>` 로 (Menagerie 기본은 `position`)
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

### Phase 5 — `--robot` CLI (**UR5e 가 돌아간 뒤에**)

#### 요구사항 (사용자 결정, 2026-10-23)

- 명령은 로봇과 상관없이 **똑같이** 친다. 바뀌는 것은 `--robot` 값 하나뿐이다.
- `--robot` 은 **필수. 기본값 없음.**
- 그 값 하나로 로봇에 맞는 config · 시연 데이터 · 출력 폴더 · gym env 가 전부 정해진다.

```bash
# Panda                                             # UR5e
bash scripts/train.sh learner --robot panda          bash scripts/train.sh learner --robot ur5e
bash scripts/train.sh actor   --robot panda          bash scripts/train.sh actor   --robot ur5e
bash scripts/train.sh status  --robot panda          bash scripts/train.sh status  --robot ur5e
bash scripts/train.sh stop    --robot panda          bash scripts/train.sh stop    --robot ur5e
bash scripts/record.sh 01     --robot panda          bash scripts/record.sh 01     --robot ur5e
bash scripts/teleop.sh        --robot panda          bash scripts/teleop.sh        --robot ur5e
```

#### 적용 범위

| 명령 | `--robot` |
| --- | --- |
| `train.sh` 의 **모든** 하위 명령 — learner · actor · new · restart · status · pause · resume · save · stop | **필수** |
| `record.sh`, `teleop.sh` | **필수** |
| `dashboard.sh`, `view.py` | 받지 않음 — 보기 전용이고, 돌고 있는 learner 를 따라간다 |

`--robot` 없이 치면 **사용 가능한 로봇 목록을 보여주고 실패**한다.

> **정정**: 이전 판(2026-10-06)에는 status · pause · resume · save · stop 이
> "돌고 있는 것을 따라가므로 `--robot` 을 받지 않는다" 고 적었다. **틀렸다.**
> 지금 코드는 그렇게 동작하지 않는다(아래). 사용자 요구대로 전부 필수로 한다.

#### 왜 status · save · stop 에도 필수여야 하나 — 지금 코드의 함정

```bash
# scripts/train.sh
31:  CFG="${CFG:-configs/train_config_panda_place_gpu.json}"   # 기본값이 Panda
34:  OUT="$(... CFG 의 output_dir ...)"
309: touch "$OUT/SAVE_NOW"                                     # save
316: touch "$OUT/STOP_NOW"                                     # stop
```

`save` · `stop` · `status` 는 **돌고 있는 learner 를 찾아가지 않는다.**
`CFG` 의 `output_dir` 을 볼 뿐이고, `CFG` 를 안 주면 **Panda 설정**이다.
그리고 learner 는 트리거 파일을 **처리할 때만** 지우고, 시작할 때는 정리하지 않는다
(`learner.py:521`, `:527`).

로봇이 둘이면 이렇게 된다:

1. UR5e learner 가 돌고 있다.
2. `bash scripts/train.sh stop` — `CFG` 를 안 붙였다
   → `outputs/train/panda_place_gpu/STOP_NOW` 가 생긴다. 화면에는 "요청했다". **성공처럼 보인다.**
3. UR5e learner 는 **멈추지 않는다.**
4. 며칠 뒤 Panda learner 를 이어서 띄운다 → 첫 스텝에서 남아 있던 `STOP_NOW` 를 보고
   **바로 저장하고 종료한다.** 원인을 찾기 매우 어렵다.

로봇이 하나인 지금도 `CFG=...03_exp_b_assist.json` 으로 learner 를 띄우고
`stop` 을 `CFG` 없이 치면 같은 일이 생긴다.
(2026-10-23 확인: 남아 있는 트리거 파일은 없다. 피해는 아직 없었다.)

#### Phase 5 에서 같이 넣을 가드

| 가드 | 내용 |
| --- | --- |
| **로봇 일치** | learner 가 돌고 있는데 그 config 의 로봇 ≠ `--robot` 이면 **거부**하고, 무엇이 돌고 있는지 알려준다. actor · status · pause · resume · save · stop 에 적용 |
| **대상 확인** | `save` · `stop` 은 대상 로봇의 learner 가 **실제로 돌고 있을 때만** 트리거 파일을 놓는다. 안 돌면 "그 로봇의 learner 가 없다" 로 실패 — 남은 파일이 다음 실행을 죽이는 일을 막는다 |
| (선택) 시작 시 정리 | learner 가 시작할 때 남은 `SAVE_NOW` · `STOP_NOW` 를 지우고 경고한다. `learner.py` 를 고쳐야 하므로 **lerobot 패치 0002 재생성 + 재현 점검**이 따라온다. 위 두 가드로 충분하면 생략 |

#### 로봇 키 — 내부에서는 하나로 통일한다

| | Panda | UR5e |
| --- | --- | --- |
| `--robot` 값 | **`panda`** (별칭 `franka-panda` 도 받는다) | **`ur5e`** |
| 설정 폴더 | `configs/robots/panda/` | `configs/robots/ur5e/` |
| 출력 폴더 접두사 | `panda_…` | `ur5e_…` |
| gym id | `PandaPlaceCube…-v0` | `UR5ePlaceCube…-v0` |

키를 `panda` 로 하는 이유: 이미 만든 것들 — `outputs/train/panda_place_gpu`,
`panda_place_smoke`, `train.sh new` 의 자동 접두사 — 이 전부 `panda` 를 쓴다.
`franka-panda` 로 치고 싶으면 별칭으로 받는다. **한 로봇에 이름이 둘이면** 폴더 · 로그 ·
대시보드에서 헷갈리므로 내부 키는 하나만 둔다.

```
configs/robots/
  panda/ {train,record,teleop}.json
  ur5e/  {train,record,teleop}.json
```

`CFG=` 는 일회성 설정을 가리키는 탈출구로 남긴다. `CFG` 를 주면 `--robot` 은 생략할 수 있고,
**둘 다 주면 서로 맞아야 한다**(로봇이 다르면 거부).

#### 기본값을 두지 않는 이유

기본값은 조용한 선택이다. UR5e 를 의도하고 `train.sh learner` 를 쳤는데 Panda 가 떠서
20분 뒤에 알아차리는 것이 가장 흔한 사고다. 위의 `stop` 함정도 결국 "기본값이 조용히 Panda 를
골랐다" 가 원인이다. 40시간짜리 학습에서 `--robot ur5e` 를 더 치는 비용은 없는 것과 같다.

#### 왜 Phase 5 가 마지막인가

로봇이 하나뿐일 때 이 인터페이스를 만들면 **"로봇마다 무엇이 다른가"** 를 추측으로 정하게 된다.
Phase 3 에서 `cartesian_bounds` 를 config 로 올려야 한다고 밝혀질 수도 있다(§5).
UR5e 를 `CFG=` 로 한 번 돌려 보면 차이가 전부 드러나고, 그 뒤엔 위 표를 그대로 구현하면 된다.
**`CFG=` 로 이미 기능은 100% 가능하다** — Phase 5 는 편의와 안전장치다.

#### 완료 기준

1. 위 명령 표의 Panda · UR5e 명령이 각각 맞는 config · 데이터 · 출력 폴더로 간다
2. `--robot` 을 빼면 로봇 목록과 함께 실패한다
3. UR5e learner 가 도는 중에 `train.sh stop --robot panda` → **거부**된다
4. Panda 와 UR5e 명령은 **`--robot` 값 말고는 글자 하나도 다르지 않다**
5. 기존 Panda 산출물(`panda_place_gpu`, `panda_place_smoke`, `03_exp_b_assist`)을 그대로 이어서 쓸 수 있다

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
| **7축 → 6축: 여유자유도가 사라진다** | 아래 6.1 | `nullspace_stiffness` 를 0 으로. teleop 으로 거동 확인 |
| **특이자세에서 거동이 튄다** | 아래 6.2 | `_CARTESIAN_BOUNDS` · home 자세로 특이점 영역을 피한다 |
| 액추에이터가 position 이면 조용히 이상 동작 | opspace 가 토크를 내는데 position 으로 해석된다 | Phase 2 완료 기준(중력에 처지지 않는가)으로 걸러진다 |
| 패치를 다시 안 뜸 | clone 한 사람이 다른 코드를 받는다. **실제로 전에 이 사고가 있었다** | 각 Phase 완료 기준에 "패치 재현 점검 차이 0건" 포함 |
| 시연 재사용 시도 | 상태 차원(18→16)과 동역학이 달라 **재사용 불가** | UR5e 시연을 새로 녹화. 태스크·조작 키는 같아서 절차는 동일 |
| learner / actor 설정 어긋남 | 상태 차원이 달라 즉시 깨지거나 조용히 이상해진다 | `train.sh actor` 가 돌고 있는 learner 와 config 경로가 다르면 **중단한다** (`HIL_FORCE=1` 로만 강행) |

---

### 6.1 7축 Panda 와 6축 UR5e 의 차이 — "남는 자유도"가 없어진다

**작업공간 목표는 6개 값이다.** 손끝의 위치 3개(x, y, z) + 자세 3개(회전).

| | 관절 수 | 목표 | 남는 것 |
| --- | --- | --- | --- |
| Panda | **7** | 6 | **1개** ← 여유자유도 |
| UR5e | **6** | 6 | **0개** |

Panda 는 손끝을 목표에 맞추고도 관절 하나가 남는다. 그래서 **손끝은 그대로 둔 채
팔꿈치만 위아래로 돌릴 수 있다.** 이 "손끝에 영향을 주지 않는 움직임들의 집합" 을
**null space(영공간)** 라고 부른다.

`opspace()` 는 그 남는 자유도를 "home 자세 쪽으로 당기는" 데 쓴다.
`mujoco_gym_env.py:254` 가 넘기는 `joint=self._home_position` 이 그 목표고,
`nullspace_stiffness`(기본 0.5)가 당기는 세기다. 코드로는:

```python
# opspace.py:282-283
Jnull = M_inv @ J.T @ Mx
tau += (np.eye(len(q)) - J.T @ Jnull.T) @ ddq     # ← null space 로 투영해서 더한다
```

`(I - Jᵀ Jnullᵀ)` 가 **"손끝을 건드리지 않는 방향만 통과시키는 필터"** 다.
이걸 통과한 토크만 더하므로, 팔꿈치 자세를 정리하면서도 손끝 추종은 망가지지 않는다.

**UR5e(6축)에서는 통과시킬 방향이 없다.** 그 필터가 거의 0 행렬이 되어 항이 통째로
사라진다. 수학적으로는 "자동으로 꺼지는" 것이라 **그 자체가 틀린 동작은 아니다.**
문제는 두 가지다:

1. `joint=home_position` 으로 자세를 정리하던 효과가 없어진다 →
   팔이 이상한 자세(팔꿈치가 뒤집히는 등)로 가도 되돌릴 힘이 없다.
2. 6.2 의 특이자세 근처에서 `Jnull` 계산이 불안정해져 **0 이 아니라 큰 값**이 섞여
   들어올 수 있다. 꺼져야 할 항이 노이즈를 주입하는 셈이다.

**대처**: Phase 3 에서 `nullspace_stiffness=0.0` 으로 두고 시작한다. 어차피 쓸 수 없는
기능이고, 꺼 두면 2번 위험이 사라진다. 자세 정리는 `_CARTESIAN_BOUNDS` 와 home 자세로 한다.

---

### 6.2 특이자세(singularity) — UR5e 가 Panda 보다 잘 걸린다

**특이자세**란 팔이 **어떤 방향으로는 더 이상 움직일 수 없게 되는 자세**다.
팔을 앞으로 쭉 뻗으면 그 방향으로 더 못 나가는 것과 같다.
수학적으로는 자코비안 `J` 가 랭크를 잃는 것이고, 그러면 작업공간 질량행렬이 역행렬을 잃는다:

```python
# opspace.py:267-268
Mx_inv = J @ M_inv @ J.T
Mx = np.linalg.inv(Mx_inv) if abs(np.linalg.det(Mx_inv)) >= 1e-2 else np.linalg.pinv(Mx_inv, rcond=1e-2)
                                                                      ^^^^^^^^^^^^^^
```

행렬식이 `1e-2` 밑으로 떨어지면 `inv` 대신 `pinv`(유사역행렬)로 넘어간다.
**이 전환은 연속적이지 않다.** 그 경계를 오갈 때 토크가 계단처럼 바뀌어
**팔이 홱 움직인다.** 조용히 일어나고 에러도 안 난다.

UR5e 가 더 잘 걸리는 이유:

| | |
| --- | --- |
| Panda (7축) | 남는 자유도로 **특이자세를 우회**할 수 있다. 손끝은 목표대로 가면서 팔꿈치만 돌려서 빠져나간다 |
| UR5e (6축) | 우회로가 없다. 그 자세를 지나야 하면 **그냥 통과해야 한다** |

UR5e 는 특히 세 가지가 알려져 있다:

| 특이점 | 언제 | 피하는 법 |
| --- | --- | --- |
| **어깨(shoulder)** | 손목이 베이스 회전축 바로 위/아래를 지날 때 | 작업 영역을 로봇 정면으로 치우치게 (`x` 를 0 에서 떨어뜨림) |
| **팔꿈치(elbow)** | 팔을 완전히 쭉 폄 | 도달 범위 끝까지 쓰지 않는다. `_CARTESIAN_BOUNDS` 를 안쪽으로 |
| **손목(wrist)** | 4번·6번 관절 축이 일직선 | home 자세에서 손목을 미리 틀어 둔다 |

**대처 (Phase 3 에서 teleop 으로 확인)**

1. `_CARTESIAN_BOUNDS` 를 **특이점 영역 밖**으로 잡는다.
   Panda 값 `[[0.2,-0.3,0],[0.6,0.3,0.5]]` 을 그대로 쓰지 말 것 —
   UR5e 의 도달 범위(약 850 mm)와 베이스 위치가 다르다.
2. `_UR5E_HOME` 을 **팔꿈치가 충분히 굽은** 자세로 잡는다. 쭉 편 자세에서 시작하면
   첫 스텝부터 특이점이다.
3. teleop 으로 작업 영역 **구석구석을 손으로 돌려 본다.** 홱 튀는 지점이 있으면
   그 영역을 bounds 에서 잘라낸다. 학습을 시작하기 전에 해야 한다 —
   정책은 그 지점을 알아서 피해 주지 않는다.

> 이것이 Phase 3 의 완료 기준을 "사람이 teleop 으로 집어서 놓을 수 있다" 로 둔 이유다.
> 사람이 조작하다 튀는 것을 느끼면 거기가 특이점이다. 학습으로는 그걸 못 찾는다 —
> 성공률만 안 오르고 왜인지 알 수 없게 된다.

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
