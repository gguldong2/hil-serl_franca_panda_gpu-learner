#!/usr/bin/env bash
# [UR5e 메시] MuJoCo Menagerie 의 universal_robots_ur5e 를 고정 커밋으로 받아 gym-hil 자산 폴더에 둔다.
# 사용법: bash scripts/server22/fetch_ur5e_assets.sh [대상 폴더]
#   기본 대상: third_party/gym-hil/gym_hil/assets/universal_robots_ur5e
#   02_setup_project.sh 와 docs/patches/README.md 의 재현 점검이 부른다.
#
# 메시(OBJ 20개, 약 30 MB)는 패치에 넣지 않는다. lerobot·gym-hil 과 같은 이유다 —
# 복사본은 조용히 어긋나고, 고정 커밋에서 받으면 어긋날 수가 없다.
# 이 폴더를 참조하는 것은 gym-hil 패치 0006 의 assets/ur5e.xml 이다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
MENAGERIE_SHA=cda5236f98383f6fbaa4820c6b18dba892cd087f
DEST="${1:-third_party/gym-hil/gym_hil/assets/universal_robots_ur5e}"

if [ -f "$DEST/assets/wrist3.obj" ] && [ "$(ls "$DEST/assets"/*.obj | wc -l)" -eq 20 ]; then
  echo "  UR5e 메시 이미 있음: $DEST"; exit 0
fi
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
git clone -q --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git "$T/menagerie"
git -C "$T/menagerie" sparse-checkout set universal_robots_ur5e
git -C "$T/menagerie" checkout -q "$MENAGERIE_SHA"
SRC="$T/menagerie/universal_robots_ur5e"
mkdir -p "$DEST/assets"
cp "$SRC"/assets/*.obj "$DEST/assets/"
cp "$SRC/LICENSE" "$SRC/README.md" "$SRC/CHANGELOG.md" "$DEST/"
echo "  UR5e 메시 받음: $DEST (menagerie ${MENAGERIE_SHA:0:8}, $(ls "$DEST/assets"/*.obj | wc -l) 개)"
