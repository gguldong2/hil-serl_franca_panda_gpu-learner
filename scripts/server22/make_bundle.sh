#!/usr/bin/env bash
# [번들 만들기] 서버로 옮길 파일만 묶는다: README, configs, scripts, docs(패치 포함). venv/third_party/outputs/cache 는 제외.
# 사용법: bash scripts/server22/make_bundle.sh  → hil-serl-bundle.tgz (프로젝트 루트)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
OUT="hil-serl-bundle.tgz"
tar czf "$OUT" --exclude='docs/requirements-lock.server22.txt' README.md configs scripts docs
echo "생성: $ROOT/$OUT ($(du -h "$OUT" | cut -f1))"
echo "서버로:  scp $OUT <user>@<server>:~/   →  서버에서:  mkdir -p ~/project/hil-serl && tar xzf ~/$OUT -C ~/project/hil-serl"
