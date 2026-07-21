#!/bin/bash
# DPO 체크포인트 test 819건 순차 추론 체인 (2026-07-17)
# ckpt400 → 600 → 800 → 1000. 챔피언 서빙 레시피 동일: score24 + TTA3 + 32B-prequant 4bit.
# 중단돼도 같은 명령 재실행이면 progress.jsonl에서 이어서 진행됨.

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/py3_11/lib
PY=$HOME/anaconda3/envs/py3_11/bin/python
MODEL=/home/yhmin/model/hub/Qwen3-VL-32B-Instruct-bnb-4bit

for N in 400 600 800 1000; do
  OUT="runs/test_dpo_ckpt$N"
  if [ -f "$OUT/report.json" ]; then
    echo "[chain] ckpt$N 이미 완료 (report.json 존재) — 건너뜀"
    continue
  fi
  echo "[chain] ===== ckpt$N 시작 $(date '+%F %T') ====="
  "$PY" -m snuai.infer.predict --csv data/test.csv --image-dir data/test \
      --strategy score24 --tta 3 \
      --model-id "$MODEL" \
      --adapter "runs/DPO-checkpoint-$N" \
      --out "$OUT"
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "[chain] ckpt$N 실패 (rc=$RC) — 다음 체크포인트로 진행"
    continue
  fi
  echo "[chain] ===== ckpt$N 완료 $(date '+%F %T') ====="
done

echo "[chain] 전체 완료 $(date '+%F %T')"
