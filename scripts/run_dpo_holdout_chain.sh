#!/bin/bash
# DPO 체크포인트 홀드아웃 순차 평가 체인 (2026-07-17)
# ckpt400 → 600 → 800 → 1000 → ckpt200(챔피언 기준선, 사후 재검증 겸)
# 각 런: score24 + TTA3 + 32B-prequant 4bit (챔피언 서빙 레시피 동일)
# 런 종료마다 eval_report.py로 EM + 쌍순서(1-KT/6) 포렌식 리포트 생성.
# 중단돼도 같은 명령 재실행이면 progress.jsonl에서 이어서 진행됨.

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/py3_11/lib
PY=$HOME/anaconda3/envs/py3_11/bin/python
MODEL=/home/yhmin/model/hub/Qwen3-VL-32B-Instruct-bnb-4bit

declare -A JOBS=(
  [cal_dpo_ckpt400]="runs/DPO-checkpoint-400"
  [cal_dpo_ckpt600]="runs/DPO-checkpoint-600"
  [cal_dpo_ckpt800]="runs/DPO-checkpoint-800"
  [cal_dpo_ckpt1000]="runs/DPO-checkpoint-1000"
  [cal_dpo_ckpt200]="runs/checkpoint-200-Ver8 DPO"
)
ORDER=(cal_dpo_ckpt400 cal_dpo_ckpt600 cal_dpo_ckpt800 cal_dpo_ckpt1000 cal_dpo_ckpt200)

for NAME in "${ORDER[@]}"; do
  ADAPTER="${JOBS[$NAME]}"
  OUT="runs/$NAME"
  if [ -f "$OUT/report.json" ]; then
    echo "[chain] $NAME 이미 완료 (report.json 존재) — 건너뜀"
    continue
  fi
  echo "[chain] ===== $NAME 시작 (adapter=$ADAPTER) $(date '+%F %T') ====="
  "$PY" -m snuai.infer.predict --csv data/train.csv --image-dir data/train \
      --holdout-val --eval \
      --strategy score24 --tta 3 \
      --model-id "$MODEL" \
      --adapter "$ADAPTER" \
      --out "$OUT"
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "[chain] $NAME 실패 (rc=$RC) — 다음 체크포인트로 진행"
    continue
  fi
  "$PY" scripts/eval_report.py --progress "$OUT/progress.jsonl" \
      --csv data/train.csv --image-dir data/train --holdout-val --val-frac 0.1 \
      --out "$OUT/forensic.md" || echo "[chain] $NAME forensic 리포트 실패 (추론 결과는 보존됨)"
  echo "[chain] ===== $NAME 완료 $(date '+%F %T') ====="
done

echo "[chain] 전체 완료 $(date '+%F %T')"
