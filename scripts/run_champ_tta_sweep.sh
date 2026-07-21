#!/bin/bash
# 챔피언(32B DPO ckpt600) TTA 스윕 (2026-07-20 사용자 지시)
#   TTA3 = 기존 runs/test_dpo_ckpt600 (07-18, 동일 레시피·결정적) — 재추론 안 함
#   TTA4(Klein 균형세트)~TTA8 을 test 819건에 순차 추론 → runs/test_champ_ttaN
#   채점은 상태보드(status_8b_chain.sh)가 grade.py로 자동 수행.
# 중단 시 같은 명령 재실행이면 이어서 진행(report.json/progress.jsonl resume).

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/py3_11/lib
PY=$HOME/anaconda3/envs/py3_11/bin/python
MODEL=/home/yhmin/model/hub/Qwen3-VL-32B-Instruct-bnb-4bit
ADAPTER=runs/DPO-checkpoint-600

for T in 4 5 6 7 8; do
  OUT=runs/test_champ_tta$T
  if [ -f "$OUT/report.json" ]; then
    echo "[tta-sweep] TTA$T 이미 완료 — 건너뜀"; continue
  fi
  echo "[tta-sweep] ===== TTA$T 시작 $(date '+%F %T') ====="
  "$PY" -m snuai.infer.predict --csv data/test.csv --image-dir data/test \
      --strategy score24 --tta "$T" \
      --model-id "$MODEL" --adapter "$ADAPTER" \
      --out "$OUT"
  RC=$?
  [ $RC -ne 0 ] && echo "[tta-sweep] TTA$T 실패 (rc=$RC) — 다음으로 진행"
done
echo "[tta-sweep] 전체 완료 $(date '+%F %T')"
