#!/bin/bash
# 캐스케이드(2단계 pairwise 재검증) — test 819건 실측, 사용자 지시(2026-07-21)로
# 홀드아웃 대신 실제 test 데이터 + ../grade/grade.py 참고채점으로 전환.
#   baseline은 이미 있음(runs/test_champ_tta8b/submission.csv, 현 챔피언 LB 0.93019
#   실측) — 재실행 없이 그대로 기준으로 재사용. cascade 변형만 새로 돌려서
#   grade.py(자기참조 편향 있는 로컬 참고채점) + score_agreement.py(진짜 의미있는
#   champion 대비 일치율)로 비교.
#   tau=0.15는 예전 버전 기본값이라 미검증 — 이 probe로 실측 후 LB 제출 여부는
#   사용자 판단(하루 2회 제한).
# 산출: runs/test_champ_tta8b_cascade/submission.csv + 비교 로그
# 중단 시 같은 명령 재실행이면 progress.jsonl에서 이어서 진행.

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
PY=python3
MODEL=unsloth/Qwen3-VL-32B-Instruct-bnb-4bit
ADAPTER=runs/DPO-checkpoint-600
BASE_SUB=runs/test_champ_tta8b/submission.csv
OUT=runs/test_champ_tta8b_cascade

if [ ! -f "$BASE_SUB" ]; then
  echo "[cascade-test] baseline 없음: $BASE_SUB — 중단"; exit 1
fi

if [ -f "$OUT/report.json" ]; then
  echo "[cascade-test] 이미 완료 — $OUT/report.json 존재"
else
  echo "[cascade-test] ===== TTA8-balanced + cascade(tau 0.15) test 819건 시작 $(date '+%F %T') ====="
  "$PY" -m snuai.infer.predict --csv data/test.csv --image-dir data/test \
      --strategy score24 --tta 8 --tta-balanced8 --cascade --tau 0.15 \
      --model-id "$MODEL" --adapter "$ADAPTER" \
      --out "$OUT"
  RC=$?
  if [ $RC -ne 0 ]; then echo "[cascade-test] 실패(rc=$RC) — 중단"; exit $RC; fi
fi

echo "[cascade-test] ===== 비교 $(date '+%F %T') ====="
echo "--- grade.py (자기참조 편향 있음, 참고용) ---"
"$PY" ../grade/grade.py "$OUT/submission.csv" --show-wrong 0 | tee "$OUT/grade.txt"
echo "--- score_agreement.py (champion 대비 진짜 의미있는 비교) ---"
"$PY" scripts/score_agreement.py --champion "$BASE_SUB" --candidates "$OUT/submission.csv" \
    --out "$OUT/agreement_vs_tta8b.json" --print-best
echo "[cascade-test] 완료 $(date '+%F %T')"
