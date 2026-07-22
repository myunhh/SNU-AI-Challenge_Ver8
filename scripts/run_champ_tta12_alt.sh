#!/bin/bash
# 챔피언(32B DPO ckpt600) 교대군 A4(짝순열 전부 12개, ALTERNATING12) test 819건 추론.
#   근거: TTA24(S4 전체, 짝12+홀12 혼합)가 실LB에서 TTA8-balanced(BALANCED8=D4, 짝4+홀4
#   혼합, 현 챔피언 LB 0.93019)에 패했다. "홀순열(원본과 크게 어긋난 배열)이 모델을
#   헷갈리게 한다"는 가설이 맞다면, 홀순열을 아예 안 쓰는 A4가 D4보다 나을 수 있다.
#   A4도 D4보다 못하면 그 가설 자체가 기각되고 "8 근처가 최적점" 쪽으로 기운다.
#   ⚠️ V(4)→D4(8)→S4(24) 계열의 자연스러운 연장이 아니라 별개 구조(src/snuai/infer/tta.py
#   ALTERNATING12 주석 참고) — 균형(각 입력이 각 슬롯 정확히 3회)은 property test로 확인됨.
#   예상 소요: TTA8 실측 ~9.6s/샘플의 12/8배 ≈ ~14.4s/샘플 × 819건 ≈ 3.3h.
#   산출: runs/test_champ_tta12_alt/submission.csv (LB 제출은 사용자 슬롯 판단)
# 중단 시 같은 명령 재실행이면 progress.jsonl에서 이어서 진행.

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
PY=python3
MODEL=unsloth/Qwen3-VL-32B-Instruct-bnb-4bit
ADAPTER=runs/DPO-checkpoint-600
OUT=runs/test_champ_tta12_alt

if [ -f "$OUT/report.json" ]; then
  echo "[tta12-alt] 이미 완료 — $OUT/report.json 존재"; exit 0
fi
echo "[tta12-alt] ===== 교대군 A4(짝순열 12개) 시작 $(date '+%F %T') ====="
"$PY" -m snuai.infer.predict --csv data/test.csv --image-dir data/test \
    --strategy score24 --tta 12 \
    --model-id "$MODEL" --adapter "$ADAPTER" \
    --out "$OUT"
RC=$?
if [ $RC -ne 0 ]; then echo "[tta12-alt] 실패(rc=$RC) — 중단"; exit $RC; fi
echo "[tta12-alt] 완료 $(date '+%F %T') — 채점: python3 ../grade/grade.py $OUT/submission.csv"

BASE_SUB=runs/test_champ_tta8b/submission.csv
if [ -f "$BASE_SUB" ]; then
  echo "[tta12-alt] ===== 챔피언(TTA8-balanced) 대비 일치율 $(date '+%F %T') ====="
  "$PY" scripts/score_agreement.py --champion "$BASE_SUB" --candidates "$OUT/submission.csv" \
      --out "$OUT/agreement_vs_tta8b.json" --print-best
fi
