#!/bin/bash
# 챔피언(32B DPO ckpt600) 균형 8뷰(BALANCED8 = Klein V + 역원닫힌 코셋) test 819건 추론.
#   근거: TTA3→TTA4 실LB +0.524pp는 "균형 구조" 효과(tta_report_2026-07-20.md §1),
#   2026-07-21 26건 육안 감수에서 랜덤계 다뷰 합의가 TTA4 노이즈를 13:8로 교정
#   → 균형 유지 + 뷰 2배(분산 절반)가 두 효과의 원리적 결합. 예상 소요 ~2.2h(4090).
#   산출: runs/test_champ_tta8b/submission.csv (LB 제출은 사용자 슬롯 판단)
# 중단 시 같은 명령 재실행이면 progress.jsonl에서 이어서 진행.

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
PY=python3
MODEL=unsloth/Qwen3-VL-32B-Instruct-bnb-4bit
ADAPTER=runs/DPO-checkpoint-600
OUT=runs/test_champ_tta8b

if [ -f "$OUT/report.json" ]; then
  echo "[tta8b] 이미 완료 — $OUT/report.json 존재"; exit 0
fi
echo "[tta8b] ===== 균형 8뷰 시작 $(date '+%F %T') ====="
"$PY" -m snuai.infer.predict --csv data/test.csv --image-dir data/test \
    --strategy score24 --tta 8 --tta-balanced8 \
    --model-id "$MODEL" --adapter "$ADAPTER" \
    --out "$OUT"
echo "[tta8b] 완료 $(date '+%F %T') — 채점: python3 ../grade/grade.py $OUT/submission.csv"
