#!/bin/bash
# 챔피언(32B DPO ckpt600) 완전균형 24뷰(BALANCED24 = S4 전체) test 819건 추론.
#   근거: BALANCED8 실측 확인 결과 실제 부분군(Sylow-2, 8차 이면군)이었다 — 즉
#   Klein 4원군(4차 정규부분군)->Sylow-2(8차)로 이어지는 "대수적 완전성 강화" 계열의
#   자연스러운 종점이 전체군 S4(24차) 그 자체. TTA4->TTA8 실LB +1.22pp(사전 추정보다
#   컸음, ../grade/grade.py 주석 참고)가 "뷰 개수"가 아니라 "완전성" 덕이라면 다음
#   실측 대상은 이거뿐 — 더 큰 균형 부분집합이 없다(24가 전체).
#   예상 소요: TTA8 실측 ~9.6s/샘플의 24/8배 ≈ ~29s/샘플 × 819건 ≈ 6.6h. 24h 예산 여유.
#   산출: runs/test_champ_tta24/submission.csv (LB 제출은 사용자 슬롯 판단)
# 중단 시 같은 명령 재실행이면 progress.jsonl에서 이어서 진행.

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/py3_11/lib
PY=$HOME/anaconda3/envs/py3_11/bin/python
MODEL=/home/yhmin/model/hub/Qwen3-VL-32B-Instruct-bnb-4bit
ADAPTER=runs/DPO-checkpoint-600
OUT=runs/test_champ_tta24

if [ -f "$OUT/report.json" ]; then
  echo "[tta24] 이미 완료 — $OUT/report.json 존재"; exit 0
fi
echo "[tta24] ===== 완전균형 24뷰(S4 전체) 시작 $(date '+%F %T') ====="
"$PY" -m snuai.infer.predict --csv data/test.csv --image-dir data/test \
    --strategy score24 --tta 24 \
    --model-id "$MODEL" --adapter "$ADAPTER" \
    --out "$OUT"
echo "[tta24] 완료 $(date '+%F %T') — 채점: python ../grade/grade.py $OUT/submission.csv"
