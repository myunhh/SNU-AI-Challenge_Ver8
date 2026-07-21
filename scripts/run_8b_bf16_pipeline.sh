#!/bin/bash
# 8B bf16 트랙 전체 파이프라인 (2026-07-19, 사용자 결정: 채점·선택은 사용자가 직접)
#
#   1) SFT 2000스텝 (train 100%, 홀드아웃 없음, ckpt 200마다)
#   2) SFT 전 ckpt 스윕 추론: ckpt200~2000(10개) × test 819건 (score24+TTA3, bf16 서빙)
#      → runs/test_8b_bf16_sft_ckptN/submission.csv (사용자가 채점)
#   3) ⏸ 사용자 선택 대기: runs/dpo8b_bf16_v8/base_adapter.txt 가 생기면 자동 재개
#      (내용: 체크포인트 경로 또는 스텝 숫자만. 예: echo 1600 > runs/dpo8b_bf16_v8/base_adapter.txt)
#   4) 선택된 SFT ckpt를 base로 DPO 1000스텝 — **업그레이드 레시피** (2026-07-19 사용자 지시):
#      hard-negative(인접 스와프 3종 중 현재 모델 최고점 오답, 사전 스코어링 ~30-60분)
#      + --reversal-rejected(완전역전 d=6, ckpt600 오답 분석 기반). 챔피언 레시피
#      (--no-hard-negative)와 다름 — 이 조합의 첫 실측.
#   5) DPO ckpt 스윕 추론: ckpt200~800 + final → runs/test_8b_bf16_dpo_*/submission.csv
#
# 중단 시 같은 명령 재실행이면 각 단계가 이어서 진행됨(체크포인트/report.json/progress.jsonl).

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/py3_11/lib
PY=$HOME/anaconda3/envs/py3_11/bin/python
MODEL=/home/yhmin/model/hub/Qwen3-VL-8B-Instruct
SFT_OUT=runs/sft8b_bf16_v8
DPO_OUT=runs/dpo8b_bf16_v8
BASE_RECORD="$DPO_OUT/base_adapter.txt"

infer() { # adapter_dir out_dir
  local ADAPTER=$1 OUT=$2
  if [ -f "$OUT/report.json" ]; then
    echo "[pipeline] $OUT 이미 완료 — 건너뜀"; return 0
  fi
  if [ ! -d "$ADAPTER" ]; then
    echo "[pipeline] 어댑터 없음: $ADAPTER — 건너뜀"; return 1
  fi
  echo "[pipeline] ===== 추론 $OUT 시작 $(date '+%F %T') ====="
  "$PY" -m snuai.infer.predict --csv data/test.csv --image-dir data/test \
      --strategy score24 --tta 3 --model-id "$MODEL" \
      --adapter "$ADAPTER" --out "$OUT"
}

# ── 1) SFT ──────────────────────────────────────────────────────────────
if [ -f "$SFT_OUT/adapter_final/adapter_model.safetensors" ]; then
  echo "[pipeline] SFT 이미 완료 — 건너뜀"
else
  echo "[pipeline] ===== SFT 시작 $(date '+%F %T') ====="
  "$PY" -m snuai.train.train_sft --csv data/train.csv --image-dir data/train \
      --model-id "$MODEL" --precision bf16 --val-frac 0 \
      --max-steps 2000 --save-steps 200 --grad-accum 16 \
      --out "$SFT_OUT" --resume
  RC=$?; [ $RC -ne 0 ] && { echo "[pipeline] SFT 실패 (rc=$RC) — 중단"; exit $RC; }
  echo "[pipeline] ===== SFT 완료 $(date '+%F %T') ====="
fi

# ── 2) SFT 전 ckpt 스윕 추론 (10개) ─────────────────────────────────────
for N in 200 400 600 800 1000 1200 1400 1600 1800 2000; do
  infer "$SFT_OUT/checkpoint-$N" "runs/test_8b_bf16_sft_ckpt$N"
done
echo "[pipeline] SFT 스윕 추론 완료 — 제출물: runs/test_8b_bf16_sft_ckpt*/submission.csv"

# ── 3) 사용자 채점·선택 대기 ────────────────────────────────────────────
if [ ! -f "$BASE_RECORD" ]; then
  echo "[pipeline] ⏸ 사용자 선택 대기 중 — 채점 후 아래처럼 base를 지정하면 자동 재개:"
  echo "[pipeline]     echo 1600 > $BASE_RECORD          (스텝 숫자)"
  echo "[pipeline]     또는 echo $SFT_OUT/checkpoint-1600 > $BASE_RECORD  (경로)"
  while [ ! -f "$BASE_RECORD" ]; do sleep 60; done
fi
SEL=$(head -1 "$BASE_RECORD" | tr -d '[:space:]')
case "$SEL" in
  ''|*[!0-9]*) BEST_ADAPTER="$SEL" ;;                 # 경로로 해석
  *)           BEST_ADAPTER="$SFT_OUT/checkpoint-$SEL" ;;  # 숫자면 스텝으로 해석
esac
[ -d "$BEST_ADAPTER" ] || { echo "[pipeline] base 어댑터 없음: $BEST_ADAPTER (base_adapter.txt 확인)"; exit 1; }
echo "[pipeline] 사용자 선택 base = $BEST_ADAPTER"

# ── 4) DPO (선택된 SFT ckpt base, hard-negative+reversal 업그레이드) ────
if [ -f "$DPO_OUT/adapter_final/adapter_model.safetensors" ]; then
  echo "[pipeline] DPO 이미 완료 — 건너뜀"
else
  echo "[pipeline] ===== DPO 시작 $(date '+%F %T') base=$BEST_ADAPTER (hard-negative+reversal) ====="
  "$PY" -m snuai.train.train_dpo --csv data/train.csv --image-dir data/train \
      --model-id "$MODEL" --adapter "$BEST_ADAPTER" \
      --val-frac 0 --hard-negative --reversal-rejected \
      --max-steps 1000 --save-steps 200 --grad-accum 16 \
      --out "$DPO_OUT" --resume
  RC=$?; [ $RC -ne 0 ] && { echo "[pipeline] DPO 실패 (rc=$RC) — 중단"; exit $RC; }
  echo "[pipeline] ===== DPO 완료 $(date '+%F %T') ====="
fi

# ── 5) DPO ckpt 스윕 추론 ───────────────────────────────────────────────
for N in 200 400 600 800; do
  infer "$DPO_OUT/checkpoint-$N" "runs/test_8b_bf16_dpo_ckpt$N"
done
infer "$DPO_OUT/adapter_final" "runs/test_8b_bf16_dpo_final"

echo "[pipeline] 전체 완료 $(date '+%F %T')"
echo "[pipeline] 제출 후보: runs/test_8b_bf16_dpo_*/submission.csv (채점·LB 제출은 사용자 판단)"
