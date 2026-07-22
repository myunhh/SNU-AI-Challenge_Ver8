#!/bin/bash
# Ver8 dpo32b_v8_reversal 학습+평가 통합 대시보드. 사용법: bash scripts/dashboard_dpo_reversal.sh
# (읽기 전용 — 5초마다 화면을 갱신해서 상태를 보여줄 뿐, 아무것도 건드리지 않음)
#
# 2026-07-22 갱신: 학습(train_dpo)이 checkpoint-100(100/600 스텝)까지만 진행되고 프로세스가
# 죽어 정지된 상태 — 로그에 'Loading weights: 100%'(스코어링 재시작 신호)가 4회 나타나 크래시
# 후 재시작을 반복하다 checkpoint-100 저장 이후로는 더 진행되지 않은 것으로 보임. 현재는 학습을
# 이어가는 대신 checkpoint-100을 test 819건에 TTA3로 먼저 평가 중(runs/test_reversal_ckpt100).
# 그래서 이 대시보드는 학습 단계와 평가 단계를 함께 보여준다.
set -u
cd "$(dirname "$0")/.."

TRAIN_DIR="runs/dpo32b_v8_reversal"
TRAIN_LOG="$TRAIN_DIR/train.log"
EVAL_DIR="runs/test_reversal_ckpt100"
EVAL_LOG="runs/test_reversal_ckpt100.log"
MAX_STEPS=600
NTEST=$(( $(awk 'END{print NR}' data/test.csv 2>/dev/null || echo 1) - 1 ))
[ "$NTEST" -le 0 ] && NTEST=819

rep() { local n=$1 c=$2; [ "$n" -gt 0 ] && printf "$c%.0s" $(seq 1 "$n"); return 0; }
bar() { # cur total
  local cur=$1 total=$2 w=30
  [ "$total" -le 0 ] && total=1
  [ "$cur" -gt "$total" ] && cur=$total
  local f=$(( cur * w / total ))
  printf "[%s%s] %d/%d (%d%%)" "$(rep "$f" █)" "$(rep $((w - f)) ░)" "$cur" "$total" $(( cur * 100 / total ))
}

while true; do
  clear
  echo "================ Ver8 DPO reversal-rejected 대시보드 ================"
  echo "시각: $(date '+%Y-%m-%d %H:%M:%S')   (Ctrl+C로 종료, 아무 프로세스에도 영향 없음)"
  echo ""

  echo "── ① 학습: $TRAIN_DIR ──"
  if pgrep -f "snuai.train.train_dpo" > /dev/null; then
    ps aux | grep "train_dpo" | grep -v grep | awk '{printf "  실행 중  PID %s  CPU %s%%  MEM %s%%  경과 %s\n", $2, $3, $4, $10}'
    LAST_STEP=$(grep -oE "[0-9]+/${MAX_STEPS} \[" "$TRAIN_LOG" 2>/dev/null | tail -1 | grep -oE "^[0-9]+")
    [ -n "$LAST_STEP" ] && { printf "  진행 "; bar "$LAST_STEP" "$MAX_STEPS"; echo; }
  else
    echo "  ⛔ 프로세스 없음 (종료됨/크래시)"
    if [ -f "$TRAIN_LOG" ]; then
      LAST_STEP=$(grep -oE "[0-9]+/${MAX_STEPS} \[" "$TRAIN_LOG" | tail -1 | grep -oE "^[0-9]+")
      RESTARTS=$(grep -o "Loading weights: 100%" "$TRAIN_LOG" | wc -l)
      if [ -n "$LAST_STEP" ]; then
        printf "  마지막 도달 스텝 "; bar "$LAST_STEP" "$MAX_STEPS"; echo "  (여기서 정지)"
      fi
      [ "$RESTARTS" -gt 1 ] 2>/dev/null && echo "  ⚠ 재시작(가중치 재로딩) ${RESTARTS}회 감지 — 반복 크래시 후 진행 없이 정지된 것으로 보임"
      ERR=$(grep -aE "Traceback|CUDA out of memory|nan|NaN" "$TRAIN_LOG" | tail -3)
      [ -n "$ERR" ] && { echo "  ⚠ 에러/경고 흔적:"; echo "$ERR" | sed 's/^/    /'; }
    else
      echo "  (로그 파일 없음: $TRAIN_LOG)"
    fi
  fi
  echo "  체크포인트: $(ls -d "$TRAIN_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | xargs -n1 basename 2>/dev/null | tr '\n' ' ')$(ls -d "$TRAIN_DIR"/adapter_final 2>/dev/null >/dev/null && echo 'adapter_final')"
  echo ""

  echo "── ② 평가: checkpoint-100 → test ${NTEST}건 (TTA3) → $EVAL_DIR ──"
  if [ -f "$EVAL_DIR/report.json" ]; then
    bar "$NTEST" "$NTEST"; echo "  ✅ 완료 (report.json 생성됨)"
  elif pgrep -f "snuai.infer.predict.*checkpoint-100" > /dev/null; then
    ps aux | grep "snuai.infer.predict" | grep -v grep | awk '{printf "  실행 중  PID %s  CPU %s%%  MEM %s%%  경과 %s\n", $2, $3, $4, $10}'
    if [ -f "$EVAL_DIR/progress.jsonl" ]; then
      P=$(wc -l < "$EVAL_DIR/progress.jsonl")
      printf "  진행 "; bar "$P" "$NTEST"; echo
    fi
    LAST_BAR=$(grep -oE "predict:[^%]*%\|[^]]*\]" "$EVAL_LOG" 2>/dev/null | tail -1)
    [ -n "$LAST_BAR" ] && echo "  속도/ETA: $LAST_BAR"
  else
    echo "  ⏸ 대기 또는 프로세스 없음"
    [ -f "$EVAL_DIR/progress.jsonl" ] && { P=$(wc -l < "$EVAL_DIR/progress.jsonl"); printf "  마지막 진행 "; bar "$P" "$NTEST"; echo "  (재기동 필요할 수 있음)"; }
  fi
  echo ""

  echo "── GPU ──"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits | awk -F', ' '{printf "  사용률 %s%%   VRAM %s/%s MiB   온도 %s°C\n", $1,$2,$3,$4}'

  echo ""
  echo "======================================================================"
  sleep 5
done
