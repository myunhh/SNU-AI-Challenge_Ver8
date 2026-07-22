#!/bin/bash
# TTA12(A4) 워치독 — 죽으면 자동 재기동, VRAM 23GB(23552MiB) 초과 전이를 기록.
#   run_champ_tta12_alt.sh는 progress.jsonl 기반 resume을 지원하므로 재기동은 안전.
# 종료 조건: runs/test_champ_tta12_alt/report.json 생성(정상 완료) 시 스스로 종료.
# 로그: runs/vram_watch.log (VRAM 전이 + 재기동 이벤트만, 스팸 방지로 상태 유지 시 미기록)
set -u
cd "$(dirname "$0")/.."
OUT=runs/test_champ_tta12_alt
RUNLOG=runs/champ_tta12_alt.log
WATCHLOG=runs/vram_watch.log
THRESHOLD_MIB=23552   # 23GiB

echo "[watch] 워치독 시작 $(date '+%F %T')" >> "$WATCHLOG"
was_over=0
while true; do
  if [ -f "$OUT/report.json" ]; then
    echo "[watch] 완료 감지 $(date '+%F %T') -- 워치독 종료" >> "$WATCHLOG"
    exit 0
  fi

  if ! pgrep -f "snuai.infer.predict.*--tta 12" >/dev/null; then
    echo "[watch] 프로세스 부재 감지 $(date '+%F %T') -- 재기동" >> "$WATCHLOG"
    setsid nohup bash scripts/run_champ_tta12_alt.sh >> "$RUNLOG" 2>&1 &
    disown
    sleep 30
  fi

  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "${MEM:-}" ]; then
    if [ "$MEM" -gt "$THRESHOLD_MIB" ]; then
      if [ "$was_over" -eq 0 ]; then
        echo "[watch] VRAM 23GB 초과: ${MEM}MiB $(date '+%F %T')" >> "$WATCHLOG"
        was_over=1
      fi
    else
      if [ "$was_over" -eq 1 ]; then
        echo "[watch] VRAM 23GB 이하로 복귀: ${MEM}MiB $(date '+%F %T')" >> "$WATCHLOG"
      fi
      was_over=0
    fi
  fi

  sleep 30
done
