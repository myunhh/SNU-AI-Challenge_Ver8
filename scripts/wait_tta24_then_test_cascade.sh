#!/bin/bash
# TTA24 완료(또는 비정상 종료) 대기 후 캐스케이드 test-probe 자동 시작.
set -u
cd "$(dirname "$0")/.."
while true; do
  if [ -f runs/test_champ_tta24/report.json ]; then
    echo "[chain] TTA24 완료 감지 $(date '+%F %T') -- 캐스케이드 test-probe 시작"
    bash scripts/run_test_cascade_probe.sh
    exit $?
  fi
  if ! pgrep -f "snuai.infer.predict.*--tta 24" >/dev/null; then
    echo "[chain] TTA24 프로세스 종료(report.json 없음, 비정상 종료 가능) -- probe 보류, 확인 필요"
    exit 1
  fi
  sleep 60
done
