#!/bin/bash
# 통합 상태보드 — 현재 활성 작업만 표시.
#   현재 활성: 교대군 A4(ALTERNATING12=짝순열 12개) test 819건 추론
#             (scripts/run_champ_tta12_alt.sh) — TTA24가 TTA8-balanced에 LB 패배한 뒤
#             "홀순열이 문제인가" 가설 검증용 별개 가지(V→D4→S4 계열과 무관)
#   완료: 완전균형 24뷰(BALANCED24=S4 전체) test 819건 (scripts/run_champ_tta24.sh)
#   완료: 균형 8뷰(BALANCED8=Sylow-2) test 819건 (scripts/run_champ_tta8b.sh)
# 완료되면 ../grade/grade.py(의사 정답키)로 채점해 grade.txt에 캐시.
# ⚠️ 챔피언 근방 변형의 grade.py 점수는 자기참조 편향 있음 — 상대 비교 참고용.
# 사용: scripts/status_8b_chain.sh               # 한 번 출력
#       watch -n 30 --color scripts/status_8b_chain.sh
cd "$(dirname "$0")/.."
LOG8B=runs/champ_tta8b.log
LOG24=runs/champ_tta24.log
LOG12=runs/champ_tta12_alt.log
PY=python3
NTEST=$(( $(awk 'END{print NR}' data/test.csv) - 1 ))

rep() { local n=$1 c=$2; [ "$n" -gt 0 ] && printf "$c%.0s" $(seq 1 "$n"); return 0; }
bar() { # cur total label
  local cur=$1 total=$2 label=$3 w=26
  [ "$cur" -gt "$total" ] && cur=$total
  local f=$(( cur * w / total ))
  printf "  %-15s [%s%s] %4d/%-4d %3d%%" \
    "$label" "$(rep "$f" █)" "$(rep $((w - f)) ░)" "$cur" "$total" $(( cur * 100 / total ))
}

GRADEPY=../grade/grade.py
grade_line() { # run_dir → "EM 0.xxxx · LB≈0.xxxx" (전체 출력은 run_dir/grade.txt에 캐시)
  local DIR=$1 SUB=$1/submission.csv CACHE=$1/grade.txt
  [ -f "$SUB" ] && [ -f "$GRADEPY" ] || return 0
  if [ ! -s "$CACHE" ] || [ "$SUB" -nt "$CACHE" ]; then
    "$PY" "$GRADEPY" "$SUB" --show-wrong 0 > "$CACHE" 2>&1 || { rm -f "$CACHE"; return 0; }
  fi
  awk '/exact-match accuracy/{em=$NF}
       /estimated leaderboard/{lb=$(NF-2)}
       /self-referential/{sr=1}
       END{if(em!=""){printf "EM %s", em;
                      if(lb!="")printf " · LB≈%s", lb;
                      else if(sr)printf " (자기참조·상대비교용)"}}' "$CACHE"
}

infer_row() { # label out_dir
  local LABEL=$1 DIR=$2
  if [ -f "$DIR/report.json" ]; then
    local G; G=$(grade_line "$DIR")
    bar "$NTEST" "$NTEST" "$LABEL"; echo "  ✅ 완료${G:+  $G}"
  elif [ -f "$DIR/progress.jsonl" ]; then
    local P; P=$(wc -l < "$DIR/progress.jsonl")
    bar "$P" "$NTEST" "$LABEL"; echo "  ◀ 진행 중"
  else
    bar 0 "$NTEST" "$LABEL"; echo "  ⏸ 대기"
  fi
}

echo "══════ TTA 균형 계열(V→Sylow-2→S4) + A4 별개 가지  $(date '+%F %T') ══════"
echo
echo "── 현재 활성: 교대군 A4(짝순열 12개, TTA24 LB패배 후 '홀순열이 문제인가' 검증) ──"
if pgrep -f "bash scripts/run_champ_tta12_alt.sh|snuai.infer.predict.*--tta 12" >/dev/null; then :
elif grep -qa "완료" "$LOG12" 2>/dev/null; then echo "  ✅ 완료"
else echo "  ❌ 프로세스 없음 — 재발사: setsid nohup bash scripts/run_champ_tta12_alt.sh >> $LOG12 2>&1 &"
fi
infer_row "TTA12(A4)" "runs/test_champ_tta12_alt"
echo
echo "── 완료: V→Sylow-2→S4 계열 ──"
infer_row "TTA24(S4 전체)" "runs/test_champ_tta24"
infer_row "TTA8-Balanced" "runs/test_champ_tta8b"
echo
echo "── 대기열: 캐스케이드(2단계 pairwise) test-probe, 819건 ──"
infer_row_cas() {
  local DIR=runs/test_champ_tta8b_cascade
  if [ -f "$DIR/report.json" ]; then
    bar "$NTEST" "$NTEST" "cascade"; echo "  ✅ 완료 — 상세 runs/test_cascade_chain.log"
  elif [ -f "$DIR/progress.jsonl" ]; then
    local P; P=$(wc -l < "$DIR/progress.jsonl")
    bar "$P" "$NTEST" "cascade"; echo "  ◀ 진행 중"
  else
    bar 0 "$NTEST" "cascade"; echo "  ⏸ 대기"
  fi
}
if [ -f runs/test_champ_tta8b_cascade/report.json ]; then
  infer_row_cas
elif pgrep -f "snuai.infer.predict.*--cascade" >/dev/null; then
  infer_row_cas
elif pgrep -f "wait_tta24_then_test_cascade" >/dev/null; then
  echo "  ⏸ TTA24 완료 대기 중 (자동 시작 예약됨)"
elif grep -qa "TTA24 프로세스 종료" runs/test_cascade_chain.log 2>/dev/null; then
  echo "  ⚠ TTA24가 비정상 종료돼 대기열이 보류됨 — 확인 필요"
else
  echo "  ❌ 대기열 프로세스 없음 — 재발사: setsid nohup bash scripts/wait_tta24_then_test_cascade.sh >> runs/test_cascade_chain.log 2>&1 &"
fi
echo
echo "── GPU ──"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader | sed 's/^/  /'
echo
echo "(자동 갱신: watch -n 30 --color scripts/status_8b_chain.sh)"
