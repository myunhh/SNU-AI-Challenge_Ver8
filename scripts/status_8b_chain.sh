#!/bin/bash
# 통합 상태보드 — 현재 활성 작업만 표시 (완료된 8B 파이프라인·Ver12 체인은 07-20 정리).
#   현재: 챔피언(32B DPO ckpt600) TTA 스윕 (scripts/run_champ_tta_sweep.sh)
#         + Ver11 Ver8-warmstart SFT ckpt-600 로컬 평가 (07-20 추가, 외부 대시보드 대체)
# 완료된 추론은 ../grade/grade.py(의사 정답키)로 채점해 grade.txt에 캐시.
# ⚠️ 챔피언 근방 변형의 grade.py 점수는 자기참조 편향 있음 — 상대 비교 참고용.
# 사용: scripts/status_8b_chain.sh               # 한 번 출력
#       watch -n 30 --color scripts/status_8b_chain.sh
cd "$(dirname "$0")/.."
LOG=runs/champ_tta_sweep.log
PY=$HOME/anaconda3/envs/py3_11/bin/python
NTEST=$(( $(awk 'END{print NR}' data/test.csv) - 1 ))
V11=../Ver11
V11_CKPT=$V11/runs/ver11-checkpoint-600
V11_OUT=$V11/runs/test_ver11_ckpt600

rep() { local n=$1 c=$2; [ "$n" -gt 0 ] && printf "$c%.0s" $(seq 1 "$n"); return 0; }
bar() { # cur total label
  local cur=$1 total=$2 label=$3 w=26
  [ "$cur" -gt "$total" ] && cur=$total
  local f=$(( cur * w / total ))
  printf "  %-15s [%s%s] %4d/%-4d %3d%%" \
    "$label" "$(rep "$f" █)" "$(rep $((w - f)) ░)" "$cur" "$total" $(( cur * 100 / total ))
}
tqdm_line() { tr '\r' '\n' < "$LOG" 2>/dev/null | grep -aE "\| *[0-9]+/$1 \[" | tail -1 | sed 's/^ *//'; }

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
    local T; T=$(tqdm_line "$NTEST"); [ -n "$T" ] && echo "       └ $T"
  else
    bar 0 "$NTEST" "$LABEL"; echo "  ⏸ 대기"
  fi
}

echo "══════ 챔피언(32B DPO ckpt600, LB 0.91099) TTA 스윕  $(date '+%F %T') ══════"
echo
echo "── TTA 스윕 (test ${NTEST}건 · score24 · 32B-4bit 서빙 · TTA4=Klein 균형세트) ──"
if pgrep -f "bash scripts/run_champ_tta_swee[p].sh" >/dev/null; then :
elif grep -qa "전체 완료" "$LOG" 2>/dev/null; then echo "  ✅ 스윕 체인 전체 완료"
else echo "  ❌ 스윕 프로세스 없음 — 재발사: setsid nohup bash scripts/run_champ_tta_sweep.sh >> runs/champ_tta_sweep.log 2>&1 &"
fi
infer_row "TTA3 (챔피언)" "runs/test_dpo_ckpt600"
for T in 4 5 6 7 8; do
  infer_row "TTA$T" "runs/test_champ_tta$T"
done
echo
echo "── Ver11 (Ver8 warm-start SFT, ckpt-600/1500 = 40%) ──"
if [ -f "$V11_CKPT/adapter/adapter_model.safetensors" ]; then
  echo "  체크포인트 도착  ✅  $V11_CKPT (adapter+head, train_log 미동봉)"
else
  echo "  체크포인트 도착  ⏸ 대기  ($V11_CKPT 없음)"
fi
infer_row "test 추론(819)" "$V11_OUT"
echo "  안전 하한 LB 0.90226(Ver4 32B ckpt1600+TTA3) · 현 챔피언 LB 0.91623(Ver8 DPO ckpt600 TTA4) — 비교 참고용"
echo
echo "── GPU ──"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader | sed 's/^/  /'
echo
echo "(자동 갱신: watch -n 30 --color scripts/status_8b_chain.sh · 지난 결과: 8B스윕 runs/test_8b_bf16_*/grade.txt · Ver12 ../Ver12/runs/rerank_fit/fit.json·soup_gate.json)"
