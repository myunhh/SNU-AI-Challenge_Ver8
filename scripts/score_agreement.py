#!/usr/bin/env python
"""챔피언 일치율 프록시 채점기 — test 제출물끼리의 일치율로 체크포인트 후보를 압축.

test 819건은 정답 라벨이 없어 로컬 채점이 불가능하다. 대신 **현 챔피언 제출물**
(LB 0.91099 실측)과의 일치율을 프록시로 쓴다(2026-07-19 사용자 결정: 프록시로
후보 압축 + 최종 확정은 LB). 라벨은 일절 쓰지 않고 자기 모델 출력끼리만 비교한다.

⚠️ 한계: 이 프록시는 "챔피언과 비슷한 정도"를 재는 것이라, 챔피언이 틀리는 문제를
   맞히는(=진짜 개선인) 후보일수록 일치율이 깎인다. 후보 간 상대 순위 압축용으로만
   쓰고 최종 판정은 반드시 LB로 할 것. LB 채점은 EM 확정(부분점수 없음, 07-18)이라
   EM-일치율을 1순위, 쌍순서-일치율을 참고 지표로 둔다.

사용:
  python scripts/score_agreement.py \
      --champion runs/test_dpo_ckpt600/submission.csv \
      --candidates "runs/test_8b_bf16_sft_ckpt*/submission.csv" \
      --out runs/sft_ckpt_selection.json [--print-best]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

from snuai import perm, submission


def load_sub(path: str) -> dict[str, perm.Perm]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["Id"]: submission.parse_answer(row["Answer"])
                for row in csv.DictReader(f)}


def agreement(champ: dict, cand: dict) -> dict:
    ids = sorted(champ.keys() & cand.keys())
    if len(ids) != len(champ) or len(ids) != len(cand):
        raise SystemExit(f"ID 불일치: champion {len(champ)} vs candidate {len(cand)} (교집합 {len(ids)})")
    em = sum(champ[i] == cand[i] for i in ids) / len(ids)
    pair = sum(1 - perm.kendall_tau_distance(champ[i], cand[i]) / 6 for i in ids) / len(ids)
    return {"n": len(ids), "em_agree": round(em, 5), "pair_agree": round(pair, 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", required=True)
    ap.add_argument("--candidates", required=True, nargs="+",
                    help="submission.csv 경로 또는 glob 패턴 (여러 개 가능)")
    ap.add_argument("--out", required=True, help="채점 결과 JSON (재현성 검증 대비 전량 기록)")
    ap.add_argument("--print-best", action="store_true",
                    help="최고 EM-일치율 후보의 submission.csv 경로만 stdout 마지막 줄에 출력")
    args = ap.parse_args()

    champ = load_sub(args.champion)
    paths: list[str] = []
    for pat in args.candidates:
        hits = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        paths.extend(hits)
    if not paths:
        raise SystemExit(f"후보 없음: {args.candidates}")

    results = {}
    for p in paths:
        results[p] = agreement(champ, load_sub(p))
        print(f"[agree] {p}: EM {results[p]['em_agree']:.4f} / 쌍순서 {results[p]['pair_agree']:.4f}")

    # EM-일치율 1순위, 동률이면 쌍순서-일치율
    best = max(results, key=lambda p: (results[p]["em_agree"], results[p]["pair_agree"]))
    payload = {"champion": args.champion, "results": results, "best": best,
               "note": "champion-agreement proxy — 후보 압축용, 최종 판정은 LB"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[agree] best: {best} → {args.out}")
    if args.print_best:
        print(best)


if __name__ == "__main__":
    main()
