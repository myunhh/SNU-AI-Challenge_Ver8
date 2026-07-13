#!/usr/bin/env python
"""§1 진단 제출 생성 — EM vs 쌍순서 채점 판별 (TODO §1, 오프라인·GPU 불필요).

규정 문서는 "예선 EM만(부분점수 없음)"인데 LB 실측 정합은 쌍순서(1−KT/6) 채점을
강하게 시사한다(Ver7 드리프트 포렌식). 재추론 없이 **기존 제출의 progress.jsonl**
에서 고마진(=모델이 가장 확신하는, 즉 정답일 가능성이 가장 높은) 샘플 K개만 골라
인접 스와프 1개를 적용한 뒤 그 델타로 판별한다:

  EM 채점이면    ΔLB ≈ −K/N · 100  (그 K건이 전부 정답→오답으로 뒤집힌다고 가정)
  쌍순서 채점이면 ΔLB ≈ −K/(6N) · 100  (스와프 1개당 쌍순서 손실은 정확히 1/6)

N=819(test 전체), K=60 기본이면 EM 가설 −7.3%p대, 쌍순서 가설 정확히 −1.22%p —
LB 재제출 후 실측 델타를 이 두 수치와 비교해 판별한다. Kaggle 제출 1슬롯을 쓰므로
실제 제출은 사용자가 별도로 확인 후 진행할 것(이 스크립트는 파일만 생성한다).

사용:
  python scripts/diag_swap_submission.py \
      --progress ../Ver7/runs/test_v7_ckpt1800_tta3/progress.jsonl \
      --k 60 --out runs/diag_swap60/submission.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snuai import perm, submission
from snuai.infer.cascade import margin_of


def _margin_of_record(r: dict) -> float:
    if "margin" in r:
        return float(r["margin"])
    if "scores24" in r:
        return margin_of(r["scores24"])
    raise SystemExit(f"{r.get('id')}: margin도 scores24도 없음 — predict.py progress.jsonl 확인")


def load_progress(path: str) -> list[dict]:
    recs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "error" in r or "answer" not in r:
            continue
        recs.append(r)
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    d = ap.add_argument
    d("--progress", required=True, help="현재 제출 구성의 progress.jsonl")
    d("--k", type=int, default=60, help="스와프를 적용할 고마진 샘플 수")
    d("--swap-index", type=int, default=0, choices=[0, 1, 2],
      help="adjacent_swap_ranks() 3종 중 어느 것을 적용할지 (기본 0=고정, 재현성)")
    d("--out", required=True, help="진단용 제출 CSV 경로")
    args = ap.parse_args(argv)

    recs = load_progress(args.progress)
    n = len(recs)
    if args.k <= 0 or args.k > n:
        raise SystemExit(f"--k={args.k}는 (0, {n}] 범위여야 함")

    for r in recs:
        r["_margin"] = _margin_of_record(r)
    recs.sort(key=lambda r: r["_margin"], reverse=True)
    swap_ids = {r["id"] for r in recs[: args.k]}

    rows = []
    for r in recs:
        rank = submission.parse_answer(r["answer"])
        if r["id"] in swap_ids:
            rank = perm.adjacent_swap_ranks(rank)[args.swap_index]
        rows.append((r["id"], rank))

    out_path = submission.write_submission(Path(args.out), rows, spaced=True)
    submission.validate_submission(out_path, expected_ids=[r["id"] for r in recs])

    em_delta = -100.0 * args.k / n
    pairwise_delta = -100.0 * args.k / (6 * n)
    print(f"[diag] n={n} k={args.k} (고마진 상위, min margin 적용대상={recs[args.k-1]['_margin']:.4f}) "
          f"swap_index={args.swap_index}")
    print(f"[diag] 제출 파일: {out_path}")
    print(f"[diag] 예측 LB 델타 — EM 채점 가설: {em_delta:+.2f}%p / 쌍순서 채점 가설: {pairwise_delta:+.2f}%p")
    print("[diag] 실제 LB 실측 후 위 두 값과 비교해 채점 방식 판별 (Ver7/LB_LOG.md에 기록할 것). "
          "⚠️ 이 스크립트는 파일만 만든다 — Kaggle 제출은 별도로 직접 확인 후 진행할 것"
          "(1일 2회 제한, 검증용 슬롯 소모).")


if __name__ == "__main__":
    main()
