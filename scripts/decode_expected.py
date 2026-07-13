#!/usr/bin/env python
"""metric-optimal 디코딩 (TODO_VER8 P1) — scores24 덤프에서 오프라인 재디코딩.

LB 채점이 쌍순서(1−KT/6) 부분점수로 판명(runs/drift_forensics_v7.md)됐으므로,
24-way 분포의 argmax(최빈 순열)가 아니라 **기대 쌍순서 점수를 최대화하는 순열**
을 출력하는 게 채점 기준 최적이다:

  r* = argmax_r Σ_q softmax(scores24)[q] · pairwise_score(r, q)

핵심 성질: 분포가 뾰족하면(고마진) r* = argmax와 일치, 퍼져 있으면(저마진)
후보들 사이의 "중앙값 순열" 쪽으로 이동한다. 추론 재실행 없이 progress.jsonl의
scores24 필드(predict.py가 덤프)만으로 계산 가능 — 제출 경로(predict.py의
서브미션 하드닝)는 건드리지 않고 사후 처리로 분리해 A/B를 깨끗하게 유지한다.

사용:
  # 홀드아웃 A/B (라벨 필요 — truth 인라인 또는 --csv 소급)
  python scripts/decode_expected.py --progress runs/xxx/progress.jsonl --eval

  # test 제출 파일 생성 (채택 후)
  python scripts/decode_expected.py --progress runs/test_xxx/progress.jsonl \
      --submission runs/test_xxx/submission_expected.csv
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # eval_report 재사용용

from snuai import perm, submission
from snuai.infer.cascade import softmax


def score_matrix(objective: str) -> np.ndarray:
    """24×24 부분점수 행렬 M[r, q] = objective_score(PERMS24[r], PERMS24[q])."""
    fn = {"pairwise": perm.pairwise_score, "position": perm.position_score}[objective]
    return np.array([[fn(r, q) for q in perm.PERMS24] for r in perm.PERMS24])


def decode_expected(scores24, mat: np.ndarray) -> perm.Perm:
    """기대 부분점수 최대화 순열. 동률이면 확률 높은 쪽(argmax 방향) 우선."""
    p = softmax(np.asarray(scores24, dtype=float))
    exp = mat @ p
    best = np.flatnonzero(exp >= exp.max() - 1e-12)
    if len(best) > 1:
        best = [max(best, key=lambda i: p[i])]
    return perm.PERMS24[int(best[0])]


def load_progress(path: str) -> list[dict]:
    recs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "error" in r:
            continue
        if "scores24" not in r:
            raise SystemExit(f"{path}: scores24 필드 없음 — 신규 predict.py로 재실행 필요")
        recs.append(r)
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    d = ap.add_argument
    d("--progress", required=True)
    d("--objective", default="pairwise", choices=["pairwise", "position"])
    d("--eval", action="store_true", help="argmax vs expected 디코딩 A/B (라벨 필요)")
    d("--submission", default=None, help="expected 디코딩으로 제출 CSV 생성")
    d("--csv", default=None); d("--image-dir", default=None)
    d("--caption-col", default="Caption")
    d("--holdout-val", action="store_true"); d("--val-frac", type=float, default=0.1)
    args = ap.parse_args(argv)

    recs = load_progress(args.progress)
    mat = score_matrix(args.objective)
    decoded = {r["id"]: decode_expected(r["scores24"], mat) for r in recs}
    changed = sum(1 for r in recs
                  if decoded[r["id"]] != submission.parse_answer(r["answer"]))
    print(f"[decode] n={len(recs)} objective={args.objective} "
          f"argmax와 다른 예측 {changed}건 ({changed/len(recs):.1%})")

    if args.eval:
        truth_by_id = {r["id"]: submission.parse_answer(r["truth"])
                       for r in recs if "truth" in r}
        if len(truth_by_id) < len(recs) and args.csv:
            from eval_report import _truth_from_csv
            truth_by_id = _truth_from_csv(args.csv, args.image_dir, args.caption_col,
                                          args.holdout_val, args.val_frac)
        labeled = [r for r in recs if r["id"] in truth_by_id]
        if not labeled:
            raise SystemExit("--eval인데 truth를 구할 수 없음 (--csv --holdout-val 필요)")
        obj_fn = {"pairwise": perm.pairwise_score,
                  "position": perm.position_score}[args.objective]

        def summary(pred_of):
            em = statistics.mean(pred_of(r) == truth_by_id[r["id"]] for r in labeled)
            ob = statistics.mean(obj_fn(pred_of(r), truth_by_id[r["id"]]) for r in labeled)
            return em, ob

        em_a, ob_a = summary(lambda r: submission.parse_answer(r["answer"]))
        em_b, ob_b = summary(lambda r: decoded[r["id"]])
        # paired bootstrap (ab_gate와 동일 규약)
        a = np.array([obj_fn(submission.parse_answer(r["answer"]), truth_by_id[r["id"]])
                      for r in labeled])
        b = np.array([obj_fn(decoded[r["id"]], truth_by_id[r["id"]]) for r in labeled])
        rng = np.random.default_rng(0)
        n = len(a)
        diffs = np.array([(b[idx].mean() - a[idx].mean())
                          for idx in rng.integers(0, n, (10_000, n))])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"[A argmax  ] EM={em_a:.4f} {args.objective}={ob_a:.4f}")
        print(f"[B expected] EM={em_b:.4f} {args.objective}={ob_b:.4f}")
        print(f"[gate] Δ{args.objective}={ob_b-ob_a:+.4f} CI95=[{lo:+.4f}, {hi:+.4f}] "
              f"(min_delta 기준은 ab_gate와 동일 +0.02)")

    if args.submission:
        rows = [(r["id"], decoded[r["id"]]) for r in recs]
        path = submission.write_submission(Path(args.submission), rows, spaced=True)
        submission.validate_submission(path, expected_ids=[r["id"] for r in recs])
        print(f"[submission] {path}")


if __name__ == "__main__":
    main()
