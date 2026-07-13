#!/usr/bin/env python
"""A/B 채택 게이트 — 같은 홀드아웃에서 나온 두 progress.jsonl을 paired bootstrap 비교.

채택 규칙 (구현 계획 §12): Δ(B-A) ≥ +2%p **그리고** 95% CI 하한 > 0 일 때만 채택.
Ver2의 단순 'b>a' 게이트가 +0.4%p짜리 노이즈를 채택했던 것의 재발 방지.

지표(--metric): em(기본) 외에 pairwise(쌍순서=1−KT/6)·position(위치일치) 지원
— Ver7에서 홀드아웃 EM 동률인데 LB +3.49pp가 실측되면서(LB는 부분점수 채점 추정,
VER7.md) EM 단독 게이트가 LB 이득을 놓칠 수 있음이 확인됨 (TODO_VER8 P0).
pairwise/position은 truth가 필요 — 과거 런(Ver1~3)은 --csv+--holdout-val로 소급.

사용:
  python scripts/ab_gate.py runs/cal_A runs/cal_B --name cascade --out runs/adoption.json
  python scripts/ab_gate.py runs/cal_A runs/cal_B --metric pairwise \
      --csv data/train.csv --image-dir data/train --holdout-val
(run 인자는 progress.jsonl이 들어있는 디렉터리 또는 파일 경로)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_correct(path: str | Path) -> dict[str, bool]:
    p = Path(path)
    if p.is_dir():
        p = p / "progress.jsonl"
    out: dict[str, bool] = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "correct" not in r:
                raise SystemExit(f"{p}: 'correct' 필드 없음 — 홀드아웃(--eval) 런이어야 함")
            out[r["id"]] = bool(r["correct"])
    return out


def load_scores(path: str | Path, metric: str, args) -> dict[str, float]:
    """샘플별 지표값 로드. em은 'correct' 필드 그대로(하위호환), 나머지는 truth 기반."""
    if metric == "em":
        return {k: float(v) for k, v in load_correct(path).items()}
    from eval_report import load_records  # 같은 scripts/ 디렉터리 — truth 소급 경로 재사용
    from snuai import perm
    fn = {"pairwise": perm.pairwise_score, "position": perm.position_score}[metric]
    p = Path(path)
    if p.is_dir():
        p = p / "progress.jsonl"
    recs = load_records(str(p), args.csv, args.image_dir, args.caption_col,
                        args.holdout_val, args.val_frac)
    return {r["id"]: fn(r["pred"], r["truth"]) for r in recs}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, iters: int = 10_000,
                     seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(iters)
    for t in range(iters):
        idx = rng.integers(0, n, n)
        diffs[t] = b[idx].mean() - a[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta": float(b.mean() - a.mean()), "ci95": [float(lo), float(hi)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", help="기준(A) progress.jsonl 경로/디렉터리")
    ap.add_argument("run_b", help="후보(B) progress.jsonl 경로/디렉터리")
    ap.add_argument("--name", default="candidate", help="adoption.json에 기록할 후보 이름")
    ap.add_argument("--min-delta", type=float, default=0.02)
    ap.add_argument("--iters", type=int, default=10_000)
    ap.add_argument("--out", default=None, help="결정 JSON 저장 경로 (생략 시 stdout만)")
    ap.add_argument("--metric", default="em", choices=["em", "pairwise", "position"],
                    help="비교 지표 — pairwise가 LB 프록시 (VER7.md)")
    ap.add_argument("--csv", default=None, help="pairwise/position에서 과거 런 truth 소급용")
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--caption-col", default="Caption")
    ap.add_argument("--holdout-val", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    ca = load_scores(args.run_a, args.metric, args)
    cb = load_scores(args.run_b, args.metric, args)
    common = sorted(ca.keys() & cb.keys())
    if len(common) != len(ca) or len(common) != len(cb):
        print(f"⚠️ id 집합 불일치: A={len(ca)} B={len(cb)} 공통={len(common)} — 공통만 비교")
    if not common:
        raise SystemExit("공통 샘플이 없음 — 같은 홀드아웃 런인지 확인")
    a = np.array([ca[i] for i in common], dtype=float)
    b = np.array([cb[i] for i in common], dtype=float)

    boot = paired_bootstrap(a, b, iters=args.iters)
    adopt = boot["delta"] >= args.min_delta and boot["ci95"][0] > 0
    result = {
        "name": args.name, "n": len(common), "metric": args.metric,
        "acc_a": float(a.mean()), "acc_b": float(b.mean()),
        **boot, "min_delta": args.min_delta, "adopt": adopt,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n[gate] Δ{args.metric}={boot['delta']:+.4f} CI95=[{boot['ci95'][0]:+.4f}, "
          f"{boot['ci95'][1]:+.4f}] → {'✅ 채택' if adopt else '❌ 기각'}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


if __name__ == "__main__":
    main()
