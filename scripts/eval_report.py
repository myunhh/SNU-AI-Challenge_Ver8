#!/usr/bin/env python
"""§18 포렌식 홀드아웃 리포트 — predict.py의 progress.jsonl에서 단일 EM 숫자를
넘어서는 진단 표를 만든다 (원 구현계획 §18). Ver2 부검 때 손으로 했던 분석
(항등/비항등 분리 EM·항등 오탐·역순열 응답률·Kendall-tau 히스토그램·직전 Ver
diff)과 정확히 같은 지표를 재사용 가능한 도구로 코드화했다.

perm.kendall_tau_distance/perm.inverse(이미 존재)와 cascade.margin_accuracy_table
(이미 존재)을 그대로 재사용 — 새 지표 계산 로직만 추가.

사용:
  python scripts/eval_report.py --progress runs/cal_v7_xxx/progress.jsonl \
      --csv data/train.csv --image-dir data/train --holdout-val --val-frac 0.1 \
      --prev ../Ver3/runs/cal_v3_tta3/progress.jsonl \
      --out runs/cal_v7_xxx/forensic.md

progress.jsonl에 predict.py가 기록하는 "truth" 필드(신규 추가분)가 있으면 그걸
쓰고, 없으면(과거 런) --csv+--holdout-val로 동일 split을 재구성해 id로 매칭한다
— 이 소급 적용 경로 덕분에 Ver1~3의 기존 progress.jsonl에도 바로 쓸 수 있다.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


# ---------------------------------------------------------------------------
# 순수 함수 (pytest 가능 — records는 이미 파싱된 dict 리스트)
# ---------------------------------------------------------------------------

def compute_forensic_stats(records: list[dict]) -> dict:
    """records: [{"id":..., "truth": rank tuple, "pred": rank tuple, "margin": float}, ...]"""
    from snuai import perm

    n = len(records)
    if n == 0:
        raise ValueError("records가 비어있음")
    correct = [r["truth"] == r["pred"] for r in records]
    margins = [r["margin"] for r in records]

    identity_idx = [i for i, r in enumerate(records) if r["truth"] == perm.IDENTITY]
    nonidentity_idx = [i for i, r in enumerate(records) if r["truth"] != perm.IDENTITY]

    def _em(idx: list[int]) -> float | None:
        return (sum(correct[i] for i in idx) / len(idx)) if idx else None

    identity_false_positives = sum(
        1 for r in records if r["pred"] == perm.IDENTITY and r["truth"] != perm.IDENTITY)

    noninvolution = [r for r in records if perm.inverse(r["truth"]) != r["truth"]]
    reversals = sum(1 for r in noninvolution if r["pred"] == perm.inverse(r["truth"]))

    kendall_hist = {k: 0 for k in range(perm.N * (perm.N - 1) // 2 + 1)}
    for r in records:
        kendall_hist[perm.kendall_tau_distance(r["pred"], r["truth"])] += 1

    return {
        "n": n,
        "em": sum(correct) / n,
        "position_score": statistics.mean(
            perm.position_score(r["pred"], r["truth"]) for r in records),
        "pairwise_score": statistics.mean(
            perm.pairwise_score(r["pred"], r["truth"]) for r in records),
        "margin_mean": statistics.mean(margins),
        "margin_median": statistics.median(margins),
        "identity_n": len(identity_idx), "identity_em": _em(identity_idx),
        "nonidentity_n": len(nonidentity_idx), "nonidentity_em": _em(nonidentity_idx),
        "identity_false_positives": identity_false_positives,
        "reversal_n": len(noninvolution),
        "reversal_rate": (reversals / len(noninvolution)) if noninvolution else None,
        "kendall_tau_histogram": kendall_hist,
    }


def compute_diff_stats(curr: list[dict], prev: list[dict]) -> dict:
    """직전 버전 progress 대비 일치율 + 일치/불일치 구간별 EM."""
    curr_by_id = {r["id"]: r for r in curr}
    prev_by_id = {r["id"]: r for r in prev}
    common = sorted(set(curr_by_id) & set(prev_by_id))
    if not common:
        return {"common_n": 0}

    def _em(ids: list[str], by_id: dict[str, dict]) -> float | None:
        if not ids:
            return None
        return sum(1 for i in ids if by_id[i]["pred"] == by_id[i]["truth"]) / len(ids)

    agree = [i for i in common if curr_by_id[i]["pred"] == prev_by_id[i]["pred"]]
    disagree = [i for i in common if curr_by_id[i]["pred"] != prev_by_id[i]["pred"]]
    return {
        "common_n": len(common), "agree_n": len(agree), "disagree_n": len(disagree),
        "agree_frac": len(agree) / len(common),
        "curr_em_on_agree": _em(agree, curr_by_id), "prev_em_on_agree": _em(agree, prev_by_id),
        "curr_em_on_disagree": _em(disagree, curr_by_id),
        "prev_em_on_disagree": _em(disagree, prev_by_id),
    }


def render_markdown(stats: dict, margin_table: list[dict],
                    diff_stats: dict | None = None) -> str:
    lines = ["# 포렌식 홀드아웃 리포트", "", f"- n = {stats['n']}", f"- EM = {stats['em']:.4f}",
            f"- 부분점수: 쌍순서(1−KT/6) = {stats['pairwise_score']:.4f} (LB 프록시) / "
            f"위치일치 = {stats['position_score']:.4f}",
            f"- margin 평균/중앙값 = {stats['margin_mean']:.4f} / {stats['margin_median']:.4f}"]
    if stats["identity_n"] and stats["nonidentity_n"]:
        lines.append(f"- 항등 EM = {stats['identity_em']:.4f} (n={stats['identity_n']}) / "
                     f"비항등 EM = {stats['nonidentity_em']:.4f} (n={stats['nonidentity_n']})")
    lines.append(f"- 항등 오탐 수 = {stats['identity_false_positives']}")
    if stats["reversal_rate"] is not None:
        lines.append(f"- 역순열 응답률 = {stats['reversal_rate']:.4f} (n={stats['reversal_n']}, "
                     "랜덤 기준선 ≈ 4.2%보다 낮아야 순열 규약 정상)")
    lines += ["", "## margin 구간별 정확도 (십분위)", "", "| margin 범위 | n | 정확도 |",
             "|---|---|---|"]
    for row in margin_table:
        lines.append(f"| [{row['margin_lo']:.3f}, {row['margin_hi']:.3f}] | "
                     f"{row['n']} | {row['accuracy']:.3f} |")
    lines += ["", "## Kendall-tau 거리 히스토그램", "", "| KT 거리 | n |", "|---|---|"]
    for k, cnt in sorted(stats["kendall_tau_histogram"].items()):
        lines.append(f"| {k} | {cnt} |")
    if diff_stats and diff_stats.get("common_n"):
        lines += ["", "## 이전 버전 대비 diff", "",
                 f"- 공통 {diff_stats['common_n']}건 중 일치 {diff_stats['agree_n']}건 "
                 f"({diff_stats['agree_frac']:.1%})",
                 f"- 일치 구간 EM: 현재 {diff_stats['curr_em_on_agree']:.4f} / "
                 f"이전 {diff_stats['prev_em_on_agree']:.4f}",
                 f"- 불일치 구간({diff_stats['disagree_n']}건) EM: 현재 "
                 f"{diff_stats['curr_em_on_disagree']:.4f} / "
                 f"이전 {diff_stats['prev_em_on_disagree']:.4f}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 실제 progress.jsonl/CSV 연동 (main()에서만 호출)
# ---------------------------------------------------------------------------

def _truth_from_csv(csv: str, image_dir: str, caption_col: str,
                    holdout_val: bool, val_frac: float) -> dict[str, tuple]:
    from snuai.data.sample import load_csv
    samples = load_csv(csv, image_dir, caption_col=caption_col)
    if holdout_val:
        from snuai.data.split import split_samples
        _, samples = split_samples(samples, val_frac=val_frac)
    return {s.id: s.rank for s in samples if s.rank is not None}


def load_records(progress_path: str, csv: str | None = None, image_dir: str | None = None,
                 caption_col: str = "Caption", holdout_val: bool = False,
                 val_frac: float = 0.1) -> list[dict]:
    from snuai import submission

    raw = []
    with open(progress_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    if not raw:
        raise SystemExit(f"{progress_path}: 빈 파일")

    truth_by_id: dict[str, tuple] = {}
    # error(fallback) 레코드는 애초에 truth가 안 붙는 게 정상(run_predict가 try 블록
    # 안에서만 truth를 기록) — "전부에 truth가 있는가" 판정에서는 제외해야 한다.
    non_error = [r for r in raw if "error" not in r]
    if non_error and not all("truth" in r for r in non_error):
        if not csv:
            raise SystemExit(f"{progress_path}에 truth 필드가 없는 레코드가 있음 — "
                            "--csv(+--holdout-val)로 정답을 재구성할 것")
        truth_by_id = _truth_from_csv(csv, image_dir, caption_col, holdout_val, val_frac)

    records = []
    for r in raw:
        if "error" in r:
            continue  # 파이프라인 실패로 항등 fallback된 샘플 — 포렌식 통계 대상 아님
        truth = submission.parse_answer(r["truth"]) if "truth" in r else truth_by_id.get(r["id"])
        if truth is None:
            continue  # 라벨 없는 샘플(test) — 스킵
        records.append({"id": r["id"], "truth": truth,
                        "pred": submission.parse_answer(r["answer"]),
                        "margin": r.get("margin", 0.0)})
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    d = ap.add_argument
    d("--progress", required=True, help="predict.py progress.jsonl 경로")
    d("--csv", default=None); d("--image-dir", default=None)
    d("--caption-col", default="Caption")
    d("--holdout-val", action="store_true"); d("--val-frac", type=float, default=0.1)
    d("--prev", default=None, help="직전 버전 progress.jsonl — diff 비교")
    d("--out", required=True)
    args = ap.parse_args(argv)

    from snuai.infer.cascade import margin_accuracy_table

    records = load_records(args.progress, args.csv, args.image_dir, args.caption_col,
                           args.holdout_val, args.val_frac)
    stats = compute_forensic_stats(records)
    correct = [r["truth"] == r["pred"] for r in records]
    margin_table = margin_accuracy_table([r["margin"] for r in records], correct)

    diff_stats = None
    if args.prev:
        prev_records = load_records(args.prev, args.csv, args.image_dir, args.caption_col,
                                    args.holdout_val, args.val_frac)
        diff_stats = compute_diff_stats(records, prev_records)

    md = render_markdown(stats, margin_table, diff_stats)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"[eval_report] 저장 → {out_path}")


if __name__ == "__main__":
    main()
