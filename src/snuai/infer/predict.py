"""추론 오케스트레이터 — 스코어러 × TTA × 캐스케이드 × 예산 × 재개 × 제출.

사용 예 (3090):
  python -m snuai.infer.predict --csv test.csv --image-dir images/test \
      --strategy score24 --model-id /home/yhmin/model/hub/Qwen3-VL-8B-Instruct --adapter runs/sft \
      --four-bit --max-pixels 602112 --tta 3 --cascade --tau 0.15 --out runs/final

검증/캘리브레이션 (train 홀드아웃, 라벨 있음 → 정확도·margin 표·τ 제안 출력):
  python -m snuai.infer.predict --csv train.csv --image-dir images/train \
      --holdout-val --strategy likelihood --model-id ... --eval --out runs/val

GPU 없이 파이프라인 자체 검증 (합성 데이터 + 오라클 스코어러):
  python -m snuai.infer.predict --synthetic 60 --strategy dummy --noise 3.0 \
      --tta 3 --cascade --eval --out runs/dryrun

안전 설계:
  - 샘플 단위 try/except → 실패 시 항등순열 fallback (24h 런이 절대 죽지 않음)
  - progress.jsonl 재개(resume) — 중단 후 같은 명령 재실행이면 이어서 진행
  - BudgetGuard가 밀리면 TTA 뷰·캐스케이드 2단계를 자동 축소
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .. import perm, submission
from ..budget import BudgetGuard
from ..data.sample import Sample, load_csv
from ..data.split import split_samples
from .cascade import (CascadeConfig, margin_accuracy_table, margin_of, run_cascade,
                      tau_by_escalation_budget)
from .tta import TTAConfig, tta_scores


# ---------------------------------------------------------------------------
# 더미(오라클) 스코어러 — GPU 없이 파이프라인 무결성 검증
# ---------------------------------------------------------------------------

class OracleScorer:
    """합성 프레임의 픽셀에서 정답을 판독하는 스코어러 (+가우시안 노이즈).

    TTA가 이미지를 셔플해 넣어도 픽셀 기준으로 채점하므로, 최종 예측이
    원본 정답과 일치하는지가 곧 'TTA 리맵·캐스케이드·제출 경로' 전체의 검증이 된다.
    """

    def __init__(self, noise: float = 0.0, seed: int = 0):
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def _rank_from_pixels(self, images: list) -> perm.Perm:
        from ..data.synthetic import read_step_from_frame
        return tuple(read_step_from_frame(im) for im in images)

    def scores(self, caption: str, images: list) -> np.ndarray:
        true_rank = self._rank_from_pixels(images)
        out = np.full(24, -4.0) + self.rng.normal(0, self.noise, 24)
        out[perm.index_of(true_rank)] += 8.0
        return out

    def make_pairwise_fn(self, caption: str, images: list):
        true_rank = self._rank_from_pixels(images)

        def fn(i: int, j: int) -> float:
            return 0.97 if true_rank[i] < true_rank[j] else 0.03
        return fn


# ---------------------------------------------------------------------------
# 실행 설정
# ---------------------------------------------------------------------------

@dataclass
class PredictConfig:
    strategy: str = "score24"       # score24|likelihood|match|cot|dummy
    tta_views: int = 1
    tta_seed: int = 1234
    tta_agg: str = "mean"
    cascade: CascadeConfig = CascadeConfig(enable=False)
    budget_hours: float = 24.0
    safety: float = 0.85
    stage2_cost_est: float = 4.0    # pairwise 2회(플립) 추정 초 — 롤링 실측으로 갱신됨
    out_dir: str = "runs/predict"


def _load_progress(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["id"]] = rec
    return done


def run_predict(samples: list[Sample], scorer, cfg: PredictConfig,
                pairwise_factory=None, clock=time.monotonic) -> dict:
    """전체 추론 실행. 반환: 리포트 dict (accuracy는 라벨이 있을 때만)."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"
    done = _load_progress(progress_path)
    if done:
        print(f"[resume] 기존 진행 {len(done)}/{len(samples)}건 재사용")

    budget = BudgetGuard(cfg.budget_hours * 3600, len(samples), cfg.safety, clock=clock)
    budget.done = len(done)
    view_cost_est = None  # 뷰 1회 비용 롤링 추정

    with open(progress_path, "a", encoding="utf-8") as prog:
        for s in samples:
            if s.id in done:
                continue
            budget.start_sample()
            rec: dict = {"id": s.id}
            try:
                images = s.load_images()

                def scorer_fn(imgs, _s=s):
                    return scorer.scores(_s.caption, imgs)

                # TTA 뷰 수: 예산이 밀리면 1로 축소
                n_views = cfg.tta_views
                if view_cost_est is not None and n_views > 1:
                    if not budget.allow(view_cost_est * n_views):
                        n_views = 1
                        rec["tta_reduced"] = True
                t_v0 = clock()
                scores, _ = tta_scores(images, scorer_fn,
                                       TTAConfig(n_views=n_views, seed=cfg.tta_seed,
                                                 agg=cfg.tta_agg))
                dt_views = clock() - t_v0
                cost = dt_views / max(n_views, 1)
                view_cost_est = cost if view_cost_est is None else 0.8 * view_cost_est + 0.2 * cost

                # 캐스케이드 (2단계는 예산 허용 시에만)
                cas_cfg = cfg.cascade
                if cas_cfg.enable and not budget.allow(cfg.stage2_cost_est):
                    cas_cfg = replace(cas_cfg, enable=False)
                    rec["stage2_skipped_budget"] = True
                pairwise_fn = None
                if cas_cfg.enable and pairwise_factory is not None:
                    pairwise_fn = pairwise_factory(s.caption, images)
                res = run_cascade(scores, pairwise_fn, cas_cfg)

                rec.update({
                    "answer": submission.format_answer(res.rank),
                    "margin": round(res.margin, 6),
                    "escalated": res.escalated,
                    "queried_pairs": res.queried_pairs,
                    # TTA 집계 후 24-way 점수(PERMS24 인덱스 공간) — metric-optimal
                    # 디코딩 등 오프라인 재분석용 (TODO_VER8 P1). truth 필드와 같은
                    # 하위호환 원칙: 추가 필드라 기존 소비자(ab_gate 등)에 무해
                    "scores24": [round(float(x), 4) for x in scores],
                })
                if s.rank is not None:
                    rec["correct"] = (res.rank == s.rank)
                    rec["truth"] = submission.format_answer(s.rank)
            except Exception as e:  # noqa: BLE001 — 24h 런은 절대 죽지 않는다
                rec.update({"answer": submission.format_answer(perm.IDENTITY),
                            "error": f"{type(e).__name__}: {e}", "margin": 0.0,
                            "escalated": False})
            rec["sec"] = round(budget.end_sample(s.id), 3)
            prog.write(json.dumps(rec, ensure_ascii=False) + "\n")
            prog.flush()
            done[s.id] = rec

    # ---- 제출 파일 ----
    rows = []
    for s in samples:
        rec = done.get(s.id)
        rank = submission.parse_answer(rec["answer"]) if rec else perm.IDENTITY
        rows.append((s.id, rank))
    # Kaggle 채점기는 Answer를 문자열 그대로 비교 — train.csv와 동일한 "[1, 2, 3, 4]" 공백 형식 필수
    sub_path = submission.write_submission(out_dir / "submission.csv", rows, spaced=True)
    submission.validate_submission(sub_path, expected_ids=[s.id for s in samples])

    # ---- 리포트 ----
    recs = [done[s.id] for s in samples if s.id in done]
    report: dict = {
        "n": len(recs),
        "errors": sum(1 for r in recs if "error" in r),
        "escalated": sum(1 for r in recs if r.get("escalated")),
        "budget": budget.stats(),
        "submission": str(sub_path),
    }
    labeled = [r for r in recs if "correct" in r]
    if labeled:
        report["accuracy"] = sum(r["correct"] for r in labeled) / len(labeled)
        margins = [r["margin"] for r in labeled]
        report["margin_accuracy_table"] = margin_accuracy_table(
            margins, [r["correct"] for r in labeled])
        report["tau_suggestions"] = {
            f"escalate_{int(f*100)}pct": tau_by_escalation_budget(margins, f)
            for f in (0.1, 0.2, 0.3)}
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


# ---------------------------------------------------------------------------
# 스코어러 팩토리 + CLI
# ---------------------------------------------------------------------------

def build_scorer(args) -> tuple[object, object]:
    """(scorer, pairwise_factory) 생성. dummy는 torch 없이 동작."""
    if args.strategy == "dummy":
        sc = OracleScorer(noise=args.noise, seed=0)
        return sc, sc.make_pairwise_fn

    from .engine import EngineConfig, VLMEngine
    from .scorers import (CoTFSMScorer, LikelihoodScorer, MatchScorer,
                          PairwiseJudge, Score24Scorer)
    # video_max_pixels 자동계산: video는 전체 프레임 합산 예산제라, dup으로 늘어난
    # 프레임 수(len(images)*dup_factor)만큼 비례 상향해야 프레임당 해상도가 보존된다.
    # (자연크기 기반의 더 작은 값 대신 max_pixels 기반을 쓰는 이유는 engine.py 주석 참고)
    video_max_pixels = args.video_max_pixels
    if video_max_pixels is None and args.video_mode and args.video_dup_factor > 1:
        video_max_pixels = perm.N * args.video_dup_factor * args.max_pixels
    eng = VLMEngine(EngineConfig(
        model_id=args.model_id, four_bit=args.four_bit, adapter_path=args.adapter,
        max_pixels=args.max_pixels, min_pixels=args.min_pixels,
        video_max_pixels=video_max_pixels, video_dup_factor=args.video_dup_factor,
        device=args.device, attn=args.attn))
    print(f"[engine] {args.model_id} attn={eng.attn_used} device={eng.device}")
    if args.strategy == "score24":
        if not args.adapter:
            print("⚠️ score24는 파인튜닝(어댑터) 후에 유의미 — zero-shot이면 likelihood 권장")
        sc = Score24Scorer(eng, video_mode=args.video_mode,
                           counterfactual=args.counterfactual, legend=args.legend,
                           dup_factor=args.video_dup_factor)
    elif args.strategy == "likelihood":
        sc = LikelihoodScorer(eng, video_mode=args.video_mode)
    elif args.strategy == "match":
        sc = MatchScorer(eng)
    elif args.strategy == "cot":
        sc = CoTFSMScorer(eng, n_samples=args.cot_samples, video_mode=args.video_mode,
                          counterfactual=args.counterfactual, dup_factor=args.video_dup_factor)
    else:
        raise SystemExit(f"unknown strategy: {args.strategy}")
    judge = PairwiseJudge(eng)

    def pairwise_factory(caption, images):
        return judge.make_fn(caption, images)
    return sc, pairwise_factory


def load_samples(args) -> list[Sample]:
    if args.synthetic:
        from ..data.synthetic import make_dataset
        return make_dataset(args.synthetic, seed=42)
    samples = load_csv(args.csv, args.image_dir, caption_col=args.caption_col)
    if args.holdout_val:
        _, samples = split_samples(samples, val_frac=args.val_frac)
        print(f"[holdout] 검증 {len(samples)}건 사용")
        # 튜닝 결정(τ·A/B·체크포인트 선택)이 전부 이 split에 묶임을 기록 (재현·감사용)
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        with open(out / "split.json", "w", encoding="utf-8") as f:
            json.dump({"val_frac": args.val_frac, "n": len(samples),
                       "ids": [s.id for s in samples]}, f, ensure_ascii=False)
    if args.limit:
        samples = samples[: args.limit]
    return samples


def main(argv=None):
    ap = argparse.ArgumentParser(description="SNU AI Challenge 추론 파이프라인")
    d = ap.add_argument
    d("--csv"); d("--image-dir"); d("--caption-col", default="Caption")
    d("--synthetic", type=int, default=0, help="합성 샘플 N개로 드라이런")
    d("--holdout-val", action="store_true", help="train CSV에서 홀드아웃만 사용")
    d("--val-frac", type=float, default=0.1)
    d("--limit", type=int, default=0)
    d("--strategy", default="score24",
      choices=["score24", "likelihood", "match", "cot", "dummy"])
    d("--model-id", default="/home/yhmin/model/hub/Qwen3-VL-8B-Instruct")
    d("--adapter"); d("--four-bit", action="store_true")
    d("--device", default="auto"); d("--attn", default=None)
    # 학습 기본값(train_sft.py)과 동일한 602112 — 학습/추론 해상도 일치 원칙
    d("--max-pixels", type=int, default=602112); d("--min-pixels", type=int, default=None)
    # Ver3 기본 OFF (Ver2 하락 원인 — VER3.md). Ver2 어댑터로 추론할 때만 --video-mode 등으로 켤 것
    d("--video-mode", action=argparse.BooleanOptionalAction, default=False,
      help="4프레임을 비디오 채널로 입력 (Ver2 어댑터 호환용)")
    # Ver7 video_dup: R1(비디오 인코딩) 재도전 — 각 프레임 연속 복제로 temporal_patch_size=2
    # 병합쌍이 항상 자기 자신과의 쌍이 되게 함(무관 프레임 병합 오염 차단). 1 또는 짝수만 유효.
    d("--video-dup-factor", type=int, default=1,
      help="video-mode에서 각 프레임을 이만큼 연속 복제 (1 또는 짝수만 유효)")
    d("--video-max-pixels", type=int, default=None,
      help="video_processor 전용 예산(전체 프레임 합산제) — 생략 시 "
           "len(images)*video-dup-factor*max-pixels로 자동 계산")
    d("--counterfactual", action=argparse.BooleanOptionalAction, default=False,
      help="반사실적 자기검증 문구 추가 (Ver2 어댑터 호환용)")
    d("--legend", action=argparse.BooleanOptionalAction, default=True,
      help="A~X↔순열 범례 명시 — 학습 프롬프트와 일치 필수 (Ver1/Ver2 어댑터면 --no-legend)")
    d("--cot-samples", type=int, default=1, help="Self-Consistency 샘플 수")
    # Ver3 기본 1(TTA 없음) — 홀드아웃에서 이득이 실증될 때만 게이트 통과 후 상향
    d("--tta", type=int, default=1, dest="tta_views")
    d("--cascade", action="store_true"); d("--tau", type=float, default=0.15)
    d("--lam", type=float, default=1.0)
    d("--budget-hours", type=float, default=24.0)
    d("--noise", type=float, default=0.0, help="dummy 스코어러 노이즈")
    d("--out", default="runs/predict")
    d("--eval", action="store_true", help="라벨 있으면 정확도 리포트 출력")
    args = ap.parse_args(argv)

    # video_dup 가드 — 플래그 조합 오류를 조용히 무시하지 않는다(Ver5 --rescore와 동일 원칙).
    # load_samples/build_scorer(모델 로딩) 이전에 즉시 하드 실패시켜 잘못된 조합으로
    # 시간 낭비하지 않게 함.
    if args.video_dup_factor > 1 and not args.video_mode:
        raise SystemExit("--video-dup-factor>1은 --video-mode에서만 유효")
    if args.video_dup_factor != 1 and args.video_dup_factor % 2 != 0:
        raise SystemExit("--video-dup-factor는 1 또는 짝수만 유효 "
                         "(홀수는 temporal 병합쌍이 프레임 블록과 어긋나 일부 교차 오염됨)")

    samples = load_samples(args)
    scorer, pairwise_factory = build_scorer(args)
    cfg = PredictConfig(
        strategy=args.strategy, tta_views=args.tta_views,
        cascade=CascadeConfig(enable=args.cascade, tau=args.tau, lam=args.lam),
        budget_hours=args.budget_hours, out_dir=args.out)
    report = run_predict(samples, scorer, cfg, pairwise_factory=pairwise_factory)

    # sample_submission.csv가 데이터 옆에 있으면 표면 형식(헤더·spaced Answer)까지 대조
    if args.csv:
        ref = Path(args.csv).parent / "sample_submission.csv"
        if ref.exists():
            submission.assert_matches_sample_format(report["submission"], ref)
            print(f"[format] sample_submission 형식 대조 통과: {report['submission']}")

    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("margin_accuracy_table",)}, ensure_ascii=False, indent=2))
    if args.eval and "margin_accuracy_table" in report:
        print("\nmargin 구간별 정확도 (τ 캘리브레이션 근거):")
        for row in report["margin_accuracy_table"]:
            print(f"  [{row['margin_lo']:.3f}, {row['margin_hi']:.3f}] "
                  f"n={row['n']:4d}  acc={row['accuracy']:.3f}")
    return report


if __name__ == "__main__":
    main()
