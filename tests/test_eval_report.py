"""scripts/eval_report.py의 순수 함수 — 합성 progress 레코드로 손계산과 대조 (CPU 전용)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from eval_report import compute_diff_stats, compute_forensic_stats, load_records  # noqa: E402

from snuai import perm


def _rec(id_, truth, pred, margin=0.5):
    return {"id": id_, "truth": truth, "pred": pred, "margin": margin}


def test_compute_forensic_stats_basic_em_and_margin():
    records = [
        _rec("a", (0, 1, 2, 3), (0, 1, 2, 3), margin=0.9),   # identity, correct
        _rec("b", (2, 0, 3, 1), (2, 0, 3, 1), margin=0.7),   # non-identity, correct
        _rec("c", (2, 0, 3, 1), (0, 2, 1, 3), margin=0.1),   # non-identity, wrong
    ]
    stats = compute_forensic_stats(records)
    assert stats["n"] == 3
    assert stats["em"] == 2 / 3
    assert stats["margin_mean"] == pytest.approx((0.9 + 0.7 + 0.1) / 3)
    assert stats["identity_n"] == 1 and stats["identity_em"] == 1.0
    assert stats["nonidentity_n"] == 2 and stats["nonidentity_em"] == 0.5


def test_compute_forensic_stats_identity_false_positive():
    records = [_rec("a", (2, 0, 3, 1), perm.IDENTITY, margin=0.2)]  # pred=identity, truth!=identity
    stats = compute_forensic_stats(records)
    assert stats["identity_false_positives"] == 1


def test_compute_forensic_stats_reversal_rate():
    # (2,0,3,1)의 역순열 자기 자신 여부 확인 후, pred==inverse(truth)인 케이스 구성
    truth = (2, 0, 3, 1)
    assert perm.inverse(truth) != truth  # non-involution 확인
    records = [
        _rec("a", truth, perm.inverse(truth), margin=0.3),  # 역순열 응답
        _rec("b", truth, truth, margin=0.3),                # 정답
    ]
    stats = compute_forensic_stats(records)
    assert stats["reversal_n"] == 2
    assert stats["reversal_rate"] == 0.5


def test_compute_forensic_stats_kendall_histogram_sums_to_n():
    records = [
        _rec("a", (0, 1, 2, 3), (0, 1, 2, 3)),
        _rec("b", (0, 1, 2, 3), (1, 0, 2, 3)),  # KT=1 (인접 스왑)
        _rec("c", (0, 1, 2, 3), (3, 2, 1, 0)),  # KT=6 (완전 역전)
    ]
    stats = compute_forensic_stats(records)
    hist = stats["kendall_tau_histogram"]
    assert sum(hist.values()) == 3
    assert hist[0] == 1
    assert hist[1] == 1
    assert hist[6] == 1


def test_compute_diff_stats_agreement_and_bucketed_em():
    curr = [
        _rec("a", (0, 1, 2, 3), (0, 1, 2, 3)),  # curr correct, agrees with prev
        _rec("b", (2, 0, 3, 1), (0, 2, 1, 3)),  # curr wrong, disagrees with prev
    ]
    prev = [
        _rec("a", (0, 1, 2, 3), (0, 1, 2, 3)),  # prev correct, agrees
        _rec("b", (2, 0, 3, 1), (2, 0, 3, 1)),  # prev correct, disagrees with curr's pred
    ]
    diff = compute_diff_stats(curr, prev)
    assert diff["common_n"] == 2
    assert diff["agree_n"] == 1 and diff["disagree_n"] == 1
    assert diff["curr_em_on_agree"] == 1.0
    assert diff["curr_em_on_disagree"] == 0.0
    assert diff["prev_em_on_disagree"] == 1.0  # "불일치 구간에서 이전 버전은 맞았다" 패턴 재현


def test_compute_diff_stats_no_common_ids():
    assert compute_diff_stats([_rec("a", (0, 1, 2, 3), (0, 1, 2, 3))],
                              [_rec("z", (0, 1, 2, 3), (0, 1, 2, 3))]) == {"common_n": 0}


def test_load_records_uses_embedded_truth_field(tmp_path):
    progress = tmp_path / "progress.jsonl"
    rows = [
        {"id": "a", "truth": "[1, 2, 3, 4]", "answer": "[1, 2, 3, 4]", "margin": 0.9,
         "correct": True},
        {"id": "b", "truth": "[3, 1, 4, 2]", "answer": "[1, 2, 3, 4]", "margin": 0.1,
         "correct": False},
        {"id": "c", "answer": "[1, 2, 3, 4]", "margin": 0.0,
         "error": "boom"},  # 실패 fallback — 스킵돼야 함
    ]
    progress.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    records = load_records(str(progress))
    assert len(records) == 2  # error 레코드는 제외
    assert records[0]["truth"] == (0, 1, 2, 3)
    assert records[1]["pred"] == (0, 1, 2, 3)


def test_load_records_missing_truth_without_csv_raises(tmp_path):
    progress = tmp_path / "progress.jsonl"
    progress.write_text(json.dumps({"id": "a", "answer": "[1, 2, 3, 4]", "margin": 0.5}),
                        encoding="utf-8")
    try:
        load_records(str(progress))
        assert False, "should have raised"
    except SystemExit:
        pass


def test_compute_forensic_stats_partial_credit_metrics():
    records = [
        _rec("a", (0, 1, 2, 3), (0, 1, 2, 3)),   # 일치: pairwise 1, position 1
        _rec("b", (0, 1, 2, 3), (1, 0, 2, 3)),   # 인접 스와프: 5/6, 0.5
    ]
    stats = compute_forensic_stats(records)
    assert stats["pairwise_score"] == pytest.approx((1 + 5 / 6) / 2)
    assert stats["position_score"] == pytest.approx((1 + 0.5) / 2)
