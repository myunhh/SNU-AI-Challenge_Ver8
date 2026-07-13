"""제출 CSV 표면 형식 — spaced Answer 미준수로 0점 났던 사고의 회귀 테스트."""

import csv

import pytest

from snuai import perm, submission


def test_format_answer_spaced_matches_kaggle():
    rank0 = (1, 3, 0, 2)  # docstring 예시 — Kaggle Answer "[2,4,1,3]"
    assert submission.format_answer(rank0) == "[2,4,1,3]"
    assert submission.format_answer(rank0, spaced=True) == "[2, 4, 1, 3]"
    assert submission.parse_answer("[2, 4, 1, 3]") == rank0


def _write_sample_ref(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Answer"])
        for sid in ("a1", "b2"):
            w.writerow([sid, ""])


def test_write_submission_atomic_and_format_ok(tmp_path):
    ref = tmp_path / "sample_submission.csv"
    _write_sample_ref(ref)
    out = tmp_path / "submission.csv"
    rows = [("a1", (0, 1, 2, 3)), ("b2", (2, 0, 3, 1))]
    submission.write_submission(out, rows, spaced=True)
    assert not (tmp_path / "submission.csv.tmp").exists()  # 원자적 쓰기 잔여물 없음
    submission.validate_submission(out, expected_ids=["a1", "b2"])
    submission.assert_matches_sample_format(out, ref)  # 통과해야 함
    got = submission.read_submission(out)
    assert got["b2"] == (2, 0, 3, 1)


def test_unspaced_submission_rejected(tmp_path):
    ref = tmp_path / "sample_submission.csv"
    _write_sample_ref(ref)
    out = tmp_path / "submission.csv"
    submission.write_submission(out, [("a1", (0, 1, 2, 3))], spaced=False)  # 사고 재현
    with pytest.raises(ValueError, match="spaced"):
        submission.assert_matches_sample_format(out, ref)
