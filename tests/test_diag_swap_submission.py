"""scripts/diag_swap_submission.py — §1 진단 제출 생성 (CPU 전용, 실제 파일 I/O로 검증)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from diag_swap_submission import main  # noqa: E402

from snuai import perm, submission  # noqa: E402


def _write_progress(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_swaps_only_top_k_highest_margin(tmp_path):
    # margin 내림차순: s0(0.99) > s1(0.9) > s2(0.5) > s3(0.1)
    records = [
        {"id": "s0", "answer": "[1,2,3,4]", "margin": 0.99},
        {"id": "s1", "answer": "[2,1,3,4]", "margin": 0.90},
        {"id": "s2", "answer": "[1,2,4,3]", "margin": 0.50},
        {"id": "s3", "answer": "[4,3,2,1]", "margin": 0.10},
    ]
    prog = tmp_path / "progress.jsonl"
    _write_progress(prog, records)
    out = tmp_path / "submission.csv"

    main(["--progress", str(prog), "--k", "2", "--swap-index", "0", "--out", str(out)])

    got = submission.read_submission(out)
    assert set(got) == {"s0", "s1", "s2", "s3"}
    orig = {r["id"]: submission.parse_answer(r["answer"]) for r in records}
    # 상위 margin 2건(s0, s1)만 인접 스와프(KT거리 1) 적용, 나머지는 원본 그대로
    assert perm.kendall_tau_distance(got["s0"], orig["s0"]) == 1
    assert perm.kendall_tau_distance(got["s1"], orig["s1"]) == 1
    assert got["s2"] == orig["s2"]
    assert got["s3"] == orig["s3"]


def test_falls_back_to_scores24_when_margin_missing(tmp_path):
    import numpy as np
    scores = np.full(24, -10.0)
    scores[perm.index_of((0, 1, 2, 3))] = 5.0
    scores[perm.index_of((1, 0, 2, 3))] = 4.0   # 준2등 — margin이 크지 않음(스와프 대상 아님 검증용)
    records = [
        {"id": "a", "answer": "[1,2,3,4]", "scores24": scores.tolist()},
        {"id": "b", "answer": "[2,1,3,4]", "scores24": [0.0] * 24},  # 완전 균등 → margin 0
    ]
    prog = tmp_path / "progress.jsonl"
    _write_progress(prog, records)
    out = tmp_path / "submission.csv"

    main(["--progress", str(prog), "--k", "1", "--out", str(out)])
    got = submission.read_submission(out)
    # margin이 더 큰 쪽(a)만 스와프됨
    assert perm.kendall_tau_distance(got["a"], (0, 1, 2, 3)) == 1
    assert got["b"] == (1, 0, 2, 3)


def test_rejects_k_out_of_range(tmp_path):
    import pytest
    records = [{"id": "a", "answer": "[1,2,3,4]", "margin": 0.5}]
    prog = tmp_path / "progress.jsonl"
    _write_progress(prog, records)
    with pytest.raises(SystemExit):
        main(["--progress", str(prog), "--k", "0", "--out", str(tmp_path / "out.csv")])
    with pytest.raises(SystemExit):
        main(["--progress", str(prog), "--k", "2", "--out", str(tmp_path / "out.csv")])
