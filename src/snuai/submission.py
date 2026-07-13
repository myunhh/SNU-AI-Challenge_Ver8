"""Kaggle 제출 파일 (Id, Answer) 입출력 + 검증.

Answer 포맷: "[r1,r2,r3,r4]" — 각 입력 이미지의 시간순 순위(1-based).
train.csv의 Answer 인코딩과 동일함을 팀 어댑터에서 이미 검증했음(노션 07.08).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Sequence

from . import perm

_ANSWER_RE = re.compile(r"^\s*\[?\s*([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\s*\]?\s*$")


def format_answer(rank0: Sequence[int], spaced: bool = False) -> str:
    """0-based rank 튜플 → Answer 문자열. 기본은 공백 없는 "[2,4,1,3]" 형태."""
    r1 = perm.to_1based(rank0)
    sep = ", " if spaced else ","
    return "[" + sep.join(str(v) for v in r1) + "]"


def parse_answer(s: str) -> perm.Perm:
    """Answer 문자열 → 0-based rank 튜플. 공백/괄호 유무에 관대, 순열 검증은 엄격."""
    m = _ANSWER_RE.match(s)
    if not m:
        raise ValueError(f"Answer 파싱 실패: {s!r}")
    return perm.from_1based(int(g) for g in m.groups())


def write_submission(path: str | Path, rows: Iterable[tuple[str, Sequence[int]]],
                     spaced: bool = False) -> Path:
    """(id, 0-based rank) 목록 → 제출 CSV. Answer에 콤마가 있으므로 반드시 quoting.

    원자적 쓰기(tmp+rename): 어떤 중단 시점에도 이전 제출물이 깨지지 않는다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["Id", "Answer"])
        for sample_id, rank0 in rows:
            w.writerow([sample_id, format_answer(rank0, spaced=spaced)])
    tmp.replace(path)
    return path


_SPACED_ANSWER_RE = re.compile(r"^\[[1-4], [1-4], [1-4], [1-4]\]$")


def assert_matches_sample_format(path: str | Path, sample_path: str | Path) -> None:
    """제출 CSV가 sample_submission과 같은 표면 형식인지 검사 (0점 사고 재발 방지).

    검사: ① 헤더 라인 바이트 동일 ② 모든 Answer가 공백 포함 "[1, 2, 3, 4]" 형식.
    (Kaggle 채점기는 Answer를 문자열 그대로 비교 — spaced=False 제출이 0점 났던 이력)
    """
    with open(sample_path, encoding="utf-8") as f:
        ref_header = f.readline()
    with open(path, encoding="utf-8") as f:
        got_header = f.readline()
        if got_header != ref_header:
            raise ValueError(f"헤더 불일치: {got_header!r} ≠ sample {ref_header!r}")
        for row in csv.DictReader(f, fieldnames=[c.strip() for c in ref_header.split(",")]):
            ans = row["Answer"]
            if not _SPACED_ANSWER_RE.match(ans):
                raise ValueError(f"Answer 형식 위반 (spaced 필요): {ans!r} @ Id={row['Id']}")


def read_submission(path: str | Path) -> dict[str, perm.Perm]:
    """제출 CSV → {id: 0-based rank}. 재개(resume)·검증 겸용."""
    out: dict[str, perm.Perm] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or [c.strip() for c in r.fieldnames[:2]] != ["Id", "Answer"]:
            raise ValueError(f"헤더가 'Id,Answer'가 아님: {r.fieldnames}")
        for row in r:
            sid = row["Id"].strip()
            if sid in out:
                raise ValueError(f"중복 Id: {sid}")
            out[sid] = parse_answer(row["Answer"])
    return out


def validate_submission(path: str | Path, expected_ids: Sequence[str] | None = None) -> dict:
    """제출 직전 최종 검증. 반환: 통계 dict. 문제 발견 시 ValueError.

    검사 항목: 헤더, Id 중복, Answer 순열 유효성, (선택) 테스트 Id 집합과 완전 일치.
    """
    got = read_submission(path)  # 파싱 실패/중복이면 여기서 raise
    stats = {"n_rows": len(got), "n_identity": sum(1 for r in got.values() if r == perm.IDENTITY)}
    if expected_ids is not None:
        exp = set(map(str, expected_ids))
        missing = exp - set(got)
        extra = set(got) - exp
        if missing or extra:
            raise ValueError(f"Id 불일치 — 누락 {len(missing)}개 {sorted(missing)[:5]}..., "
                             f"초과 {len(extra)}개 {sorted(extra)[:5]}...")
        stats["ids_match"] = True
    return stats
