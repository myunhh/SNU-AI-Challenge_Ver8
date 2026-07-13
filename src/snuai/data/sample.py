"""데이터 인터페이스 — 팀 adapter.py(3090 머신)와 이 파이프라인 사이의 계약.

파이프라인 전체는 아래 Sample만 소비한다. 팀 어댑터가 무엇을 읽든
(Kaggle CSV, 이미지 폴더 구조) 최종적으로 Sample 리스트로 변환해 주면 된다.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .. import perm
from ..submission import parse_answer


@dataclass
class Sample:
    """한 문제 = 캡션 1개 + 입력 이미지 4장 (+ 학습 시 정답 rank).

    images: PIL.Image 목록 또는 경로 목록 (lazy 로딩 허용).
    rank:   0-based rank 튜플 (Kaggle Answer의 의미, perm.py 규약). test는 None.
    """
    id: str
    caption: str
    images: list  # list[PIL.Image.Image] | list[str | Path]
    rank: perm.Perm | None = None
    meta: dict = field(default_factory=dict)

    def load_images(self):
        """경로면 PIL로 로드(RGB), 이미 이미지 객체면 그대로 반환."""
        from PIL import Image
        out = []
        for im in self.images:
            if isinstance(im, (str, Path)):
                out.append(Image.open(im).convert("RGB"))
            else:
                out.append(im)
        return out


def _first_id(csv_path: Path, id_col: str) -> str | None:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            return (row.get(id_col) or row.get("Id") or "").strip()
    return None


def _import_adapter(adapter_py: Path):
    """CSV 옆의 팀 adapter.py를 파일 경로 기준으로 import (cwd·sys.path 무관)."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("snuai_team_adapter", adapter_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod   # dataclass 등이 sys.modules에서 모듈을 찾음
    spec.loader.exec_module(mod)
    return mod


def load_csv(csv_path: str | Path,
             image_dir: str | Path,
             id_col: str = "Id",
             caption_col: str = "Caption",
             answer_col: str = "Answer",
             image_path_fn: Callable[[str, int], str] | None = None) -> list[Sample]:
    """일반형 CSV 로더. CSV 옆에 팀 adapter.py가 있으면(=Kaggle 실데이터) 그쪽으로 위임.

    어댑터 사용 시 이미지 해석은 caller의 image_dir을 그대로 존중한다:
      - image_dir 안에 <Id>/ 폴더가 있으면(원본 레이아웃) 어댑터의 fuzzy 해석
      - 평면 레이아웃({id}_{k}.jpg — importance.preprocess_samples 적용본 등)이면
        CSV 컬럼만 Kaggle 규약(Sentence/Answer)으로 읽고 이미지는 평면 경로 사용
    """
    csv_path, image_dir = Path(csv_path), Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image_dir이 존재하지 않음: {image_dir}")

    adapter_py = csv_path.parent / "adapter.py"   # cwd가 아니라 CSV 위치 기준으로 탐지
    if adapter_py.exists():
        first_id = _first_id(csv_path, id_col)
        if first_id and (image_dir / first_id).is_dir():
            load_samples = _import_adapter(adapter_py).load_samples
            # root/split == image_dir이 되도록 분해해 전달 (image_dir 존중)
            adapter_samples = load_samples(image_dir.parent, image_dir.name,
                                           csv_path=csv_path)
            return [Sample(id=s.id, caption=s.sentence, images=list(s.image_paths),
                           rank=(perm.order_to_rank(s.order) if s.order is not None else None))
                    for s in adapter_samples]
        # 평면 레이아웃: 아래 일반 루프를 Kaggle CSV 규약으로 실행
        caption_col = "Sentence"

    if image_path_fn is None:
        image_path_fn = lambda sid, k: f"{sid}_{k}.jpg"  # noqa: E731
    out: list[Sample] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sid = row[id_col].strip()
            rank = None
            if answer_col in row and (row[answer_col] or "").strip():
                rank = parse_answer(row[answer_col])
            paths = [image_dir / image_path_fn(sid, k) for k in range(perm.N)]
            out.append(Sample(id=sid, caption=row[caption_col].strip(),
                              images=list(paths), rank=rank))
    return out
