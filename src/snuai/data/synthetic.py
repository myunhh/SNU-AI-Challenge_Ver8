"""합성 픽스처 — GPU/실데이터 없이 파이프라인 전체를 검증하기 위한 가짜 문제 생성기.

"공이 왼쪽에서 오른쪽으로 이동" 스토리를 4프레임으로 그림.
프레임에 시간 단계가 시각적으로 인코딩되어 있어(공 위치), 사람이 육안으로도
순서를 판별할 수 있고, 오라클 스코어러(픽셀에서 위치 읽기)로 정답 복원이 가능하다.
"""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

from .. import perm
from .sample import Sample

SIZE = 192
BALL_R = 18
_XS = [30, 74, 118, 162]  # 시간 단계 0..3의 공 중심 x좌표


def make_frame(step: int, size: int = SIZE) -> Image.Image:
    """시간 단계 step(0..3)의 프레임: 회색 배경 + 빨간 공(x가 step에 비례)."""
    img = Image.new("RGB", (size, size), (200, 200, 205))
    d = ImageDraw.Draw(img)
    d.rectangle([0, size - 30, size, size], fill=(90, 140, 90))  # 바닥
    x = _XS[step] * size // SIZE
    y = size // 2
    d.ellipse([x - BALL_R, y - BALL_R, x + BALL_R, y + BALL_R], fill=(210, 40, 40))
    return img


def read_step_from_frame(img: Image.Image) -> int:
    """오라클: 프레임에서 공의 시간 단계를 픽셀로 판독(빨간 픽셀 x 평균)."""
    w, h = img.size
    px = img.load()
    xs = [x for y in range(0, h, 4) for x in range(0, w, 2)
          if px[x, y][0] > 150 and px[x, y][1] < 100]
    cx = sum(xs) / len(xs) * SIZE / w
    return min(range(4), key=lambda k: abs(_XS[k] - cx))


def make_sample(sample_id: str, rng: random.Random) -> Sample:
    """rank를 랜덤 추출하고 그에 맞게 프레임을 섞은 합성 Sample."""
    rank = perm.random_shuffle(rng)
    order = perm.rank_to_order(rank)
    # order[p] = 시간순 p번째에 오는 슬롯 ⇒ 슬롯 i에는 단계 rank[i]의 프레임
    images = [make_frame(rank[i]) for i in range(perm.N)]
    caption = ("A red ball rolls across the ground from the left side to the "
               "right side of the scene.")
    return Sample(id=sample_id, caption=caption, images=images, rank=rank,
                  meta={"synthetic": True, "order": order})


def make_dataset(n: int, seed: int = 0) -> list[Sample]:
    rng = random.Random(seed)
    return [make_sample(f"syn_{i:04d}", rng) for i in range(n)]


def oracle_rank(sample: Sample) -> perm.Perm:
    """이미지 픽셀만 보고 rank를 복원 — 전처리·증강의 라벨 일관성 검증용."""
    steps = [read_step_from_frame(im) for im in sample.load_images()]
    if sorted(steps) != [0, 1, 2, 3]:
        raise ValueError(f"프레임 단계 판독 실패: {steps}")
    return tuple(steps)  # rank[i] = 슬롯 i의 시간 단계
