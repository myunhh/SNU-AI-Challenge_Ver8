"""TTA 리맵 · SFT 라벨 경로 · legend 범례 — Ver3 핵심 안전망.

전부 CPU/모델 불필요. 라벨 경로 테스트는 실제 augment_sample을 통과시켜
gather/argsort 혼동이 어디서 생겨도 잡히게 한다.
"""

import random

import numpy as np
from PIL import Image

from snuai import perm
from snuai.data.augment import AugmentConfig
from snuai.data.sample import Sample
from snuai.infer.tta import TTAConfig, tta_scores
from snuai.prompting import (SCORE24_INSTRUCTION_LEGEND, build_score24_messages,
                             extract_media, score24_legend)
from snuai.train.dataset import Score24SFTDataset, SFTDatasetConfig


def _marker_image(k: int) -> Image.Image:
    """R 채널에 원본 슬롯 번호를 새긴 8x8 마커 이미지."""
    return Image.new("RGB", (8, 8), color=(k, 0, 0))


def _marker_of(im: Image.Image) -> int:
    return im.getpixel((0, 0))[0]


def test_tta_remap_recovers_truth_for_all_views():
    # 이미지 = 각 프레임의 시간순 위치(int). rank[i] = images[i] 그 자체.
    rng = random.Random(7)
    for _ in range(100):
        rank = perm.random_shuffle(rng)
        images = list(rank)

        def oracle(shuffled_imgs):
            view_rank = tuple(shuffled_imgs)
            out = np.full(24, -10.0)
            out[perm.index_of(view_rank)] = 0.0
            return out

        agg, per_view = tta_scores(images, oracle, TTAConfig(n_views=5, seed=123))
        assert int(agg.argmax()) == perm.index_of(rank)
        for v in per_view:  # 모든 뷰가 리맵 후 같은 정답을 가리켜야 함
            assert int(np.asarray(v).argmax()) == perm.index_of(rank)


def test_dataset_label_consistent_through_augment():
    """증강(셔플) 후 target letter가 '보이는 이미지 배열' 기준으로 옳은지 독립 검증."""
    base_rank = (2, 0, 3, 1)  # 자기역원 아님 — 규약 혼동이 드러나는 케이스
    sample = Sample(id="t", caption="c", images=[_marker_image(k) for k in range(4)],
                    rank=base_rank)
    cfg = SFTDatasetConfig(
        augment=AugmentConfig(perm_mode="uniform"),  # 색상증강 전부 0 (기본값)
        video_mode=False, counterfactual=False, legend=True)
    ds = Score24SFTDataset([sample], cfg)

    seen_nontrivial = 0
    for idx in range(50):
        item = ds[idx]
        shown, _ = extract_media(item["messages"])
        assert len(shown) == 4
        # 독립 재계산: 슬롯 j에 보이는 이미지의 원본 슬롯 m → 올바른 rank[j] = base_rank[m]
        expected_rank = tuple(base_rank[_marker_of(im)] for im in shown)
        assert item["rank"] == expected_rank
        assert item["target_text"] == perm.letter_of_rank(expected_rank)
        if expected_rank != base_rank:
            seen_nontrivial += 1
    assert seen_nontrivial > 10  # 셔플이 실제로 일어나는지


def test_legend_derived_from_perms():
    leg = score24_legend()
    entries = dict(e.split("=") for e in leg.split(", "))
    assert len(entries) == 24
    assert entries["A"] == "1234"
    assert entries["N"] == "3142"
    assert entries["X"] == "4321"
    for i, p in enumerate(perm.PERMS24):
        assert entries[perm.LETTERS24[i]] == "".join(str(v + 1) for v in p)


def test_score24_messages_legend_toggle():
    imgs = [_marker_image(k) for k in range(4)]
    with_leg = build_score24_messages("cap", imgs, legend=True)
    without = build_score24_messages("cap", imgs, legend=False)

    def text_of(msgs):
        return "\n".join(c["text"] for m in msgs for c in m["content"]
                         if c.get("type") == "text")

    assert "Letter code: A=1234" in text_of(with_leg)
    assert "Letter code" not in text_of(without)
    # 포맷 문자열이 실제로 치환됐는지 (규약 사고 방지)
    assert "{legend}" not in text_of(with_leg)
    assert "{legend}" in SCORE24_INSTRUCTION_LEGEND
