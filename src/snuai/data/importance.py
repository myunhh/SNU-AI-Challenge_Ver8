"""캡션 유도 시각적 중요도 전처리 (Caption-guided Visual Importance Downscaling).

노션 §10 설계 그대로:
  1) 오픈소스 CLIP/SigLIP로 캡션↔이미지 패치 유사도 맵을 이미지당 1회 계산
  2) .npy로 디스크 캐싱 (TTA로 순서를 바꿔도 재계산 없음)
  3) 저중요 영역만 블러/다운스케일 — crop 금지(정보 삭제 ✖), 원본 항상 보관
  4) 전처리본을 별도 디렉터리에 저장, 경로 스위치로 원본↔전처리본 전환

⚠️ 이 변환은 '결정적 전처리'다. 학습·추론 중 한쪽에만 적용하는 것은 분포
   불일치이므로, 반드시 exp_preproc_ab.py의 홀드아웃 A/B를 통과한 조합만 채택.
   (안전한 기본값: 양쪽 모두 적용 or 양쪽 모두 미적용)

순수 로직(맵 정규화·마스크·합성·캐시)은 CPU 단위테스트로 검증하고,
SigLIP 모델 래퍼는 3090(또는 tiny 모델 통합 테스트)에서 검증한다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# 1) 순수 로직 — 유사도 맵 → 선택적 소프닝 (GPU 불필요, 단위테스트 대상)
# ---------------------------------------------------------------------------

def normalize_map(m: np.ndarray) -> np.ndarray:
    """유사도 맵 → [0,1]. 상수 맵이면 전부 1(아무것도 저중요로 만들지 않음 = 안전)."""
    m = m.astype(np.float64)
    lo, hi = float(m.min()), float(m.max())
    if hi - lo < 1e-12:
        return np.ones_like(m)
    return (m - lo) / (hi - lo)


def upsample_map(m: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """패치 그리드 맵(h×w) → 이미지 크기(W,H) bilinear 업샘플, [0,1] 유지."""
    img = Image.fromarray((normalize_map(m) * 255).astype(np.uint8), mode="L")
    return np.asarray(img.resize(size, Image.BILINEAR), dtype=np.float64) / 255.0


@dataclass(frozen=True)
class SoftenConfig:
    keep_quantile: float = 0.5   # 중요도 상위 (1-q) 비율은 원본 유지
    mode: str = "blur"           # "blur" | "downscale"
    blur_radius: float = 3.0
    down_factor: int = 4
    mask_soft_radius: float = 8.0  # 마스크 경계 소프닝(경계 아티팩트 방지)


def selective_soften(img: Image.Image, imp_map: np.ndarray,
                     cfg: SoftenConfig = SoftenConfig()) -> Image.Image:
    """저중요 영역만 흐리게 한 새 이미지 반환. 원본은 불변.

    imp_map: 패치 그리드(h×w) 또는 이미지 크기 맵. 값이 클수록 중요.
    """
    if not (0.0 <= cfg.keep_quantile < 1.0):
        raise ValueError(f"keep_quantile 범위 오류: {cfg.keep_quantile}")
    w, h = img.size
    m = upsample_map(imp_map, (w, h)) if imp_map.shape != (h, w) else normalize_map(imp_map)

    thresh = float(np.quantile(m, cfg.keep_quantile))
    low_mask = (m < thresh).astype(np.uint8) * 255  # 255 = 소프닝 대상(저중요)
    mask_img = Image.fromarray(low_mask, mode="L")
    if cfg.mask_soft_radius > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(cfg.mask_soft_radius))

    if cfg.mode == "blur":
        soft = img.filter(ImageFilter.GaussianBlur(cfg.blur_radius))
    elif cfg.mode == "downscale":
        f = max(2, int(cfg.down_factor))
        soft = img.resize((max(1, w // f), max(1, h // f)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    else:
        raise ValueError(f"mode: {cfg.mode}")

    return Image.composite(soft, img, mask_img)  # mask=255 → soft 픽셀 채택


# ---------------------------------------------------------------------------
# 2) 디스크 캐시 — 이미지×캡션×모델당 1회 계산 보장
# ---------------------------------------------------------------------------

def cache_key(image_key: str, caption: str, model_id: str) -> str:
    return hashlib.md5(f"{model_id}|{caption}|{image_key}".encode()).hexdigest()


class MapCache:
    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.npy"

    def get(self, key: str) -> np.ndarray | None:
        p = self.path(key)
        return np.load(p) if p.exists() else None

    def put(self, key: str, m: np.ndarray) -> None:
        tmp = self.path(key).with_suffix(".tmp.npy")
        np.save(tmp, m.astype(np.float32))
        tmp.replace(self.path(key))


# ---------------------------------------------------------------------------
# 3) SigLIP/CLIP 래퍼 — 캡션↔패치 유사도 (GPU 권장, lazy import)
# ---------------------------------------------------------------------------

class ClipPatchScorer:
    """캡션 텍스트 임베딩과 비전 패치 임베딩의 코사인 유사도 그리드를 계산.

    SigLIP(권장: google/siglip2-base-patch16-256)과 CLIP 계열 모두 동작:
    vision_model.last_hidden_state의 패치 토큰 × 텍스트 pooled 임베딩.
    """

    def __init__(self, model_id: str = "google/siglip2-base-patch16-256",
                 device: str = "cuda", dtype=None):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.model_id = model_id
        self.device = device
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        kw = {"dtype": dtype} if dtype is not None else {}
        self.model = AutoModel.from_pretrained(model_id, **kw).to(device).eval()

    @staticmethod
    def _grid_hw(n_patches: int) -> tuple[int, int]:
        g = int(round(n_patches ** 0.5))
        if g * g != n_patches:
            raise ValueError(f"정사각 패치 그리드가 아님: {n_patches}")
        return g, g

    def maps(self, images: list[Image.Image], caption: str) -> list[np.ndarray]:
        """이미지 리스트 → 패치 그리드 유사도 맵 리스트 (값 클수록 캡션과 유사)."""
        torch = self.torch
        with torch.no_grad():
            ti = self.processor(text=[caption], return_tensors="pt", padding=True,
                                truncation=True).to(self.device)
            t = self.model.text_model(**ti).pooler_output  # (1, D)
            t = torch.nn.functional.normalize(t.float(), dim=-1)

            vi = self.processor(images=images, return_tensors="pt").to(self.device)
            v = self.model.vision_model(pixel_values=vi["pixel_values"]).last_hidden_state
            # CLIP류는 [CLS]가 앞에 붙음 → 정사각이 안 되면 첫 토큰 제거
            n = v.shape[1]
            if int(round(n ** 0.5)) ** 2 != n and int(round((n - 1) ** 0.5)) ** 2 == n - 1:
                v = v[:, 1:]
            v = torch.nn.functional.normalize(v.float(), dim=-1)  # (B, P, D)
            sim = (v @ t.T).squeeze(-1)  # (B, P)

        out = []
        h, w = self._grid_hw(sim.shape[1])
        for row in sim.cpu().numpy():
            out.append(row.reshape(h, w))
        return out


# ---------------------------------------------------------------------------
# 4) 배치 전처리 잡 — 전처리본 생성 (3090에서 1회 실행)
# ---------------------------------------------------------------------------

def preprocess_samples(samples, scorer, cache_dir: str | Path, out_dir: str | Path,
                       soften: SoftenConfig = SoftenConfig(), overwrite: bool = False,
                       log_every: int = 200) -> list[Path]:
    """samples의 모든 이미지에 선택적 소프닝 적용본을 out_dir에 저장.

    원본은 절대 수정하지 않는다. 반환: 저장된 파일 경로 목록.
    학습이든 추론이든 '적용본을 쓸지'는 Sample.images 경로 교체로 스위치.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = MapCache(cache_dir)
    written: list[Path] = []
    for si, s in enumerate(samples):
        imgs = s.load_images()
        keys = [cache_key(f"{s.id}#{k}", s.caption, scorer.model_id) for k in range(len(imgs))]
        maps = [cache.get(k) for k in keys]
        if any(m is None for m in maps):
            fresh = scorer.maps(imgs, s.caption)
            for k, m in zip(keys, fresh):
                cache.put(k, m)
            maps = fresh
        for k, (img, m) in enumerate(zip(imgs, maps)):
            dst = out_dir / f"{s.id}_{k}.jpg"
            if dst.exists() and not overwrite:
                written.append(dst)
                continue
            selective_soften(img, m, soften).save(dst, quality=95)
            written.append(dst)
        if log_every and (si + 1) % log_every == 0:
            print(f"[importance] {si + 1}/{len(samples)} 샘플 처리")
    return written
