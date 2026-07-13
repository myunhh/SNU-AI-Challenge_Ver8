"""patch_prequant_vision_skip — 사전양자화 skip_modules 보정 (CPU 전용).

배경: unsloth Qwen3-VL-32B-Instruct-bnb-4bit는 vision 스킵을 bare name
('visual'/'vision_tower')으로 넣는데, transformers 5.x should_convert_module은
re.match 앵커링이라 'model.visual.*' 실경로에 안 걸려 vision이 4bit로 재양자화된다.
보정 헬퍼가 'model.visual'을 추가해 이를 막는지 검증.
"""

from types import SimpleNamespace

from snuai.train.qlora import VISION_SKIP_MODULES, patch_prequant_vision_skip


def _cfg(q):
    return SimpleNamespace(quantization_config=q)


def test_adds_model_visual_to_bare_skip_list():
    c = _cfg({"llm_int8_skip_modules": ["visual", "vision_tower"]})
    assert patch_prequant_vision_skip(c) is True
    skips = c.quantization_config["llm_int8_skip_modules"]
    assert "model.visual" in skips
    # 기존 항목은 지우지 않는다 (should_convert_module은 OR 매칭이라 더하기만 안전)
    assert "visual" in skips and "vision_tower" in skips


def test_idempotent():
    c = _cfg({"llm_int8_skip_modules": ["visual"]})
    assert patch_prequant_vision_skip(c) is True
    assert patch_prequant_vision_skip(c) is False  # 두 번째는 추가할 게 없음
    assert c.quantization_config["llm_int8_skip_modules"].count("model.visual") == 1


def test_no_quantization_config_is_noop():
    assert patch_prequant_vision_skip(SimpleNamespace(quantization_config=None)) is False
    assert patch_prequant_vision_skip(SimpleNamespace()) is False


def test_object_style_config():
    q = SimpleNamespace(llm_int8_skip_modules=["visual"])
    assert patch_prequant_vision_skip(_cfg(q)) is True
    assert "model.visual" in q.llm_int8_skip_modules


def test_missing_skip_key_defaults_empty():
    c = _cfg({})  # skip 키 자체가 없는 경우
    assert patch_prequant_vision_skip(c) is True
    assert c.quantization_config["llm_int8_skip_modules"] == list(VISION_SKIP_MODULES)


def test_behavioral_match_after_patch():
    """실제 transformers 매칭이 보정 후 vision을 스킵하는지 (버그 재발 방지선)."""
    from transformers.quantizers.quantizers_utils import should_convert_module
    name = "model.visual.blocks.0.attn.qkv"
    before = ["visual", "vision_tower"]
    assert should_convert_module(name, before) is True          # 보정 전: 양자화됨(버그)
    c = _cfg({"llm_int8_skip_modules": before})
    patch_prequant_vision_skip(c)
    after = c.quantization_config["llm_int8_skip_modules"]
    assert should_convert_module(name, after) is False          # 보정 후: 스킵(정상)
    # language 레이어는 여전히 양자화돼야 함
    assert should_convert_module("model.language_model.layers.0.mlp.gate_proj", after) is True
