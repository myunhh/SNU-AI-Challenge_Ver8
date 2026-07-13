"""캡션 이벤트 분해 규칙 — 실데이터 관찰 패턴 기반 (train 9,535건 실측으로 설계).

핵심 불변식: 전진 연결어만 자른다 — 잘린 조각의 텍스트 순서 = 시간 순서.
동시("while X", "as X")·역순(bare "after")은 자르면 순서가 틀어지므로 안 자른다.
"""

from snuai.data.decompose import split_events_rule


def test_forward_connectives_split():
    ev = split_events_rule(
        "A cyclist rides down a street, transitioning to a website link; "
        "then, a hand adjusts the brake alignment.")
    assert len(ev) == 3
    assert ev[0].startswith("A cyclist rides")
    assert ev[1].startswith("a website link")
    assert ev[2].startswith("a hand adjusts")


def test_followed_by_and_before_ing():
    ev = split_events_rule(
        "The tire is removed and a patch is applied, followed by a component "
        "being secured before transitioning to the next step.")
    assert len(ev) == 3
    assert ev[1].startswith("a component being secured")
    assert ev[2].startswith("to the next step") or "next step" in ev[2]


def test_simultaneous_while_as_not_split():
    # "while"/"as"는 동시 동작 — 자르면 순서 왜곡이므로 통째로 유지
    ev = split_events_rule(
        "The performer bows with a sword, while the gymnast descends from midair "
        "as the camera zooms in.")
    assert len(ev) == 1


def test_bare_after_not_split():
    # "X after Y"는 Y가 시간상 먼저 — 텍스트 순서 분해가 순서를 뒤집으므로 안 자름
    ev = split_events_rule("The child swings from the structure after standing on a post.")
    assert len(ev) == 1


def test_never_empty_and_max_events():
    assert split_events_rule("") != []
    many = ", then ".join(f"actor does thing {i}" for i in range(12))
    assert len(split_events_rule(many, max_events=8)) == 8


def test_parse_numbered_events_and_fallback():
    from snuai.data.decompose import parse_numbered_events
    txt = "1. A man picks up a ball\n2) He throws it far\n- the dog catches it"
    ev = parse_numbered_events(txt, "fallback caption here")
    assert ev == ["A man picks up a ball", "He throws it far", "the dog catches it"]
    # 번호 형식이 전혀 없으면 규칙 기반 폴백 (절대 빈 리스트 없음)
    fb = parse_numbered_events("no structure at all", "The cat jumps, then lands softly.")
    assert len(fb) >= 1


def test_load_events_cache_roundtrip(tmp_path):
    import json
    from snuai.data.decompose import load_events_cache
    p = tmp_path / "events.jsonl"
    rows = [{"id": "a1", "caption": "c", "events": ["e1 x", "e2 y"], "raw": "r"},
            {"id": "b2", "caption": "d", "events": ["only one"], "raw": "r"}]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    got = load_events_cache(p)
    assert got == {"a1": ["e1 x", "e2 y"], "b2": ["only one"]}
