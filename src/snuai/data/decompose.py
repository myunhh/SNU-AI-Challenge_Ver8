"""캡션 이벤트 분해 (Caption Decomposition) — decompose-and-match의 입력.

두 경로 제공:
  1) 규칙 기반(split_events_rule): 비용 0, 오프라인, 항상 동작 — 기본값/폴백
  2) 모델 기반(build_decompose_messages + parse_numbered_events):
     같은 VLM의 텍스트 전용 호출(외부 API 아님 → 규정 허용)

노션 확인사항: 캡션이 항상 4문장으로 깔끔히 나뉘지 않음 → 문장 분리 대신
절(clause) 단위 분리를 지원하고, 이벤트 수 k≠4 도 match.py가 처리한다.
"""

from __future__ import annotations

import re

# 시간 연결어 — 앞에서 자르는 지점 (대소문자 무시).
#
# ⚠️ "전진(forward) 연결어"만 자른다 — 텍스트 순서 = 시간 순서가 보장되는 것만.
#    bare "after"(역순: "X after Y"는 Y가 먼저)나 "while/as"(동시 동작)로 자르면
#    이벤트 순서가 틀어지므로 넣지 않는다. 그런 캡션은 모델 기반 분해의 몫.
#
# 실측 (train 9,535 캡션, 2026-07-10): 기존 규칙은 44.1%가 분해 실패(1개)였으나
# 아래 확장(followed by/before ~ing/transitioning to/which then 등)으로 36.4%로
# 감소, 3~4개 분해는 2.8%→19.4%. 문장 분해는 캡션의 98.4%가 단문이라 무용.
_CLAUSE_SPLIT = re.compile(
    r"(?:(?<=[.!?;])\s+)"                     # 문장 경계
    r"|(?:,?\s+(?=(?:and\s+)?then\b))"        # ", then" / " and then"
    r"|(?:,?\s+(?=after\s+that\b))"           # ", after that" (전진 관용구)
    r"|(?:,?\s+(?=followed\s+by\b))"          # ", followed by X"
    r"|(?:,?\s+(?=before\s+\w+ing\b))"        # "... before transitioning/being ..."
    # "before transitioning"과의 이중 발화 방지 — before 뒤의 transitioning은 위에서 이미 잘림
    r"|(?:(?<!before)(?<!Before),?\s+(?=transitioning\s+to\b))"
    r"|(?:,?\s+(?=which\s+then\b))"           # "..., which then X"
    r"|(?:,?\s+(?=while\s+in\s+the\s+next\s+scene\b))"
    # finally/next/later는 절두 삽입어 형태만 (", finally," / "and finally") —
    # "the next step" 같은 형용사 용법 오발 방지
    r"|(?:,?\s+(?=and\s+(?:finally|next|later)\b))"
    r"|(?:,\s+(?=(?:finally|next|later|afterwards?)\s*,))"
    r"|(?:\s*(?=→))",                         # 화살표 표기
    re.IGNORECASE,
)
_LEADING_CONNECTIVE = re.compile(
    r"^(?:and\s+then|then|after\s+that|followed\s+by|before|transitioning\s+to|"
    r"which\s+then|while\s+in\s+the\s+next\s+scene|and|next|finally|first|later|"
    r"afterwards?|→)[\s,:]*",
    re.IGNORECASE,
)


def split_events_rule(caption: str, max_events: int = 8) -> list[str]:
    """캡션 → 시간순 이벤트 절 리스트 (규칙 기반).

    보수적 설계: 최소 1개(원문 전체)는 항상 반환 → 파이프라인이 절대 비지 않음.
    """
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(caption) if p and p.strip()]
    events: list[str] = []
    for p in parts:
        p = _LEADING_CONNECTIVE.sub("", p).strip(" .,;")
        if len(p.split()) >= 2:  # 한 단어 조각은 앞 절의 잔여물로 보고 제거
            events.append(p)
    if not events:
        events = [caption.strip(" .")]
    return events[:max_events]


DECOMPOSE_PROMPT = (
    "Break the following storyline into its distinct events in chronological order. "
    "Write one event per line, numbered like '1. ...'. Use at most {max_events} events, "
    "keep each event short, and do not add events that are not in the storyline.\n\n"
    "Storyline: {caption}"
)


def build_decompose_messages(caption: str, max_events: int = 6) -> list[dict]:
    """VLM 텍스트 전용 호출 메시지 (이미지 없음 → 규정상 외부 API 아님)."""
    return [{
        "role": "user",
        "content": [{"type": "text",
                     "text": DECOMPOSE_PROMPT.format(caption=caption, max_events=max_events)}],
    }]


_NUMBERED = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*(.+?)\s*$")


def parse_numbered_events(text: str, caption: str, max_events: int = 8) -> list[str]:
    """모델 출력 → 이벤트 리스트. 어떤 형식이 와도 규칙 기반 폴백으로 안전."""
    events = []
    for line in text.splitlines():
        m = _NUMBERED.match(line)
        if m and len(m.group(1).split()) >= 2:
            events.append(m.group(1).strip(" .,;"))
    if not events:
        return split_events_rule(caption, max_events)
    return events[:max_events]


def load_events_cache(path) -> dict[str, list[str]]:
    """scripts/build_decompose_cache.py 산출 JSONL → {sample_id: events}.

    MatchScorer의 events_fn 주입이나 재스코어 노트 구성 시 이 dict를 참조하고,
    캐시 미스는 split_events_rule 폴백을 쓸 것 (파이프라인이 절대 비지 않도록).
    """
    import json
    out: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["id"]] = [str(e) for e in rec["events"]]
    return out
