"""SNU AI Challenge 2026 — 캡션 기반 4프레임 시간순 재배열 파이프라인.

패키지 구조:
  perm        순열 대수 (인코딩 규약의 단일 진실 공급원)
  submission  Kaggle Answer 포맷 입출력
  data/       전처리 (학습·추론 공용/분리)
  train/      학습 (QLoRA SFT, DPO, 보조 태스크)
  infer/      추론 (score24, TTA, 캐스케이드, FSM, decompose-and-match)
"""

__version__ = "0.1.0"
