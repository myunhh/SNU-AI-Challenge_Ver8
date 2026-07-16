# CLAUDE.md — Ver8

셔플된 비디오 프레임 4장 + 캡션 → 시간순 재배열(4!=24). SNU AI Challenge 2026.

**이 저장소 = Ver7을 베이스로 시작한 학습 단 개선 트랙 — 현재 대회 챔피언(LB 0.90401).**
추론 단 개선(재질의·디코딩·TTA 확장·서빙 정밀도)이 Ver1~7에서 전부 실측 소진된 뒤, 남은 합법
레버인 학습 단(가중치를 바꾸는 것)에 집중. **DPO(인접 스와프 hard-negative)**를 Ver4 ckpt1600
base 위에 A100에서 실행해 checkpoint-200이 LB **0.90401**(Ver4 ckpt1600의 0.90226 대비
+0.175pp) — 홀드아웃 게이트 없이 LB로 직접 판정된 결과라 재현성·과적합 검증 미완. 상세 경위·
게이트 판정 이력은 `../PROJECT_SUMMARY.md` §2 Ver8 항목, 공통 규정·함정은 `../CLAUDE.md` 참고.

## 진입점

```bash
pytest tests/ -q
python -m snuai.train.train_dpo --adapter runs/sft32b_v4/adapter_final --no-hard-negative \
    --max-steps 1000 --save-steps 200 --out runs/dpo_v8
python -m snuai.infer.predict --csv data/test.csv --image-dir data/test \
    --strategy score24 --adapter <dpo checkpoint> --tta 3 --out runs/test_v8

# 멀티GPU(DDP, 2026-07-16 추가) — device_map을 PartialState().process_index로 고정해
# torchrun/accelerate 멀티프로세스에서 각 rank가 자기 GPU에만 올라감 (단일프로세스는 기존과 동일)
torchrun --nproc_per_node=2 -m snuai.train.train_dpo --adapter runs/sft32b_v4/adapter_final \
    --max-steps 2000 --save-steps 200 --out runs/dpo_v8_ddp
```

## 저장소 구조

- `src/snuai/train/train_dpo.py` — 단일토큰 DPO(TRL 미사용, 참조모델은 `disable_adapter()`로 대체)
- `src/snuai/train/dpo_pairs.py` — 인접 스와프 hard-negative 선호쌍 생성
- `scripts/diag_swap_submission.py` — 고마진 K건 인접 스와프로 EM/쌍순서 채점방식 가설 구분용 진단 제출 생성기(실제 Kaggle 제출은 미실행)

## 다음 액션

- checkpoint-400/600/800/1000이 A100에서 저장 중 — ckpt200보다 나은지 홀드아웃으로 먼저 비교 후
  LB 슬롯 사용(`../TODO.md` 참고).

## 이 버전 고유 함정

- DPO 참조 로그확률은 별도 참조모델 없이 `model.disable_adapter()`로 얻음(reference-free, 메모리 절약).
- `--hard-negative`(기본 on) 사전 스코어링이 로컬 4090에서 8,600건 순회에 80분+ 걸려 사실상 못 돎 —
  A100 또는 `--no-hard-negative`(스코어링 스킵, 무작위 rejected) 필요.
- **DDP(멀티GPU)에서 hard-negative 스코어링은 rank0만 수행**(그 비싼 forward pass를 world_size배
  중복하지 않으려고) — `dpo_pairs.build_dpo_records`의 `rejected_ranks_cache` 인자로 rank0가 만든
  `runs/.../hard_negative_cache.json`을 다른 rank들이 읽어 동일 records를 재구성(augment_sample이
  seed 고정 rng만 써서 rank 간 결정적으로 동일함을 이용). **`--include-random-rejected`는 이 DDP
  캐시 경로와 미지원**(rng 스트림이 rank마다 갈라질 위험) — 켜려면 단일 프로세스로 돌릴 것.
  `--ddp-find-unused-parameters`는 gradient-checkpointing+PEFT+DDP 조합에서 크래시 시에만 켤 것.
