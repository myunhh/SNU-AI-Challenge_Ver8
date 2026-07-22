# Ver8 hard-negative DPO(reversal-rejected) — A100 실행 지시서 (2026-07-22)

**대상**: A100 렌탈 박스에서 새로 시작하는 Claude Code(또는 Fable) 세션. 이 문서만 읽고
처음부터 작업 가능하도록 자기완결적으로 작성했다. 이전 대화 맥락은 없다고 가정할 것.
**예산: A100 약 5시간.** 마감 2026-07-24 23:59.

## 0. 이게 뭔지

이 대회(SNU AI Challenge, 셔플된 프레임 4장 시간순 재배열)의 **현재 실LB 챔피언은
Ver8 DPO checkpoint-600 + 균형 8뷰 TTA(BALANCED8), LB 0.93019**다. 이 작업은 서빙
쪽(TTA/캐스케이드) 실험이 아니라 **챔피언 가중치 자체를 개선**하려는 시도다 — 최근
서빙 축(TTA24, cascade)은 둘 다 실LB에서 손실로 확인돼 종결됐고, 아직 안 해본 게
학습 쪽 레버 하나: **DPO 선호쌍에 "완전역전(d=6)" 네거티브를 추가하는 것**이다.

**배경**: ckpt600(2026-07-18 채택, 인접스와프 hard-negative DPO)의 홀드아웃 오답을
분석했더니 "확신에 찬 방향 역전"(모델이 margin 0.98+ 확신으로 정답을 완전히 거꾸로
내놓는 경우, d≥4가 64건, 완전역전 d=6이 12건)이 있었다. 인접스와프(d=1) 네거티브만으론
이 오류 모드에 그래디언트가 실리지 않는다 — `dpo_pairs.py`에 이미 이 문제를 겨냥한
`--reversal-rejected` 플래그가 **구현까지 끝나있는데 실제 GPU 학습만 안 돌려본 상태**로
남아있었다. 그걸 지금 돌린다.

**안전 하한**: 이 실험이 뭘 내놓든 실LB 챔피언(0.93019)은 그대로 유지된다 — 여기서
나온 체크포인트는 검증 게이트를 통과해야만 챔피언 승격 후보가 된다(아래 §3).

## 1. 착수 전 준비물 체크 (반드시 `ls`로 실제 확인, 있다고 가정하지 말 것)

이 박스가 기존에 이 저장소를 쓰던 박스가 아니면(Ver13용으로 별도 렌탈됐던 박스라면
특히) 아래가 없을 수 있다. 없는 게 있으면 **임기응변으로 대체하지 말고** 사용자(dev box)
에게 보고할 것 — 특히 어댑터를 다른 체크포인트로 대체하면 완전히 다른(비교 불가능한)
실험이 된다.

1. **이 저장소가 `myunhh/SNU-AI-Challenge_Ver8`의 clone인지** — 아니면
   `git clone https://github.com/myunhh/SNU-AI-Challenge_Ver8` 부터.
2. **`runs/DPO-checkpoint-600/adapter_model.safetensors` + `adapter_config.json`**
   (챔피언 자체의 DPO 어댑터 — 이번 실험의 warm-start 베이스). **GitHub엔 100MB
   하드리밋으로 못 올라가 있으니 git clone만으로는 안 온다** — dev box에서
   scp로 받거나, GitHub Release 에셋으로 올라와 있으면 그걸로. 없으면 학습을
   시작하지 말고 dev box에 요청할 것.
3. **`runs/cal_dpo_ckpt600/progress.jsonl`** (ckpt600의 기존 홀드아웃 평가 결과 —
   §3 비교의 기준선A). 이건 작은 텍스트 파일이라 scp 부담 없음. **없으면** §3의
   "3a. 기준선 재생성" 단계부터 해야 함(추가 GPU 시간 소요, 아래 §4 시간배분에 반영).
   (참고: dev box에는 이 파일이 이미 있음 — `Ver8/runs/cal_dpo_ckpt600/progress.jsonl`.)
4. **`data/train.csv` + `data/train/` 이미지** — 기존 트랙에서 쓰던 것 그대로.
5. **베이스 모델**: `train_dpo.py` 기본값은 로컬 경로
   `/home/yhmin/model/hub/Qwen3-VL-32B-Instruct-bnb-4bit`다. 이 경로가 이 박스에
   없으면 `--model-id unsloth/Qwen3-VL-32B-Instruct-bnb-4bit`(HF hub id)로 바꿔서
   실행 — 최초 1회 다운로드(~19GB)가 들어가니 §4 시간 배분에 반영할 것.
6. `export PYTHONPATH=$PWD/src`, `export LD_LIBRARY_PATH=$HOME/anaconda3/envs/py3_11/lib`
   (GLIBCXX 함정 — pandas import 에러 방지, 전 버전 공통).

## 2. 학습 실행

```bash
python -m snuai.train.train_dpo \
  --adapter runs/DPO-checkpoint-600 \
  --reversal-rejected \
  --max-steps 600 --save-steps 100 \
  --out runs/dpo32b_v8_reversal
```

- **ckpt600에서 warm-start** — 처음부터 1500스텝 다시 하지 않는다. 이미 잘 학습된
  모델에 새 네거티브 신호 하나만 더하는 것이므로 짧게(600스텝) 잡았다.
- `--hard-negative`는 기본 True라 기존 인접스왑 네거티브도 그대로 유지됨. `--reversal-rejected`
  **이것 하나만 새로 켜는 것** — beta/lr/grad-accum 등 나머지는 **절대 바꾸지 말 것**
  (변수 하나만 격리해야 결과 해석이 됨).
- 100스텝마다 체크포인트 저장 → 최악의 경우 시간 안에 다 못 끝나도 여러 지점이 평가
  가능하게 남는다.
- **초반 10~20스텝에서 loss가 정상 스케일인지(발산·NaN 없는지) 눈으로 확인하고 자리를
  떠날 것.** ckpt600에서 시작하므로 loss는 처음부터 낮아야 정상(무전이면 즉시 중단하고
  보고 — 임기응변으로 재시도하지 말 것).
- **시간 재projection**: 50~100스텝 지난 시점에 스텝당 소요시간을 실측해서 600스텝
  전체가 시간 예산(§4) 안에 들어오는지 다시 계산할 것. 안 들어오면 그 시점에서 멈추고
  이미 저장된 체크포인트로 §3 진행(끝까지 억지로 밀어붙이지 말 것).

## 3. 평가 — 기존 도구 재사용, 새로 짜지 말 것

**3a. 기준선(ckpt600) 홀드아웃 — `runs/cal_dpo_ckpt600/progress.jsonl`이 이미 있으면 생략**:
```bash
python -m snuai.infer.predict --csv data/train.csv --image-dir data/train \
    --holdout-val --eval --strategy score24 --tta 3 \
    --model-id <위 §1.5에서 확정한 경로/id> \
    --adapter runs/DPO-checkpoint-600 \
    --out runs/cal_dpo_ckpt600
```

**3b. 새 체크포인트마다 홀드아웃 평가** (저장된 것 전부 — 예: checkpoint-100/200/.../600,
있으면 adapter_final도):
```bash
python -m snuai.infer.predict --csv data/train.csv --image-dir data/train \
    --holdout-val --eval --strategy score24 --tta 3 \
    --model-id <동일> \
    --adapter runs/dpo32b_v8_reversal/checkpoint-<N> \
    --out runs/cal_reversal_ckpt<N>
```

**3c. 채택 게이트 — `scripts/ab_gate.py`(이 프로젝트의 기존 홀드아웃 채택 게이트, 그대로
재사용)로 ckpt600 대비 비교**:
```bash
python scripts/ab_gate.py runs/cal_dpo_ckpt600 runs/cal_reversal_ckpt<N> \
    --name reversal_ckpt<N> --out runs/adoption_reversal_ckpt<N>.json
```
`adopt: true`(Δem ≥ +2%p **그리고** 95% CI 하한 > 0)가 나온 체크포인트만 "성공" —
시간이 부족하면 **가장 많이 학습된(스텝 수가 큰) 체크포인트부터** 평가할 것(하나라도
실측 비교점을 확보하는 게 우선).

## 4. 시간 배분 (총 5시간 기준, 목표치일 뿐 §2의 재projection이 우선)

| 단계 | 목표 시간 |
|---|---|
| §1 준비물 확인·(필요시) 전송/다운로드 | ~30분 이내 목표, 넘어가면 사용자에게 보고 |
| §2 학습(최대 600스텝) | ~3~3.5시간 |
| §3a 기준선 재생성(필요한 경우만) | ~20~30분 |
| §3b 후보 체크포인트 홀드아웃 평가(1~3개) | ~30~45분 |
| §3c 게이트 비교 + 정리 | ~10분 |

## 5. 금지사항

- **LB 제출 금지** — 결과가 아무리 좋아 보여도 사용자 확인 없이 제출하지 말 것(오늘
  하루 슬롯은 이미 상당수 소진, 4일 2회 한도).
- **다른 버전 저장소(Ver11/Ver12/Ver13 등) 건드리지 말 것.**
- **§2에 명시한 것 이외의 하이퍼파라미터 변경 금지** — beta/lr/grad-accum/hard-negative
  는 손대지 말 것.
- **§3의 ab_gate.py 비교를 건너뛰고 loss 곡선이나 in-sample 인상만으로 "성공"을
  선언하지 말 것.** 이 프로젝트에서 이번 주에만 이미 두 번 당했다 — ① Ver11에서
  in-sample 1등 체크포인트가 실제 test에서는 2등이었음, ② cascade가 로컬에서 "근소
  우세"로 읽혔는데 실LB는 −0.873pp 손실이었음. 로컬/in-sample 낙관은 못 믿는다.
- 학습·추론 진행바 유지(프로젝트 공통 규약).

## 6. 보고 요구사항 (dev box에 전달)

① §1 준비물이 이미 있었는지/전송·다운로드가 필요했는지와 소요시간 ② 학습 loss 곡선
요약(발산 없이 정상 하강했는지, 몇 스텝까지 실제로 완료했는지) ③ 체크포인트별
`ab_gate.py` 결과 전부(delta, CI95, adopt 여부) ④ 최종 권고 — 승격 후보 있음/없음,
있으면 어느 체크포인트 ⑤ 총 소요시간 ⑥ 이상 징후 전부.

## 7. 추후 계획 (2026-07-22 갱신)

이번 라운드는 시간 예산상 checkpoint-100(§2 목표 600스텝 중 17%)까지만 돌리고
중단. loss 곡선 자체는 정상(발산·NaN 없음)이었고, §3 정식 게이트 대신 test.csv+
grade.py로 새는 바람에 결론은 못 낸 채로 있음. 나중에 다시 손댈 여유가 생기면:

- **제일 먼저**: 이미 있는 checkpoint-100을 §3 그대로(945 홀드아웃 + `ab_gate.py`
  vs `cal_dpo_ckpt600`)로 채점부터 해볼 것 — 추가 학습 없이도 100스텝만으로 뭔가
  잡히는지 확인 가능, 여기서부터 재개 여부를 판단하는 게 순서.
- **재개 비용 줄이기**: 체크포인트에서 이어 돌리면 `--hard-negative`(기본 on)가
  매번 인접스와프 재스코어링을 새로 하는 게 비용의 대부분으로 보임 — 급하면
  `--no-hard-negative`로 재개해 스코어링을 건너뛰는 방법도 있음(단 인접스와프가
  "가장 헷갈리는 오답" 대신 랜덤이 돼서 변수가 하나 더 섞임 — 결과 해석 시 감안).
- 원래 목표인 600스텝까지 가더라도, ckpt600 위에 얹는 거라 신호가 언제부터 보이기
  시작하는지는 이번 100스텝 구간만으론 아직 알 수 없는 상태 — 더 태워봐야 답이 남음.
