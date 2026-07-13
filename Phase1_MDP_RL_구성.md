# Phase 1 MDP/RL 구성 정리

작성일: 2026-06-24

이 문서는 현재 과제의 Phase 1 학습 구조를 설명한다. 핵심은 중간발표 범위를 **블록 단위 Bay 배정**으로 고정하고, Phase 2의 W/O batch/설비 스케줄링은 뒤 단계로 분리하는 것이다.

---

## 1. 현재 문제 분리

### Phase 1: 블록 단위 Bay 배정

입력은 블록이다. 같은 블록의 W/O는 모두 같은 절단 Bay로 간다.

결정:

```text
블록 선택 -> Bay 선택
```

목적:

1. 강재수량 평준화
2. 절단장 평준화
3. 베벨수량 평준화
4. 절단장 1000 초과 블록의 Bay 24 배정 최소화

현업 답변 기준으로 4번은 hard 제약이 아니라 목적함수/선호다. 즉 Bay 24를 절대 금지하지 않고, 가능하면 Bay 22/23으로 보내는 방향이다.

### Phase 2: Bay 내부 설비/W/O batch scheduling

Phase 1 결과를 받아 Bay 안에서 설비와 W/O batch를 결정한다.

결정:

```text
W/O batch 선택 -> 설비 선택 -> batch 투입 순서 결정
```

Phase 2에서는 W/O 1~3개, 길이 합 55,000 이하, batch는 같이 들어오고 같이 빠지는 제약을 사용한다.

---

## 2. Phase 1 MDP 정의

### State

현재 상태는 아래 3개 묶음이다.

- 남아 있는 블록 후보의 feature
- 현재 Bay별 누적 부하
- 전체 진행률과 부하 gap

블록 feature:

```text
steel_quantity_sum
cut_length_sum
bevel_quantity_sum
wo_count
long_cut_over_1000
allowed_bay_count
remaining_flag
assigned_flag
```

Bay feature:

```text
bay_id_numeric
steel_quantity_sum
cut_length_sum
bevel_quantity_sum
wo_count
block_count
long_cut_bay24_count
candidate_allowed_flag
```

환경 feature:

```text
progress_ratio
remaining_block_ratio
steel_quantity_gap
cut_length_gap
bevel_quantity_gap
long_cut_bay24_count
assigned_block_count
```

### Action

Phase 1은 계층형 action이다.

```text
1단계: SELECT_BLOCK
2단계: SELECT_BAY
```

예:

```text
step 0: SELECT_BLOCK 후보 = [P1::A, P1::B, P1::C], 선택 = P1::A
step 1: SELECT_BAY 후보 = [22, 23, 24], 선택 = 22
step 2: SELECT_BLOCK 후보 = [P1::B, P1::C], 선택 = P1::C
step 3: SELECT_BAY 후보 = [22, 23, 24], 선택 = 23
```

### Transition

`SELECT_BLOCK`만으로는 부하가 바뀌지 않는다.

`SELECT_BAY`가 끝나면 선택된 블록의 강재수량, 절단장, 베벨수량, W/O 수, 블록 수가 해당 Bay load에 더해진다.

### Reward 또는 Label

현재 1차 구현은 reward 학습이 아니라 self-labeling/imitation이다.

즉, 이미 검증한 휴리스틱 `long_cut_preferred_balanced` 결과를 teacher label로 사용한다.

```text
입력 state + 후보 action들 -> 휴리스틱이 선택한 action index를 정답 label로 학습
```

RL reward는 이후 제약과 선호가 더 늘어난 뒤 붙이는 것이 맞다.

---

## 3. 네트워크 구조

졸업 코드의 핵심 구조는 GNN이라기보다 pointer-style 후보 scoring 구조다.

현재 Phase 1에 맞춘 구조:

```text
Block feature encoder
Bay feature encoder
Environment encoder
Mean pooling/global context
Pointer scoring
```

두 개의 scoring 함수를 둔다.

```python
score_blocks(block_features, env_features)
score_bays(bay_features, env_features, selected_block_features)
```

구현 파일:

```text
Phase1/pointer_policy.py
Phase1/imitation.py
```

학습 루프는 `phase1_action_table.jsonl`을 읽어서 `selected_action_index`에 대해 cross-entropy를 건다. 입력 feature는 trainer 내부에서 mean/std 정규화하고, 정규화 파라미터는 checkpoint metadata에 저장한다.

---

## 4. 현재 구현된 산출물

Phase 1 MDP/self-label package 생성 명령:

```bash
python3 main.py phase1-mdp-trace \
  --config config_np_100.yaml \
  --algorithm long_cut_preferred_balanced \
  --bay-ids 22,23,24 \
  --output-dir output/phase1_mdp_trace_np_100_long_cut_preferred
```

산출물:

```text
phase1_mdp_trace.csv
phase1_action_table.jsonl
phase1_mdp_manifest.json
phase1_block_bay_plan.json
phase1_block_assignments.csv
phase1_bay_loads.csv
```

Phase 1 imitation 학습 명령:

```bash
python3 main.py phase1-train-imitation \
  --action-table output/phase1_mdp_trace_np_100_long_cut_preferred/phase1_action_table.jsonl \
  --output-dir output/phase1_imitation_np_100_long_cut_preferred_norm_overfit \
  --epochs 300 \
  --lr 0.001 \
  --hidden-dim 128 \
  --seed 0
```

학습 산출물:

```text
phase1_pointer.pt
metrics.csv
summary.json
```

100건 실행 결과:

```text
job_count = 100
block_count = 18
trace_row_count = 36
decision_count_per_block = 2
algorithm = long_cut_preferred_balanced
steel_quantity_gap = 1
cut_length_gap = 172.763
bevel_quantity_gap = 8
long_cut_bay24_count = 0
```

100건 trace overfit smoke 결과:

```text
example_count = 36
epochs = 300
hidden_dim = 128
final_loss = 0.050265
final_accuracy = 1.000000
```

---

## 5. 왜 multi-agent가 아닌가

현재는 하나의 agent가 계층형으로 결정하는 것이 맞다.

```text
SELECT_BLOCK -> SELECT_BAY
```

두 agent로 나누면 block agent와 Bay agent의 목적 충돌, credit assignment, 학습 불안정이 생긴다. 지금 문제 규모에서는 이득보다 디버깅 비용이 크다.

나중에 Phase 2까지 붙으면 아래처럼 분리할 수 있다.

```text
Phase 1 agent: block -> Bay
Phase 2 agent: W/O batch -> machine -> sequence
```

하지만 현재 중간발표와 Phase 1 학습에는 단일 계층형 agent가 가장 단순하고 검증 가능하다.

---

## 6. 다음 구현 순서

1. 학습된 policy를 같은 Phase 1 환경에서 inference하는 runner 작성
2. inference 결과를 실적/휴리스틱과 8개 후보일 기준으로 비교
3. synthetic block data로 train/test split을 만들고 일반화 성능 확인
4. Phase 2로 넘어가 W/O batch와 설비 scheduling을 붙임

현재 완료된 것은 self-label trace 생성, pointer network, imitation trainer, 100건 overfit smoke다. 아직 완료되지 않은 것은 학습 policy inference와 8개 후보일 비교다.

---

## 7. 실적 기반 합성 episode 학습

Phase 1 학습용으로는 W/O가 아니라 블록 단위 episode를 만든다.

생성 명령:

```bash
python3 main.py phase1-build-episode-dataset \
  --block-xlsx input/절단03~04_NP물량_마스킹_블록_수정_260618.xlsx \
  --gyel NP \
  --episode-count 40 \
  --min-blocks 12 \
  --max-blocks 80 \
  --train-ratio 0.8 \
  --bay-ids 22,23,24 \
  --algorithm long_cut_preferred_balanced \
  --seed 2026 \
  --noise-ratio 0.03 \
  --output-dir output/phase1_episode_dataset_40x12_80
```

생성 구조:

```text
실적 블록 데이터
-> NP 필터
-> LTH/THK/CUT_LTH/MARK_LTH/STL_QTY/BV_QTY 검증
-> episode마다 블록 수를 12~80개 사이에서 샘플링
-> 실적 분포 bootstrap + 3% noise
-> Phase 1 휴리스틱으로 teacher label 생성
-> train/test action table 분리
```

생성 결과:

```text
episode_count = 40
train_episode_count = 32
test_episode_count = 8
episode block count range = 12~80
total_action_count = 3700
train_action_count = 2968
test_action_count = 732
```

학습 명령:

```bash
python3 main.py phase1-train-imitation \
  --action-table output/phase1_episode_dataset_40x12_80/phase1_train_action_table.jsonl \
  --eval-action-table output/phase1_episode_dataset_40x12_80/phase1_test_action_table.jsonl \
  --output-dir output/phase1_imitation_episode_40x12_80_epoch100 \
  --epochs 100 \
  --lr 0.001 \
  --hidden-dim 128 \
  --seed 0
```

학습 결과:

```text
train examples = 2968
eval examples = 732
train accuracy = 0.800876
eval accuracy = 0.760929
train SELECT_BLOCK accuracy = 0.911051
eval SELECT_BLOCK accuracy = 0.879781
train SELECT_BAY accuracy = 0.690701
eval SELECT_BAY accuracy = 0.642077
```

해석:

```text
학습은 된다.
블록 선택은 비교적 잘 학습된다.
Bay 선택이 현재 병목이다.
```

Bay 선택이 어려운 이유는 `long_cut_preferred_balanced`가 beam/local-search 기반 최종 배정을 teacher label로 주기 때문이다. 즉 어떤 Bay를 골랐는지가 현재 순간의 Bay load만으로 완전히 설명되지 않을 수 있다. 다음 개선은 학습 속도 mini-batch화와 Bay 선택 feature 보강이다.
