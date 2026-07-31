# Phase 2 16th → 18th Validation 변경 및 실행 가이드

## 1. 문서 목적

이 문서는 `16th` 이후 `18th`까지 Phase 2 self-labeling 학습과 validation에서
무엇이 바뀌었는지 설명한다. 특히 다음 질문에 답하는 것을 목적으로 한다.

- `16th`와 비교해 학습 loss가 무엇이 바뀌었는가?
- validation 문제가 매 checkpoint마다 같은 문제인가?
- `--validation-episodes`는 현재 무엇을 의미하는가?
- Proposed(best-of-K)와 휴리스틱을 어떤 단위로 비교하는가?
- block size별, Type별, Bay별 결과는 어디에 저장되는가?
- 전체 rank와 방법별 1위 횟수는 어떻게 계산되는가?
- 처음 학습, 이어서 학습, loss 그래프 생성 명령은 무엇인가?

근거로 사용한 Git 상태는 다음과 같다.

| Branch | Commit | 핵심 상태 |
| --- | --- | --- |
| `16th` | `3631772` | 단일-target CE, checkpoint episode에 종속된 기존 validation |
| `17th` | `c9b4a4e` | tie-aware CE, 고정 block size×Type validation, Bay/parent 리포트 |
| `18th` | `bcef84d` | checkpoint별 전체 rank·winner 집계와 누적 추세 그래프 |

---

## 2. 핵심 변경 요약

| 구분 | 16th | 17th | 18th |
| --- | --- | --- | --- |
| CE target | 선택된 index 1개 | 관측상 동일한 휴리스틱 동률 index 집합 | 17th와 동일 |
| Validation 문제 | checkpoint episode가 seed에 포함됨 | 학습 시작 시 고정 grid 생성 | 17th와 동일 |
| 문제 크기 | 별도 크기 grid 없음 | min/max/gap으로 고정 | 17th와 동일 |
| 분포 Type | validation 반복 번호 | 크기마다 동일 의미를 갖는 분포 Type | 17th와 동일 |
| 평가 단위 | 기존 전체 candidate bank | Bay별 평가 후 parent 문제로 결합 | 17th와 동일 |
| Proposed | agent 후보 중 best | Bay별 greedy+sample best-of-K 결합 | 17th와 동일 |
| Best checkpoint | 별도 고정 문제 계약 없음 | parent 문제 동일 가중치로 선택 | 17th와 동일 |
| 그래프 | 최신 flat validation 그래프 | checkpoint/크기/Type/Bay 3단계 | 17th 그래프 + rank 추세 |
| Rank 집계 | 제한적 policy rank | parent 문제별 rank 근거 저장 | 전체/크기/Type winner·평균 rank |

가장 중요한 변화는 다음 세 가지다.

1. 학습이 ID tie-break를 외우지 않도록 tie-aware CE로 변경했다.
2. 모든 checkpoint가 완전히 같은 validation 문제와 sampling seed를 사용한다.
3. Bay별 결과를 합친 상위 문제 기준으로 전체 rank와 best checkpoint를 계산한다.

---

## 3. 16th → 17th: 학습 target 변경

### 3.1 16th의 단일-target CE

`16th`는 teacher가 선택한 action index 하나만 정답으로 사용했다.

```text
target = selected_action_index
loss = -log P(selected_action_index | state)
```

예를 들어 정책이 볼 수 있는 모든 수치 특성이 같은 `WO_A`, `WO_B`가 있고,
LPT 휴리스틱의 마지막 동률 해소가 `job_id` 문자열 순서라면 다음 문제가 생긴다.

- 휴리스틱은 `WO_A`를 선택한다.
- 정책 state에는 `job_id` 문자열이 없다.
- 정책 입장에서는 `WO_A`와 `WO_B`를 구분할 수 없다.
- 그런데 CE는 `WO_A` index 하나에만 확률을 몰도록 요구한다.

이는 정책이 관측할 수 없는 정보를 정답으로 강제하는 학습 불가능 target이다.

### 3.2 17th부터의 tie-aware CE

17th부터 transition은 `selected_action_index`와 함께
`target_action_indices`를 저장한다.

휴리스틱 teacher에 대해서는 다음 조건을 모두 만족하는 후보를 동률 정답으로 본다.

1. `machine_id`, `job_id` 같은 재현성용 identity tie-break를 제외한 휴리스틱
   우선순위 key가 같다.
2. Pointer scoring에 실제로 들어가는 W/O node feature와 projected action feature가
   같다.

정답 집합을 `T`라고 하면 loss는 다음과 같다.

```text
loss = -log Σ P(action=i | state),  i ∈ T
```

예:

```text
동률 target index T = {2, 5}
P(index=2) = 0.30
P(index=5) = 0.20

loss = -log(0.30 + 0.20)
     = -log(0.50)
```

즉 특정 ID 하나를 외우는 대신 동률 후보 전체에 배분된 확률 합을 높인다.

### 3.3 단일 target이 유지되는 경우

다음은 기존처럼 target index 하나만 사용한다.

- agent greedy teacher
- agent sample teacher
- 휴리스틱 우선순위 또는 정책 관측 feature가 실제로 다른 후보
- 동률 후보가 하나뿐인 경우

정책 architecture와 state 차원은 이 변경 때문에 바뀌지 않는다. 변경된 것은
teacher target과 CE 계산 계약이다.

---

## 4. 16th → 17th: 고정 Validation Grid

### 4.1 기존 문제

16th에서는 validation Phase 1 assignment seed와 agent sampling seed에 현재 학습
episode가 포함됐다.

따라서 episode 1,000과 episode 2,000에서 다음이 동시에 바뀔 수 있었다.

- validation 문제
- Phase 1 Bay 배정
- agent sampling 결과

이 상태에서는 성능 차이가 모델 개선 때문인지 문제 변경 때문인지 구분하기 어렵다.

### 4.2 현재 고정 방식

17th부터 validation 문제는 학습 시작 시 한 번 생성한다.

```text
Physical block size grid
×
Distribution Type
=
고정 parent validation 문제
```

각 checkpoint는 동일한 문제 payload, 동일한 Phase 1 seed, 동일한 agent sampling
seed로 다시 평가한다. 학습 episode 번호는 validation seed에 들어가지 않는다.

### 4.3 Block size grid

다음 세 CLI 옵션으로 크기를 지정한다.

```text
--validation-min-blocks
--validation-max-blocks
--validation-block-gap
```

예:

```text
min=10, max=100, gap=10
→ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
→ 총 10개 크기
```

최댓값은 반드시 정확히 포함되어야 한다.

```text
(max - min) % gap == 0
```

조건을 만족하지 않으면 다른 값으로 조용히 보정하지 않고 오류를 발생시킨다.

### 4.4 `--validation-episodes`의 현재 의미

17th부터 `--validation-episodes`는 전체 문제 수가 아니다.

```text
--validation-episodes = 각 block size에서 생성할 Distribution Type 수
```

전체 parent 문제 수는 다음과 같다.

```text
block size 개수 × validation-episodes
```

현재 실험:

```text
block size: 10~100, gap 10 → 10개
Type 수: 5
전체 parent validation 문제: 10 × 5 = 50개
```

### 4.5 Distribution Type 구성

Type은 단순 seed 번호가 아니라 다음 8개 분포 축의 정규화 순위 목표다.

| 분포 축 | 의미 |
| --- | --- |
| `np_wo_ratio` | 전체 W/O 중 NP 비율 |
| `nc_wo_ratio` | 전체 W/O 중 NC 비율 |
| `fn_wo_ratio` | 전체 W/O 중 FN 비율 |
| `fl_wo_ratio` | 전체 W/O 중 FL 비율 |
| `wo_per_block` | 물리 블록당 W/O 수 |
| `cut_per_wo` | W/O당 평균 CUT_LTH |
| `bevel_per_wo` | W/O당 평균 BV_QTY |
| `tact_per_wo` | W/O당 평균 TACT_TIME |

Type별 목표는 Latin-hypercube 방식으로 분산시킨다. 동일 Type 번호는 모든 block
size에서 같은 정규화 분포 방향을 의미한다.

Type 수가 5라면 각 block size에서 다음 수만큼 후보 문제를 먼저 만든다.

```text
max(12, Type 수 × 3) = max(12, 15) = 15개
```

15개 후보 중 Type 목표와 가장 가까운 문제를 중복 없이 선택한다. 생성 수식이나
실적 공동분포를 임의로 변형하는 방식은 아니다.

### 4.6 Validation contract

다음 정보가 checkpoint의 `validation_contract`와
`validation/grid_contract.json`에 저장된다.

- problem ID
- block count
- distribution Type
- generation seed
- 목표·정규화·실제 분포
- 전체 Job payload SHA-256

재개 학습 시 현재 grid와 checkpoint contract가 다르면 fallback 없이 실패한다.
따라서 validation min/max/gap, Type 수, seed 또는 생성 데이터 계약을 바꾸려면 새
output directory에서 다시 시작해야 한다.

16th checkpoint에는 이 contract가 없으므로 17th/18th 학습을 그대로 resume할 수
없다.

---

## 5. Validation의 Bay Subproblem과 Parent Problem

### 5.1 Bay별 평가

각 parent validation 문제는 Phase 1 결과에 따라 다음 Bay subproblem으로 나뉜다.

```text
Bay 22
Bay 23
Bay 24
Bay 25
Bay trans
```

실제 W/O가 없는 Bay는 평가값 0을 만들지 않는다. metadata만 남기고 가짜 그래프나
가짜 score를 생성하지 않는다.

### 5.2 Bay별 candidate bank

각 non-empty Bay에서 다음 후보를 만든다.

```text
agent_greedy 1개
agent_sample_* K개
휴리스틱별 1개
```

현재 실험은 다음과 같다.

```text
K = 64
휴리스틱 = 4개
Bay별 candidate 수 = 1 + 64 + 4 = 69개
```

현재 4개 휴리스틱:

```text
lpt_batch
long_cut_batch
best_fit_lth
long_bevel_batch
```

### 5.3 Proposed(best-of-K)

각 Bay에서 `agent_greedy + agent_sample_1..64` 중 사전식 score가 가장 좋은 agent
후보를 하나 선택한다.

```text
Bay 22 Proposed = Bay 22의 agent 후보 중 best
Bay 23 Proposed = Bay 23의 agent 후보 중 best
...
```

선택된 Bay별 Proposed를 합쳐 parent 문제의 `proposed_best_of_k` 해를 만든다.

각 휴리스틱도 같은 방식으로 모든 non-empty Bay의 동일 휴리스틱 결과를 합쳐 parent
문제 해를 만든다.

따라서 전체 rank는 개별 Bay row의 rank를 합산한 값이 아니라, Bay별 해를 실제로
결합한 parent schedule의 최종 score로 계산한다.

### 5.4 Parent score

`raw` score mode의 사전식 순서는 다음과 같다.

```text
(
    hard_violation_count,
    makespan,
    Bay 내부 CUT_LTH gap 합,
    Bay 내부 W/O 수 gap 합,
    Bay 내부 BV_QTY gap 합,
    Bay 내부 occupancy gap 합
)
```

앞 원소가 다르면 뒤 원소는 비교하지 않는다.

`hard_violation_count`는 CSV와 rank 계산에는 유지하지만 정상 결과에서는 항상 0이어야
한다. hard violation 그래프는 생성하지 않는다.

---

## 6. Best Checkpoint 선택

각 `(block size, Type)` parent 문제를 동일한 가중치 1개 문제로 취급한다.
Bay가 많이 활성화된 문제가 더 큰 가중치를 갖지 않는다.

각 checkpoint에서 Proposed를 최상 휴리스틱과 비교해 다음을 기록한다.

```text
win  = Proposed score < best heuristic score
tie  = Proposed score = best heuristic score
loss = Proposed score > best heuristic score
```

best checkpoint 선택 key는 다음 사전식 tuple이다.

```text
(
    Proposed loss 수,
    -Proposed win 수,
    Proposed 평균 rank
)
```

즉 우선순위는 다음과 같다.

1. loss 수 최소
2. win 수 최대
3. 평균 rank 최소

선택된 checkpoint는 다음 파일에 저장된다.

```text
<output>/phase2_best.pt
<output>/validation/best_checkpoint.json
```

주기 checkpoint와 best checkpoint는 다른 파일이다.

```text
주기 저장:
<output>/checkpoints/phase2_batch_machine_policy_epNNNNN.pt

validation best:
<output>/phase2_best.pt
```

---

## 7. 17th → 18th: Rank 및 Winner 집계

### 7.1 문제별 경쟁 rank

각 parent 문제에서 모든 method의 최종 사전식 score를 비교한다.

rank는 다음 식으로 계산한다.

```text
rank(method) = 1 + 자신보다 엄격히 좋은 score를 가진 method 수
```

따라서 동률은 같은 rank를 받는다.

```text
score 순서: A = B < C
rank:       A=1, B=1, C=3
```

### 7.2 동률 winner 집계

공동 1위 method는 모두 winner count가 1 증가한다. 따라서 동률이 있으면 method별
winner count의 합이 전체 문제 수보다 클 수 있다.

다음 값을 별도로 저장한다.

- 전체 1위 횟수
- 단독 1위 횟수
- 공동 1위 횟수
- method별 평균 rank

### 7.3 집계 scope

checkpoint마다 다음 scope로 집계한다.

```text
overall                1행
block_size             block size별 1행
distribution_type      Type별 1행
```

현재 크기 10개, Type 5개이므로:

```text
1 + 10 + 5 = checkpoint당 validation_rank_summary.csv 16행
```

### 7.4 현재 ep_001000 실제 결과

현재 output:

```text
output/phase2_phase1_wo_first_train_b10_30_val_b10_100_g10_t5_h16_heur4
```

`ep_001000`의 전체 50문제 결과:

| 항목 | 값 |
| --- | ---: |
| Proposed win/tie/loss | 18 / 0 / 32 |
| Proposed 1위 문제 수 | 18 |
| Proposed 평균 rank | 2.68 |
| LPT 1위 | 28 |
| LongCut 1위 | 3 |
| LongBevel 1위 | 1 |
| BestFitLTH 1위 | 0 |

현재 validation checkpoint가 `ep_001000` 하나뿐이므로 root history PNG에도 점이
하나만 있다. `ep_002000`, `ep_003000`이 생성되면 같은 PNG에 선으로 누적된다.

---

## 8. Validation 출력 폴더 구조

```text
<output>/
├─ metrics.csv
├─ subproblem_metrics.csv
├─ phase2_best.pt
├─ checkpoints/
│  └─ phase2_batch_machine_policy_ep01000.pt
└─ validation/
   ├─ grid_contract.json
   ├─ best_checkpoint.json
   ├─ latest_evaluation.json
   ├─ validation_bay_history.csv
   ├─ validation_parent_history.csv
   ├─ validation_candidate_latest.csv
   ├─ validation_rank_history.csv
   ├─ validation_overall_winner_count_history.png
   ├─ validation_proposed_mean_rank_history.png
   ├─ problems/
   │  └─ blocks_010/
   │     └─ type_01/
   │        ├─ jobs.csv
   │        ├─ problem_metadata.json
   │        └─ distribution_profile.json
   └─ evaluations/
      └─ ep_001000/
         ├─ aggregate/
         │  ├─ method_problem_scores.csv
         │  ├─ method_summary_by_block_size.csv
         │  ├─ validation_rank_summary.csv
         │  └─ 목적함수별 block-size boxplot 5개
         └─ blocks_010/
            ├─ by_type/
            │  ├─ parent/
            │  │  ├─ method_summary_by_type.csv
            │  │  └─ 목적함수별 Type 그래프 5개
            │  └─ bay_22 ... bay_trans/
            │     ├─ method_summary_by_type.csv
            │     └─ 목적함수별 Type 그래프 5개
            └─ type_01/
               ├─ evaluation_metadata.json
               ├─ phase1_assignments.csv
               ├─ parent_method_summary.csv
               └─ bay_22 ... bay_trans/
                  ├─ jobs.csv
                  ├─ problem_metadata.json
                  ├─ candidate_summary.csv
                  └─ proposed_solution.csv
```

### 8.1 핵심 CSV

| 파일 | 용도 |
| --- | --- |
| `validation_bay_history.csv` | 모든 checkpoint의 Bay subproblem 성능 이력 |
| `validation_parent_history.csv` | 모든 checkpoint의 parent 문제 Proposed 성능 이력 |
| `validation_candidate_latest.csv` | 최신 checkpoint의 Bay별 전체 후보만 보존 |
| `method_problem_scores.csv` | 해당 checkpoint의 parent 문제×method score와 rank |
| `validation_rank_summary.csv` | 해당 checkpoint의 overall/크기/Type rank 집계 |
| `validation_rank_history.csv` | 모든 checkpoint의 rank summary 누적 |

`validation_candidate_latest.csv`는 장기 학습 중 파일이 무한히 커지는 것을 막기 위해
최신 checkpoint만 저장한다. 과거 checkpoint의 상세 후보는
`validation/evaluations/ep_NNNNNN/.../candidate_summary.csv`에 보존된다.

### 8.2 생성되는 목적함수 그래프

다음 5개 지표를 그린다.

1. Makespan
2. Bay 내부 CUT_LTH gap 합
3. Bay 내부 W/O 수 gap 합
4. Bay 내부 BV_QTY gap 합
5. Bay 내부 occupancy gap 합

그래프 계층:

- 정확한 `block size × Type × Bay` raw CSV
- block size 고정 상태의 Type별 parent/Bay 그래프
- 전체 block size별 Type 분포 boxplot
- checkpoint별 전체 winner count 추세
- checkpoint별 Proposed 평균 rank 추세

---

## 9. CLI 옵션 의미

| 옵션 | 의미 |
| --- | --- |
| `--min-blocks`, `--max-blocks` | 학습 episode의 물리 블록 수 범위 |
| `--rollout-samples` | 학습 Bay별 stochastic agent 후보 수 |
| `--rollout-samples_validation` | validation Bay별 stochastic agent 후보 수 |
| `--heuristic-algorithms` | 학습·validation에 공통 사용되는 휴리스틱 bank |
| `--validation-every` | 몇 학습 episode마다 고정 grid를 재평가할지 |
| `--validation-min-blocks` | validation grid 최소 물리 블록 수 |
| `--validation-max-blocks` | validation grid 최대 물리 블록 수 |
| `--validation-block-gap` | validation block size 간격 |
| `--validation-episodes` | 각 block size의 Distribution Type 수 |
| `--checkpoint-every` | 주기 checkpoint 저장 간격 |
| `--resume-checkpoint latest` | output/checkpoints의 최신 주기 checkpoint에서 재개 |
| `--action-pool-limit None` | 모든 feasible W/O를 action 후보로 사용 |
| `--max-wo-count 3` | batch당 W/O 최대 3개 |
| `--max-length-sum 55000` | batch W/O LTH 합 최대 55,000 |

`--validation-every`와 `--checkpoint-every`는 서로 다른 기능이다.

```text
validation-every:
고정 문제 평가 + rank/graph + phase2_best.pt 판단

checkpoint-every:
학습 재개용 주기 checkpoint 저장
```

---

## 10. 현재 실험과 동일한 처음 학습 명령

아래 명령은 현재 output 이름과 checkpoint 계약에 맞춘 권장 명령이다.

```cmd
python main.py phase2-train-batch-machine-self-labeling --config config_mixed.yaml --phase1-heuristic wo_first_balanced --episodes 20000 --min-blocks 10 --max-blocks 30 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms lpt_batch,long_cut_batch,best_fit_lth,long_bevel_batch --phase2-score-mode raw --hidden-dim 16 --lr 0.001 --action-pool-limit None --max-wo-count 3 --max-length-sum 55000 --checkpoint-every 1000 --validation-every 1000 --validation-min-blocks 10 --validation-max-blocks 100 --validation-block-gap 10 --validation-episodes 5 --seed 0 --device cuda --output-dir output/phase2_phase1_wo_first_train_b10_30_val_b10_100_g10_t5_h16_heur4
```

현재 `ep_001000` checkpoint에서 직접 확인되는 계약:

```text
hidden_dim=16
heuristics=lpt_batch,long_cut_batch,best_fit_lth,long_bevel_batch
train rollout=64
validation rollout=64
validation_every=1000
validation Type 수=5
validation parent 문제 수=50
max_wo_count=3
max_length_sum=55000
action_pool_limit=None
score_mode=raw
```

`min-blocks=10`, `max-blocks=30`은 output 이름의 `train_b10_30` 계약에 맞춘 값이다.
이 두 값과 `checkpoint-every`는 현재 checkpoint payload에 별도 저장되지 않으므로,
실험 이름과 실행 기록을 함께 유지해야 한다.

---

## 11. 이어서 학습 명령

처음 명령의 RunSpec과 validation grid 옵션을 그대로 유지하고
`--resume-checkpoint latest`만 추가한다.

```cmd
python main.py phase2-train-batch-machine-self-labeling --config config_mixed.yaml --phase1-heuristic wo_first_balanced --episodes 20000 --min-blocks 10 --max-blocks 30 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms lpt_batch,long_cut_batch,best_fit_lth,long_bevel_batch --phase2-score-mode raw --hidden-dim 16 --lr 0.001 --action-pool-limit None --max-wo-count 3 --max-length-sum 55000 --checkpoint-every 1000 --validation-every 1000 --validation-min-blocks 10 --validation-max-blocks 100 --validation-block-gap 10 --validation-episodes 5 --seed 0 --device cuda --resume-checkpoint latest --output-dir output/phase2_phase1_wo_first_train_b10_30_val_b10_100_g10_t5_h16_heur4
```

주의:

- `--episodes 20000`은 추가 20,000회가 아니라 최종 도달 episode다.
- 현재 output에서 latest가 episode 1,000이면 1,001부터 20,000까지 진행한다.
- heuristic bank 순서와 항목도 RunSpec 일부이므로 그대로 유지한다.
- validation grid가 다르면 `validation_contract_mismatch`로 실패한다.
- 16th checkpoint는 validation contract가 없으므로 이 명령으로 재개할 수 없다.

---

## 12. Loss 그래프 명령

```cmd
python scripts/plot_phase1_loss_only.py output/phase2_phase1_wo_first_train_b10_30_val_b10_100_g10_t5_h16_heur4 --window 100
```

생성 파일:

```text
loss_curve.png
loss_curve_normalized.png
loss_only_status.json
```

validation 그래프는 별도 plot 명령이 필요하지 않다.
`episode % validation_every == 0`일 때 학습 명령 안에서 자동 생성·갱신된다.

현재 loss 스크립트는 파일명과 차트 제목에 `Phase 1` 명칭이 남아 있지만,
입력 폴더의 `metrics.csv`에서 `episode`와 `loss` 열만 읽는다. 따라서 Phase 2
output에도 같은 명령을 사용할 수 있다.

---

## 13. Rank 결과 빠른 확인 명령

전체 checkpoint의 overall rank 행만 출력:

```cmd
python -c "import csv; p=r'output\phase2_phase1_wo_first_train_b10_30_val_b10_100_g10_t5_h16_heur4\validation\validation_rank_history.csv'; f=open(p,encoding='utf-8-sig',newline=''); rows=list(csv.DictReader(f)); [print(r) for r in rows if r['scope']=='overall']"
```

최신 checkpoint 경로 확인:

```cmd
type output\phase2_phase1_wo_first_train_b10_30_val_b10_100_g10_t5_h16_heur4\validation\latest_evaluation.json
```

best checkpoint 선택 근거 확인:

```cmd
type output\phase2_phase1_wo_first_train_b10_30_val_b10_100_g10_t5_h16_heur4\validation\best_checkpoint.json
```

---

## 14. 코드 위치

| 파일 | 역할 |
| --- | --- |
| `Phase2/validation_grid.py` | 고정 block size×Type 문제 생성과 SHA-256 contract |
| `Phase2/merged.py` | tie-aware CE, Bay/parent 평가, best checkpoint, rank·그래프 |
| `main.py` | validation grid CLI 옵션과 종료 경로 출력 |
| `tests/test_phase2_validation_grid.py` | grid, contract, 출력 tree, rank history 검증 |
| `tests/test_phase2_wo_only_policy.py` | tie-aware target과 확률합 CE 검증 |
| `Phase1_Phase2_CE_Target_5Step_완전예제.md` | Phase 1/2 CE target을 5-step 예제로 설명 |

---

## 15. 검증 명령과 완료 기준

Phase 2 validation 집중 시험:

```bash
python3 -m unittest tests.test_phase2_validation_grid
python3 -m unittest tests.test_phase2_wo_only_policy
```

전체 시험:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

문법 검사:

```bash
python3 -X pycache_prefix=/tmp/pmsp_pycache -m py_compile main.py Agent/*.py Phase1/*.py Phase2/*.py Environment/*.py Environment/constraints/*.py Utils/data/*.py Utils/learning/*.py Utils/phase1/*.py Utils/reporting/*.py Train/network/*.py scripts/*.py
```

Git whitespace 검사:

```bash
git diff --check
```

18th 기준 확인 결과:

```text
Phase 2 validation grid tests: 5 tests OK
전체 unit tests: 204 tests OK
Python compile: passed
git diff --check: passed
```

---

## 16. 결과 해석 시 주의사항

1. loss 감소와 validation rank 개선은 같은 지표가 아니다.
   - loss는 teacher action 집합 모방 정도다.
   - rank는 고정 문제에서 최종 schedule score를 비교한 결과다.

2. Proposed는 greedy 단독이 아니다.
   - 각 Bay에서 greedy와 64개 sample 중 가장 좋은 agent 해를 사용한다.

3. 전체 rank는 Bay rank 평균이 아니다.
   - Bay별 해를 결합해 parent schedule을 만든 뒤 다시 score를 계산한다.

4. Type 5개는 validation 문제 5개를 뜻하지 않는다.
   - 크기 10개라면 전체 parent 문제는 50개다.

5. winner count 합이 전체 문제 수보다 클 수 있다.
   - 공동 1위 method를 모두 winner로 집계하기 때문이다.

6. root history 그래프의 X축은 모든 학습 episode가 아니다.
   - `--validation-every`마다 평가된 checkpoint episode만 표시한다.

7. validation grid를 바꾸면서 같은 output을 resume하지 않는다.
   - 문제 contract가 달라지므로 새 output directory가 필요하다.
