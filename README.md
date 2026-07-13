# pmsp

절단 공정용 `PMSP(Parallel Machine Scheduling Problem)` 기반 스케줄링 프로젝트입니다.

이 저장소는 절단 공장의 실제 데이터와 합성 데이터를 같은 제약·평가 계약으로 검증하는 실행 가능한 스케줄링 환경입니다.

- 병렬 이기종 설비 환경 `reset / step`
- 휴리스틱 실행
- Phase 1 / merged Phase 2 self-labeling 학습 루프
- category / hard / soft / override 제약 구조
- calendar 제약
- 설비별 운영시간 / 계획 정지 / 고장
- 샘플 시나리오 기반 시뮬레이션 / trace / Phase별 평가

확정되지 않은 셋업·계열 선호·후공정 반출 규칙은 임의로 활성화하지 않습니다. 현재 확정된 batch 규칙은 W/O 1~3개, W/O `LTH` 합 55,000 이하, batch 처리시간은 W/O `TACT_TIME`의 최댓값입니다.

---

## 1. 프로젝트 목적

이 프로젝트는 절단 공정을 아래 문제로 정의합니다.

- 여러 절단 설비가 동시에 존재
- 설비마다 가능한 계열 / 두께 / 길이 / 속도가 다름
- 작업마다 후공정 베이와 우선순위가 존재
- 설비 부하 평준화도 중요
- 운영일 / 점심시간 / 설비 고장 같은 캘린더 제약도 중요

즉, 본질적으로는 **비관련 병렬기계 스케줄링 + 설비 가능 제약 + 후공정 연계 제약** 문제입니다.

---

## 2. 현재 구현 수준

현재 상태를 한 문장으로 정리하면 아래와 같습니다.

> Phase 1 Block-Bay 배정과 merged Phase 2 `(W/O batch, Machine)` 스케줄링을 strict RunSpec, 공통 제약 audit, self-labeling, actual 8일 평가로 연결한 상태

현재 구조는 2단계로 정리합니다.

- Phase 1: 같은 블록의 모든 W/O를 하나의 절단 Bay에 배정
- Phase 2: Phase 1 결과를 받아 `SELECT_MACHINE -> SELECT_WO 반복 -> batch 자동 close`로 batch와 설비를 함께 결정

Phase 2의 1순위 목적은 전체 `makespan` 최소화입니다. 2순위 이후에 Bay 내부 설비별 W/O 수, 절단장, 베벨수량, 점유시간 부하평준화를 봅니다. 순차 선택이 이후의 batch 구성과 machine 배정 가능성을 바꾸므로 MDP/RL로 정의합니다. 다만 setup time이 꺼진 현재 조건에서는 최종 batch 구성과 machine 배정이 같을 때 단순한 batch 투입 순서 교환만으로 목적값이 달라지지는 않습니다.

상세 진행 상태와 다음 작업은 [현재과제_진행체크리스트.md](현재과제_진행체크리스트.md)를 기준으로 봅니다.

Phase 1 학습/MDP 구성은 [Phase1_MDP_RL_구성.md](Phase1_MDP_RL_구성.md)에 정리되어 있습니다.

### 현재 실행 명령

Phase 1 단독 학습:

```bash
python main.py phase1-train-pair-self-labeling --config config_np_100.yaml --bay-ids 22,23,24 --block-xlsx "input/절단03~04_NP물량_마스킹_블록_수정_260618.xlsx" --gyel NP --episodes 20000 --min-blocks 12 --max-blocks 80 --noise-ratio 0.03 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms ppb_lcp6 --hidden-dim 128 --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --actual-validation-candidate-xlsx "착수일 후보 블록.xlsx" --output-dir output/phase1_pair_train
```

Merged Phase 2 학습:

```bash
python main.py phase2-train-batch-machine-self-labeling --config config_np_100.yaml --phase1-heuristic bevel_first_balanced --phase1-bay-ids 22,23,24 --synthetic-source report_formula --gyel NP --min-blocks 12 --max-blocks 80 --episodes 20000 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms min_makespan,lookahead_min_makespan,best_fit_lth,balanced_tact_load,spt_batch,lpt_batch --phase2-score-mode normalized --hidden-dim 128 --action-pool-limit None --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --output-dir output/phase2_batch_machine_train
```

Phase 2 재개 학습은 위와 동일한 RunSpec 인자를 유지하고 아래 옵션만 추가합니다.

```bash
--resume-checkpoint latest
```

Frozen Phase 2 feedback을 사용하는 Phase 1 계층 학습:

```bash
python main.py phase1-train-pair-self-labeling --config config_np_100.yaml --bay-ids 22,23,24 --gyel NP --episodes 20000 --min-blocks 12 --max-blocks 80 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms ppb_lcp6 --hidden-dim 128 --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --actual-validation-candidate-xlsx "착수일 후보 블록.xlsx" --enable-phase2-feedback-score --phase2-feedback-checkpoint output/phase2_batch_machine_train/phase2_batch_machine_policy.pt --output-dir output/phase1_with_phase2_feedback
```

Phase 1 -> merged Phase 2 full-flow checkpoint 평가:

```bash
python main.py phase2-run-full-workflow --config config_np_100.yaml --phase1-checkpoint output/phase1_with_phase2_feedback/phase1_pair_pointer.pt --phase1-bay-ids 22,23,24 --phase1-samples 64 --batch-machine-checkpoint output/phase2_batch_machine_train/phase2_batch_machine_policy.pt --synthetic-source report_formula --synthetic-blocks 30 --gyel NP --output-dir output/full_flow_checkpoint
```

Actual 8일 Phase 2 평가는 W/O 착수시간을 재구성하지 않고 `착수일 후보 블록.xlsx`의 8개 sheet를 authoritative block universe로 사용합니다.

```bash
python scripts/evaluate_phase2_candidate_workbook.py --config config_np_100.yaml --wo-xlsx "input/절단03~04_NP물량_마스킹_WO_수정_260618.xlsx" --candidate-xlsx "착수일 후보 블록.xlsx" --workdays 20260331,20260407,20260408,20260413,20260414,20260415,20260424,20260429 --bay-ids 22,23,24 --phase1-heuristic bevel_first_balanced --checkpoint output/phase2_batch_machine_train/phase2_batch_machine_policy.pt --output-dir output/phase2_actual8_checkpoint
```

Phase 2 checkpoint에는 feature schema, score mode, batch limit, action pool, heuristic bank, sampling 수, Phase 1 Bay 용량비/장척 mask, constraint profile을 포함한 단일 `run_spec`이 저장됩니다. 재개·actual 평가·full-flow는 이 값이 하나라도 다르거나 RunSpec/model/optimizer state가 누락되면 자동 보정하지 않고 실패합니다. Frozen feedback으로 학습한 Phase 1 checkpoint도 원본 Phase 2 checkpoint SHA256과 RunSpec이 정확히 같은 full-flow에서만 사용할 수 있습니다.

### 현재 Phase / Network 경계

- `Environment/simulation.py`: `simulate/replay/Gym`이 사용하는 custom event-driven DES다. SimPy를 사용하지 않고 decision epoch와 machine clock을 직접 전진시킨다.
- `Environment/hierarchical.py`: Phase 1 planning state와 Phase 2 runtime/event state, snapshot/fork, batch transition의 공통 소유자다. full DES와 data/event/constraint 계약은 공유하지만 transition loop는 별도다.
- `Phase1/`: Block -> Bay 배정. `pointer_policy.py`가 공통 planning state에서 만든 pair/env feature를 점수화한다.
- `Phase2/`: merged batch-machine scheduling. `set_pointer_policy.py`가 가변 Machine/W/O set을 인코딩하고, `merged.py`가 순차 action, Bay별 self-labeling, batch/timeline 저장을 담당한다.
- Phase 2 및 full-flow synthetic 학습/검증은 `[제출본]가공공장_중간보고.pdf` 수식을 고정 구현한 `Utils/data/report_formula_data_generator.py`의 `report_formula` source를 사용한다. Phase 1 단독 학습은 기존 block bootstrap을 유지한다.
- `Train/network/mlp.py`: Phase1/2 self-labeling edge/action scorer와 Phase1 pointer policy가 공통으로 사용하는 유일한 `Train` 공용 network helper다.
- `Train/algorithm/`, `Train/runner.py`, legacy `gnn/policy_value/feature_builder` 스택은 현재 실행 경로에서 제거했다. Phase별 학습 루프는 각 `Phase*/` 패키지가 소유한다.
- 현재 Phase 1은 graph-derived pair feature를 사용하는 pointer policy이고, Phase 2는 mean/max set pooling 기반 pointer policy다. 둘 다 GNN message passing은 사용하지 않는다. Frozen Phase 2 best-of-K score를 Phase 1 후보 평가 앞에 붙이는 feedback 기반 계층 학습까지 구현했으며, alternating joint optimization은 검증되지 않은 별도 방식이라 기본 실행 경로에 포함하지 않는다.

---

## 2.1 Phase 1 블록-Bay 휴리스틱 결론

현업 확인 기준 Phase 1 목적은 블록을 Bay 22/23/24에 배정해 Bay별 부하를 평준화하는 것입니다. Bay 25가 포함된 블록은 현재 테스트/발표 범위에서 제외합니다.

평준화 지표는 아래 4개입니다.

1. 강재수량(`STL_QTY`) gap 최소화
2. 절단장(`CUT_LTH`) gap 최소화
3. 베벨수량(`BV_QTY`) gap 최소화
4. 절단장 1000 초과 블록의 Bay 24 배정 최소화

현업 우선순위는 `강재수량 -> 절단장 -> 베벨수량 -> 장척 Bay 22/23 선호`입니다. 다만 장척 블록은 운영상 Bay 24를 먼저 피하는 선호 제약에 가깝기 때문에, 실험에서는 장척 Bay24 회피를 Bay 선택 단계에서 먼저 고려하는 방식이 안정적이었습니다.

### 단순 greedy 조합 실험

설명 가능한 단순 greedy는 아래처럼 나눠 비교했습니다.

- 블록 선택 규칙: `steel_first`, `cut_first`, `bevel_first`, `long_cut_first`
- Bay 배정 규칙: `balance_lexicographic`, `long_cut_preference`

이 4 x 2 조합 중 가장 설명성과 성능이 좋았던 아이디어는 다음입니다.

```text
bevel_first + long_cut_preference
```

의미:

- 블록 선택: 베벨수량이 큰 블록부터 먼저 배정
- Bay 선택: 장척 블록 Bay24 회피 -> 강재수량 -> 절단장 -> 베벨수량 순으로 평가

이 조합이 좋은 이유:

- 강재수량과 절단장은 상관이 높아 하나를 맞추면 다른 하나도 어느 정도 같이 맞춰지는 경향이 있음
- 베벨수량은 특정 블록에 몰리는 희소 지표라 뒤로 미루면 복구가 어려움
- 장척 블록은 개수가 적어 Bay 선택 단계에서 Bay24 회피를 우선 적용해도 제어 가능함

### 코드에서 사용하는 공식 휴리스틱

실제 코드에서 중간발표용 기본 실행 알고리즘은 아래입니다.

```text
long_cut_preferred_balanced
```

이 알고리즘은 `bevel_first + long_cut_preference`의 핵심 해석을 포함하되, 단순 greedy 하나로 고정하지 않고 기존 `priority_sweep_balanced`와 같은 beam/local-search 기반 탐색을 사용합니다.

의미:

- `CUT_LTH > 1000` 블록은 Bay 24를 먼저 피함
- 이후 강재수량, 절단장, 베벨수량 평준화를 평가
- 여러 블록 순서를 탐색해 단순 greedy보다 안정적인 배정을 찾음

8개 현업 후보 작업일 실험 결과:

| 알고리즘 | all_nonworse | any_worse | 장척 Bay24 악화 | 강재수량 악화 | 절단장 악화 | 베벨수량 악화 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `steel_lpt_greedy_insertion` | 1/8 | 7/8 | 6/8 | 0/8 | 0/8 | 2/8 |
| `multi_objective_balanced` | 6/8 | 2/8 | 1/8 | 0/8 | 0/8 | 1/8 |
| `priority_greedy_insertion` | 4/8 | 4/8 | 2/8 | 0/8 | 0/8 | 2/8 |
| `long_cut_preferred_balanced` | 6/8 | 2/8 | 0/8 | 0/8 | 0/8 | 2/8 |

따라서 결론은 다음과 같습니다.

```text
발표에서 설명할 핵심 아이디어:
bevel_first + long_cut_preference

실제 실행/산출물용 알고리즘:
long_cut_preferred_balanced
```

### Phase 1 MDP/self-labeling

Phase 1 학습 구조는 `SELECT_BLOCK -> SELECT_BAY` 계층형 MDP로 둡니다. 현재 live 구현은 invalid action masking과 self-labeling 후보 비교를 사용하며, Phase 2는 별도 W/O 배정 단계 없이 `(W/O batch, Machine)`을 한 번에 선택합니다.

구현 파일:

- `Utils/phase1/phase1_mdp.py`
- `Phase1/pointer_policy.py`
- `Phase1/imitation.py`
- `Phase1/self_labeling.py`
- `Phase1/pair_self_labeling.py`

100건 smoke 기준 산출 결과:

```text
job_count = 100
block_count = 18
trace_row_count = 36
decision_count_per_block = 2
steel_quantity_gap = 1
cut_length_gap = 172.763
bevel_quantity_gap = 8
long_cut_bay24_count = 0
```

100건 self-label overfit smoke 기준 학습 결과:

```text
example_count = 36
epochs = 300
hidden_dim = 128
final_loss = 0.050265
final_accuracy = 1.000000
```

실적 블록분포 기반 variable-size episode 학습 결과:

```text
episode_count = 40
block_count_per_episode = 12~80
train_action_count = 2968
test_action_count = 732
eval_accuracy = 0.760929
eval_SELECT_BLOCK_accuracy = 0.879781
eval_SELECT_BAY_accuracy = 0.642077
```

---

## 3. 폴더 구조

```text
pmsp/
├─ main.py
├─ config.yaml
├─ Agent/
│  └─ heuristics.py
├─ Environment/
│  ├─ data.py
│  ├─ environment.py
│  ├─ hierarchical.py
│  ├─ metrics.py
│  ├─ reward.py
│  ├─ simulation.py
│  ├─ state.py
│  └─ constraints/
│     ├─ base.py
│     ├─ calendar_rules.py
│     ├─ downstream_rules.py
│     ├─ machine_rules.py
│     ├─ profiles.py
│     ├─ registry.py
│     └─ soft_rules.py
├─ Phase1/
│  ├─ orchestrator.py
│  ├─ pair_self_labeling.py
│  └─ pointer_policy.py
├─ Phase2/
│  ├─ merged.py
│  ├─ orchestrator.py
│  ├─ run_spec.py
│  ├─ set_pointer_policy.py
│  └─ state.py
├─ Train/
│  └─ network/
│     └─ mlp.py
├─ Utils/
│  ├─ config.py
│  ├─ data/
│  │  ├─ cutting_data_loader.py
│  │  ├─ cutting_scenario_builder.py
│  │  ├─ factory_builder.py
│  │  ├─ io.py
│  │  ├─ phase2_candidate_workbook.py
│  │  ├─ report_formula_data_generator.py
│  │  ├─ scenario_generator.py
│  │  └─ test_data_selection.py
│  ├─ phase1/
│  │  ├─ phase1_bay_balancer.py
│  │  ├─ phase1_block_data_generator.py
│  │  ├─ phase1_episode_dataset.py
│  │  └─ phase1_mdp.py
│  ├─ learning/
│  │  ├─ learning_data_builder.py
│  │  ├─ phase1_phase2_communication.py
│  │  ├─ phase_agent_checkpoints.py
│  │  └─ phase_graph_mdp.py
│  └─ reporting/
│     ├─ playback_builder.py
│     ├─ pygame_factory_viewer.py
│     ├─ share_report_builder.py
│     ├─ tact_gap_analysis.py
│     └─ tact_time.py
├─ input/
├─ output/
└─ 현재과제_진행체크리스트.md
```

---

## 4. 핵심 개념

### 4.1 action

`CuttingSimulation` full DES의 기본 action mode는 `batch_open`입니다.

즉, 한 step에서 고르는 것은:

- batch 열기: `open_batch:job_id@machine_id`
- batch에 W/O 추가: `add_to_batch:batch_id:job_id`
- batch 닫기 및 투입: `close_batch:batch_id@machine_id`

의 세 가지 action 중 하나입니다.

Merged Phase 2 policy는 이 저수준 action을 직접 고르지 않습니다. 정책은
`SELECT_MACHINE -> SELECT_WO 반복`을 수행하고, `CommonHierarchicalEnvironment`가
batch 목표 개수 도달 또는 추가 가능한 W/O 소진 시 batch를 자동으로 닫습니다.

현업 확인 기준:
- W/O 1~3개를 batch로 묶을 수 있습니다.
- batch의 `길이` 합은 55,000 이하입니다.
- batch는 같이 들어오고 같이 빠집니다.
- `job_machine_pair` 단건 방식은 baseline/debug 전용입니다.

예:
- `open_batch:job_003@PLS21`
- `add_to_batch:B000001:job_008`
- `close_batch:B000001@PLS21`

---

### 4.2 state

환경이 매 step마다 보는 상태는 크게 두 부분입니다.

- 남은 작업 상태
- 설비 / 후공정 / 캘린더 상태

포함 예:
- 남은 작업 수
- 설비별 현재 부하
- 설비별 현재 가용 시각
- 후공정 bay 적치량
- 현재 날짜 / 현재 시각
- 현재 가능한 action 목록

---

### 4.3 reward

현재 reward는 아래를 조합합니다.

- makespan 증가 벌점
- 설비 간 부하 불균형 벌점
- 우선 작업 bonus
- soft 제약 penalty

즉, “빨리 끝내되, 너무 한 설비에 몰리지 말고, 소프트 제약도 가능하면 지키는 방향”입니다.

---

## 5. 제약 구조

이 프로젝트의 제약은 4개 축으로 관리합니다.

1. `category`
2. `hard`
3. `soft`
4. `override`

### 5.1 category

사람이 이해하기 쉬운 큰 분류입니다.

- `machine`
- `capacity`
- `downstream`
- `priority`
- `preference`
- `layout`
- `calendar`

### 5.2 hard

위반하면 후보 action에서 제거됩니다.

예:
- 두께 범위 위반
- 정반 길이 초과
- 설비 비활성
- 적치량 초과

### 5.3 soft

위반해도 후보는 남고 penalty만 부여됩니다.

예:
- 부하 평준화 선호
- 납기 긴급도 반영
- 선호 설비 타입

### 5.4 override

날짜별 운영 예외를 반영합니다.

예:
- 특정 날짜 설비 용량 50%
- 특정 날짜 설비 비가동
- 특정 날짜 하루 작업 수 제한

상세 설명은 [현재과제_진행체크리스트.md](현재과제_진행체크리스트.md)의 제약조건 섹션을 기준으로 봅니다.

---

## 6. calendar 기능

현재 calendar 계열에서 지원하는 기능은 아래와 같습니다.

- 휴무일
- 반일 가동
- 점심시간
- 공장 전체 shutdown window
- 설비별 운영시간
- 설비별 계획 정지
- 설비별 고장

중요:
- 현재는 **작업 시작 시점 기준**입니다.
- 작업이 이미 시작된 뒤 점심시간/고장이 끼어드는 `preemption / resume`은 아직 없습니다.

상세 설명은 [현재과제_진행체크리스트.md](현재과제_진행체크리스트.md)의 calendar/제약 항목을 기준으로 봅니다.

---

## 7. 제약을 추가하는 방법

초보자 기준으로는 아래 3단계만 기억하면 됩니다.

1. 제약 함수 1개 작성
2. `registry.py`에 등록
3. `config.yaml`에서 켜기

즉, simulation 본문을 매번 수정하지 않아도 됩니다.

상세 가이드는 [AGENTS.md](AGENTS.md)와 [현재과제_진행체크리스트.md](현재과제_진행체크리스트.md)의 제약 추가 기준을 따릅니다.

---

## 8. 학습 알고리즘

현재 실행 경로에서 사용하는 학습 방식은 Phase별 self-labeling입니다.

- Phase 1: block-Bay 후보 또는 pair 후보 중 가장 좋은 후보를 pseudo-label로 사용
- Merged Phase 2: `(W/O batch, Machine)` 후보 bank 중 makespan 우선 score가 가장 좋은 후보를 pseudo-label로 사용

legacy `ppo`, `reinforce` runner는 현재 live path에서 제거했습니다.

현재 네트워크는 Phase별 graph state를 MLP edge/action scorer로 점수화합니다.
full message-passing GNN 정책은 아직 실행 경로가 아니므로 legacy 코드를 제거했습니다.

관련 코드:
- [Phase1/pair_self_labeling.py](Phase1/pair_self_labeling.py)
- [Phase2/merged.py](Phase2/merged.py)
- [Train/network/mlp.py](Train/network/mlp.py)

참고 논문/코드는 필요 시 별도 정리하고, 현재 실행 기준은 master checklist에만 유지합니다.

---

## 9. 빠른 실행

### 9.1 현재 설정 요약

```bash
python3 main.py show-config --config config.yaml
```

### 9.2 휴리스틱 시뮬레이션

```bash
python3 main.py simulate --config config.yaml
```

### 9.3 Phase 1 블록-Bay 배정

```bash
python3 main.py phase1 \
  --config config_np_100.yaml \
  --mode heuristic \
  --algorithm long_cut_preferred_balanced \
  --bay-ids 22,23,24 \
  --output-dir output/phase1_np_100_long_cut_preferred_balanced
```

주요 산출물:

- `phase1_block_bay_plan.json`
- `phase1_block_assignments.csv`
- `phase1_bay_loads.csv`

### 9.4 Phase 1 MDP/self-label trace

```bash
python3 main.py phase1-mdp-trace \
  --config config_np_100.yaml \
  --algorithm long_cut_preferred_balanced \
  --bay-ids 22,23,24 \
  --output-dir output/phase1_mdp_trace_np_100_long_cut_preferred
```

주요 산출물:

- `phase1_mdp_trace.csv`
- `phase1_action_table.jsonl`
- `phase1_mdp_manifest.json`

### 9.5 Phase 1 imitation 학습

```bash
python3 main.py phase1-train-imitation \
  --action-table output/phase1_mdp_trace_np_100_long_cut_preferred/phase1_action_table.jsonl \
  --output-dir output/phase1_imitation_np_100_long_cut_preferred_norm_overfit \
  --epochs 300 \
  --lr 0.001 \
  --hidden-dim 128 \
  --seed 0
```

주요 산출물:

- `phase1_pointer.pt`
- `metrics.csv`
- `summary.json`

### 9.6 Phase 1 variable-size episode dataset

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

### 9.7 step-by-step trace

```bash
python3 main.py trace --config config.yaml
```

### 9.8 택트타임 분석 스켈레톤

```bash
python3 main.py analyze-tact --config config.yaml
```

### 9.9 시나리오 생성

```bash
python3 main.py generate-scenario --config config.yaml --duplicate-jobs 2 --output-path output/generated_scenario.yaml
```

### 9.10 학습

generic `train/eval` 명령은 제거했습니다. 현재 학습은 Phase별 CLI를 사용합니다.

```bash
python main.py phase1-train-pair-self-labeling --config config_np_100.yaml --bay-ids 22,23,24 --block-xlsx "input/절단03~04_NP물량_마스킹_블록_수정_260618.xlsx" --gyel NP --episodes 20000 --min-blocks 12 --max-blocks 80 --noise-ratio 0.03 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms ppb_lcp6 --hidden-dim 128 --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --actual-validation-candidate-xlsx "착수일 후보 블록.xlsx" --output-dir output/phase1_pair_v8_candidate_actual8_ppb_lcp6
```

---

## 10. 샘플 설정 예시

예를 들어 `calendar`와 `machine` 제약을 켜고 싶다면:

```yaml
constraints:
  categories:
    machine: true
    calendar: true

  hard_enabled:
    calendar_open: true
    machine_calendar_open: true
    machine_breakdown: true
    machine_enabled: true
    family_eligibility: true
    thickness_range: true
    table_length_limit: true
```

설비별 운영시간 예시:

```yaml
calendar:
  enable_machine_operating_windows: true
  machine_operating_windows:
    default:
      laser_01:
        - "08:00-18:00"
      plasma_01:
        - "00:00-24:00"
```

---

## 11. 아직 비어 있는 것

아직 구현은 되어 있지만 실제 로직이 비어 있거나 단순화된 부분:

- 실데이터 기반 택트타임 산출식
- 셋업 상세 규칙
- 정반 3분할 / 길이 분할 물리 제약
- 후공정 반입/반출 시간 흐름
- preemption / resume
- 2D 레이아웃 상세 모델

즉, 지금은 **연구계획서 전체를 따라갈 수 있는 구조는 완료**되었고,
**현업 상세값과 물리 세부는 앞으로 채우는 단계**입니다.

---

## 12. 문서 안내

- 작업 규칙: [AGENTS.md](AGENTS.md)
- 단일 master checklist: [현재과제_진행체크리스트.md](현재과제_진행체크리스트.md)
- 메일/Q&A 원문 정리: [메일내용/메일내용_정리.md](메일내용/메일내용_정리.md)

---

## 13. 한 줄 요약

이 저장소는 **절단 공정 PMSP형 스케줄링 문제를 위한 실행 가능한 연구용 프레임워크**이며,
**초보자도 제약과 운영 규칙을 함수/설정 단위로 추가할 수 있게 설계된 코드베이스**입니다.
