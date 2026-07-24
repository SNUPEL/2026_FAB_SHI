# 다계열 Phase 1·Phase 2 학습 실행 가이드

이 문서는 2026-07-21 확정 코드의 장시간 학습 명령과 비교 실험 계약을 고정한다.
Windows `cmd`에서 저장소 루트로 이동하고 `conda activate simenv`를 실행한 상태를
기준으로 한다.

## 1. 현재 완료 경계

코드·state·action·score·합성데이터·EQP 매핑·15대 planning factory와 짧은 smoke
검증은 완료됐다. 다음은 아직 완료가 아니다.

1. Phase 1·Phase 2 장시간 학습
2. holdout/full-scale 성능 비교
3. 신규 W/O 전체 actual time-axis replay에서 mapped 15대와 `EQP_3` 분리 audit

따라서 아래 명령을 성공적으로 시작했다는 사실과 연구 성능 검증 완료를 같은 의미로
사용하지 않는다.

## 2. 구형 NP-only 명령과의 차이

현재 공개 학습 경로는 NP/FN/FL/NC를 함께 생성하는 `MIXED` 경로 하나다. Phase 1은
Bay `22`, `23`, `24`, `25`, `trans`와 설비 수 기반 용량비 `4:3:4:2:2`를 코드에서
가져온다. 다음 구형 인자는 더 이상 Phase 1 학습 CLI에 넣지 않는다.

- `--bay-ids`
- `--block-xlsx`
- `--gyel`
- `--noise-ratio`
- `--hard-case-ratio` 및 나머지 hard-case 인자
- `--actual-validation-candidate-xlsx`
- `--heuristic-algorithms ppb_lcp6`

과거 `phase1_pair_v9_capacity_actual8_ppb_lcp6` 및 `joint_five_bay_*` checkpoint는
NP-only 또는 5-Bay 단일 teacher feature로 학습됐으므로 재개하지 않는다. 현재
scope는 `resource_pool_subproblems_v1`, feature schema는 `20+8`이며 새 output에서
처음 학습해야 한다.

## 3. CUDA 확인

```cmd
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

`torch.cuda.is_available()`이 `True`여야 아래 `--device cuda` 명령을 사용할 수 있다.
CUDA가 보이지 않으면 학습 코드는 CPU로 조용히 대체하지 않고 오류로 종료한다.

## 4. Phase 1 학습

### 4.1 목적함수 범위 선택

Phase 1은 부모 episode를 `NP_NC(22/23/24)`와 `FN_FL(25/trans)` 두 자원군
서브문제로 나눈다. 다음 목적함수 범위는 **각 서브문제 안에서만**
적용된다. `--objective-scope`를 생략하면 `shared_and_series`가 적용되지만,
현재 권장 실험은 `series_only`다.

| `--objective-scope` | 사전식 score | validation 그래프 |
| --- | --- | --- |
| `shared_and_series` | 활성 자원군 전체 W/O -> 활성 계열별 W/O 합 -> 전체 CUT -> 계열별 CUT 합 -> 전체 BV -> 계열별 BV 합 | score 6개 + 세부 gap |
| `series_only` | 활성 계열별 W/O gap 합 -> CUT gap 합 -> BV gap 합 | score 3개 + 세부 gap |

> **`series_only` 정확한 score:**
> `NP_NC=(NP gap+NC gap)`, `FN_FL=(FN gap+FL gap)`을 W/O -> CUT_LTH -> BV_QTY
> 순서로 별도 비교한다. 두 자원군 score를 합쳐 teacher를 선정하지 않는다.

> **Checkpoint에는 objective scope가 저장된다. 동일 `resource_pool_subproblems_v1`
> schema 안에서 다른 scope로 재개하면 기존 model,
> optimizer, 완료 episode는 그대로 복원하고 요청한 새 scope로 이어서 학습한다. 이때
> 의미가 달라진 과거 best-validation score만 초기화하며 전환 사실을 로그와 새
> checkpoint에 기록한다. 19차원 과거 checkpoint는 전환 대상이 아니다.**

### 4.2 비교용 `shared_and_series` 처음 학습

```cmd
python main.py phase1-train-pair-self-labeling --config config_mixed.yaml --episodes 20000 --min-blocks 12 --max-blocks 80 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms all --hidden-dim 128 --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --seed 0 --device cuda --objective-scope shared_and_series --output-dir output/phase1_mixed_resource_pool_shared_v1
```

매 episode에서 물리 블록 12~80개를 가진 MIXED 합성문제를 새로 만든다.
그 뒤 각 자원군에서 policy greedy/sample 후보 64개와 다음 세 휴리스틱을
별도로 비교한다.

| CLI 이름 | 블록 선택 기준 | Bay 선택 기준 |
| --- | --- | --- |
| `wo_first_balanced` | W/O 수가 큰 block-series 우선 | 선택한 objective scope의 사전식 gap |
| `cut_first_balanced` | CUT_LTH가 큰 block-series 우선 | 선택한 objective scope의 사전식 gap |
| `bevel_first_balanced` | BV_QTY가 큰 block-series 우선 | 선택한 objective scope의 사전식 gap |

모든 후보는 동일한 hard mask와 서브문제별 사전식 score로 평가된다.
`NP_NC teacher -> CE update`를 수행한 뒤 `FN_FL teacher -> CE update`를 수행한다.
두 자원군이 모두 있으면 parent episode당 update는 2번이다. 없는 자원군은
row를 만들지 않는다.

현재 Phase 1 state는 pair 20차원, environment 8차원이다. pair의 NP/NC/FN/FL
indicator는 모두 별도다. 과거 checkpoint는 입력/scope 계약이 달라 재사용하지 않는다.

### 4.3 계열별 목적함수만 사용하는 처음 학습

공유 설비군 gap을 제외하고 계열별 평준화 3개만 학습하려면 다음 명령을 사용한다.

```cmd
python main.py phase1-train-pair-self-labeling --config config_mixed.yaml --episodes 20000 --min-blocks 12 --max-blocks 80 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms all --hidden-dim 128 --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --seed 0 --device cuda --objective-scope series_only --output-dir output/phase1_mixed_resource_pool_v1
```

각 `subproblem_metrics.csv` row의 `score_json`은 다음 3개 값이다.

```text
NP_NC: (NP gap+NC gap의 W/O, CUT_LTH, BV_QTY)
FN_FL: (FN gap+FL gap의 W/O, CUT_LTH, BV_QTY)
```

### 4.4 이어서 학습

기존 공유+계열 목적함수를 이어서 학습하는 명령은 다음과 같다.

```cmd
python main.py phase1-train-pair-self-labeling --config config_mixed.yaml --episodes 20000 --min-blocks 12 --max-blocks 80 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms all --hidden-dim 128 --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --seed 0 --device cuda --objective-scope shared_and_series --resume-checkpoint latest --output-dir output/phase1_mixed_resource_pool_shared_v1
```

계열별 목적함수를 이어서 학습하는 명령은 다음과 같다.

```cmd
python main.py phase1-train-pair-self-labeling --config config_mixed.yaml --episodes 20000 --min-blocks 12 --max-blocks 80 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms all --hidden-dim 128 --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --seed 0 --device cuda --objective-scope series_only --resume-checkpoint latest --output-dir output/phase1_mixed_resource_pool_v1
```

동일한 신규 scope의 `shared_and_series` checkpoint를 `series_only`로 전환하려면
다음 명령을 사용한다.

```cmd
python main.py phase1-train-pair-self-labeling --config config_mixed.yaml --episodes 20000 --min-blocks 12 --max-blocks 80 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms all --hidden-dim 128 --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --seed 0 --device cuda --objective-scope series_only --output-dir output/phase1_mixed_resource_pool_shared_v1 --resume-checkpoint latest
```

최신 checkpoint가 episode 2,700이면 episode 2,701부터 시작한다. 기존 CSV에는
이전 objective scope와 새 scope가 함께 보존되고 best-validation score는 초기화된다.

`--episodes`는 추가 학습 횟수가 아니라 최종 episode 번호다. 예를 들어 3,100에서
20,000까지 이어서 학습할 때 `20000`을 유지한다. 이미 20,000까지 완료한 모델을
10,000회 더 학습하려면 `30000`으로 지정한다. output 경로와 나머지 계약 인자는
기존 실행과 같아야 한다.

### 4.5 Loss 및 validation 그래프

```cmd
python scripts/plot_phase1_loss_only.py output/phase1_mixed_resource_pool_shared_v1 --window 100
python scripts/plot_phase1_loss_only.py output/phase1_mixed_resource_pool_v1 --window 100
```

`metrics.csv`만 읽어 `loss_curve.png`, `loss_curve_normalized.png`,
`loss_only_status.json`을 갱신한다.

Phase 1 validation은 parent holdout을 `NP`, `NC`, `NP_NC`, `FN`, `FL`, `FN_FL`
view로 나눈다. `validation_candidate_summary.csv`에는 NP/NC/NP_NC/FN/FL/FN_FL의
W/O·CUT·BV gap 18개 column이 있다. 해당 계열이 없는 값은 0이 아니라
빈값/N/A다. 값이 존재하는 진단 field는 `validation_<field>.png`로 그린다.
`series_only` teacher/rank는 계열별 gap 합만 사용한다.

학습 update 단위는 `subproblem_metrics.csv`에 한 row씩 저장된다. `metrics.csv`의
parent loss는 해당 episode에서 수행한 1~2개 subproblem loss의 평균이며, teacher를
다시 통합 비교한 값이 아니다.

## 5. Phase 2 비교 실험 설계

Phase 2는 매 episode마다 MIXED 문제를 생성하고 Phase 1 결과로 block-series를 Bay에
배정한 뒤, Bay별 W/O batch-machine subproblem을 푼다. Phase 1은 Phase 2 학습 중
업데이트하지 않는다. 다음 네 실험을 **동일한 seed와 나머지 Phase 2 인자**로 각각
실행한다.

현재 Phase 2 학습 action은 `SELECT_WO` 하나다. 환경은 현재 전역 시각에 실행 가능한
유휴 설비 중 machine ID가 작은 설비를 확정하고 목표 batch 크기를 만든다. 유휴 설비가
없을 때만 전체 설비의 다음 최소 완료 시각으로 event jump한다. batch close는 설비의
미래 완료 시각만 예약하므로 여러 설비와 Bay가 같은 시각에 병렬 시작할 수 있다.
설비 결정과 event jump는 candidate 수, sampling, CE loss에 포함되지 않는다.

Phase 2 후보 비교는 다음 raw 사전식 score를 사용한다.

```text
hard violation
-> makespan
-> Bay 내부 CUT_LTH max-min gap 합
-> Bay 내부 W/O 수 max-min gap 합
-> Bay 내부 BV_QTY max-min gap 합
-> Bay 내부 점유시간 max-min gap 합
```

Bay별 subproblem의 teacher는 해당 Bay gap으로 선택하고, 여러 Bay를 합친 validation과
full-flow 결과는 Bay별 gap을 합산한다. `--phase2-score-mode normalized`는 각 Bay의
`max-min`을 같은 Bay의 평균 부하로 나눈 뒤 합산한다.

| 실험 | Phase 1 upstream | Phase 1 학습 여부 | output |
| --- | --- | --- | --- |
| H-WO | `wo_first_balanced` | 고정, 학습 안 함 | `phase2_upstream_wo` |
| H-CUT | `cut_first_balanced` | 고정, 학습 안 함 | `phase2_upstream_cut` |
| H-BV | `bevel_first_balanced` | 고정, 학습 안 함 | `phase2_upstream_bevel` |
| A-P1 | frozen Phase 1 checkpoint best-of-32 | 고정, 학습 안 함 | `phase2_upstream_agent_s32` |

`--phase1-heuristic`과 `--phase1-checkpoint`는 동시에 사용할 수 없다. 네 실험의
output을 절대 공유하지 않는다.

Phase 2 candidate bank는 모든 실험에서 동일하게 유지한다.

```text
workload_makespan_dispatch
min_makespan
lookahead_min_makespan
best_fit_lth
spt_batch
lpt_batch
agent_greedy + agent_sample_*
```

### 5.1 W/O 우선 Phase 1 휴리스틱 고정

```cmd
python main.py phase2-train-batch-machine-self-labeling --config config_mixed.yaml --phase1-heuristic wo_first_balanced --episodes 20000 --min-blocks 12 --max-blocks 80 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms workload_makespan_dispatch,min_makespan,lookahead_min_makespan,best_fit_lth,spt_batch,lpt_batch --phase2-score-mode raw --hidden-dim 128 --action-pool-limit None --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --seed 0 --device cuda --output-dir output/phase2_upstream_wo
```

### 5.2 절단장 우선 Phase 1 휴리스틱 고정

```cmd
python main.py phase2-train-batch-machine-self-labeling --config config_mixed.yaml --phase1-heuristic cut_first_balanced --episodes 20000 --min-blocks 12 --max-blocks 80 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms workload_makespan_dispatch,min_makespan,lookahead_min_makespan,best_fit_lth,spt_batch,lpt_batch --phase2-score-mode raw --hidden-dim 128 --action-pool-limit None --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --seed 0 --device cuda --output-dir output/phase2_upstream_cut
```

### 5.3 베벨 우선 Phase 1 휴리스틱 고정

```cmd
python main.py phase2-train-batch-machine-self-labeling --config config_mixed.yaml --phase1-heuristic bevel_first_balanced --episodes 20000 --min-blocks 12 --max-blocks 80 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms workload_makespan_dispatch,min_makespan,lookahead_min_makespan,best_fit_lth,spt_batch,lpt_batch --phase2-score-mode raw --hidden-dim 128 --action-pool-limit None --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --seed 0 --device cuda --output-dir output/phase2_upstream_bevel
```

### 5.4 학습된 Phase 1 agent 고정

Phase 1 장시간 학습이 끝난 뒤 실행한다.

```cmd
python main.py phase2-train-batch-machine-self-labeling --config config_mixed.yaml --phase1-checkpoint output/phase1_mixed_resource_pool_v1/phase1_pair_pointer_best.pt --phase1-samples 32 --phase1-temperature 1.0 --episodes 20000 --min-blocks 12 --max-blocks 80 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms workload_makespan_dispatch,min_makespan,lookahead_min_makespan,best_fit_lth,spt_batch,lpt_batch --phase2-score-mode raw --hidden-dim 128 --action-pool-limit None --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --seed 0 --device cuda --output-dir output/phase2_upstream_agent_s32
```

이 모드에서 Phase 1 agent는 freeze 상태다. 매 Phase 2 episode마다
`NP_NC`와 `FN_FL` 각각에서 best-of-32를 별도 선정한 뒤 assignment를 합친다.
Phase 2 CE/NLL만 update되며 Phase 1 parameter는 바뀌지 않는다.

위 명령은 권장 `series_only` checkpoint 기준이다. `shared_and_series` 비교는
`output/phase1_mixed_resource_pool_shared_v1/phase1_pair_pointer_best.pt`를 사용하고 Phase 2
output도 별도로 지정한다. best-of-K 선택은 checkpoint에 저장된 objective scope를
그대로 사용한다.

## 6. Phase 2 재개 학습

각 실험의 처음 학습 명령과 모든 RunSpec 인자를 그대로 유지하고
`--resume-checkpoint latest`만 추가한다. 예를 들어 frozen Phase 1 agent 실험은 다음과
같다.

```cmd
python main.py phase2-train-batch-machine-self-labeling --config config_mixed.yaml --phase1-checkpoint output/phase1_mixed_resource_pool_v1/phase1_pair_pointer_best.pt --phase1-samples 32 --phase1-temperature 1.0 --episodes 20000 --min-blocks 12 --max-blocks 80 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms workload_makespan_dispatch,min_makespan,lookahead_min_makespan,best_fit_lth,spt_batch,lpt_batch --phase2-score-mode raw --hidden-dim 128 --action-pool-limit None --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --seed 0 --device cuda --resume-checkpoint latest --output-dir output/phase2_upstream_agent_s32
```

Phase 2도 `--episodes`는 최종 episode 번호다. score mode, batch limit, action pool,
heuristic bank, Phase 1 upstream 계약이 checkpoint와 다르면 fallback 없이 실패한다.
`phase2_wo_pointer_v3_family` 이전의 Machine/W/O 2-stage checkpoint는 state 차원과
학습 action이 다르므로 재개할 수 없으며 새 output 디렉터리에서 처음 학습해야 한다.

## 7. Phase 2 Loss 및 validation 결과

loss-only script는 `metrics.csv`의 공통 `episode/loss` 컬럼만 사용하므로 Phase 2
output에도 사용할 수 있다.

```cmd
python scripts/plot_phase1_loss_only.py output/phase2_upstream_wo --window 100
python scripts/plot_phase1_loss_only.py output/phase2_upstream_cut --window 100
python scripts/plot_phase1_loss_only.py output/phase2_upstream_bevel --window 100
python scripts/plot_phase1_loss_only.py output/phase2_upstream_agent_s32 --window 100
```

Phase 2는 validation 주기마다 다음 파일을 자동 갱신한다.

- `validation_summary.csv`
- `validation_candidate_summary.csv`
- `validation_makespan.png`
- `validation_wo_gap.png`
- `validation_cut_gap.png`
- `validation_bevel_gap.png`
- `validation_occupancy_gap.png`
- `validation_policy_rank.png`
- `validation_best_source_counts.png`

## 8. 학습 checkpoint를 full flow에서 사용

### 8.1 Phase 1 휴리스틱 + 학습된 Phase 2 agent

아래 예시는 W/O 우선 Phase 1 고정으로 학습한 Phase 2 checkpoint를 같은 upstream
계약으로 실행한다.

```cmd
python main.py phase2-run-full-workflow --config config_mixed.yaml --phase1-heuristic wo_first_balanced --batch-machine-checkpoint output/phase2_upstream_wo/phase2_batch_machine_policy.pt --synthetic-blocks 30 --seed 0 --output-dir output/full_flow_upstream_wo
```

절단장·베벨 고정 실험은 `--phase1-heuristic`과 checkpoint/output 경로를 해당 실험
이름으로 함께 바꾼다.

### 8.2 학습된 Phase 1 agent + 학습된 Phase 2 agent

```cmd
python main.py phase2-run-full-workflow --config config_mixed.yaml --phase1-checkpoint output/phase1_mixed_resource_pool_v1/phase1_pair_pointer_best.pt --phase1-samples 32 --phase1-temperature 1.0 --batch-machine-checkpoint output/phase2_upstream_agent_s32/phase2_batch_machine_policy.pt --synthetic-blocks 30 --seed 0 --output-dir output/full_flow_agent_agent
```

Phase 2 checkpoint 실행은 저장된 RunSpec을 상속한다. 학습 때와 다른 Phase 1
checkpoint, sampling 수, score mode 또는 batch/action 계약을 섞으면 실행하지 않는다.

## 9. 권장 실행 순서

1. CUDA 인식 확인
2. Phase 1 `series_only`를 장시간 학습
3. 필요할 때만 `shared_and_series`를 별도 output에서 비교 학습
4. 두 Phase 1 자원군의 subproblem loss와 6개 validation view 확인
5. Phase 2 H-WO, H-CUT, H-BV 세 실험 실행
6. Phase 2 A-P1 frozen-agent 실험 실행
7. 네 Phase 2 실험의 동일 holdout validation 비교
8. 가장 좋은 계약으로 full-flow checkpoint 평가
9. 신규 실제 W/O time-axis audit와 최종 full-scale 평가

네 Phase 2 실험은 upstream만 다르고 나머지 seed·episode·rollout·heuristic bank·score
mode가 같아야 한다. 그렇지 않으면 성능 차이를 Phase 1 upstream의 효과로 해석할 수
없다.
