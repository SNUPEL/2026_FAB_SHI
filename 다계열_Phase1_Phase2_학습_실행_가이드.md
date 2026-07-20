# 다계열 Phase 1·Phase 2 학습 실행 가이드

이 문서는 2026-07-16 확정 코드의 장시간 학습 명령과 비교 실험 계약을 고정한다.
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

과거 `phase1_pair_v9_capacity_actual8_ppb_lcp6` checkpoint는 NP-only feature와 과거
Bay 계약으로 학습됐으므로 재개하지 않는다. 현재 scope는
`joint_five_bay_v3_shared_pool`이며 새 output에서 처음 학습해야 한다.

## 3. CUDA 확인

```cmd
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

`torch.cuda.is_available()`이 `True`여야 아래 `--device cuda` 명령을 사용할 수 있다.
CUDA가 보이지 않으면 학습 코드는 CPU로 조용히 대체하지 않고 오류로 종료한다.

## 4. Phase 1 학습

### 4.1 처음 학습

```cmd
python main.py phase1-train-pair-self-labeling --config config_np_100.yaml --episodes 20000 --min-blocks 12 --max-blocks 80 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms all --hidden-dim 128 --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --seed 0 --device cuda --output-dir output/phase1_mixed_shared_pool_v3
```

매 episode에서 물리 블록 12~80개를 가진 MIXED 합성문제를 새로 만든다. 현재
policy의 greedy/sample 후보 64개와 다음 세 휴리스틱을 비교한다.

| CLI 이름 | 블록 선택 기준 | Bay 선택 기준 |
| --- | --- | --- |
| `wo_first_balanced` | W/O 수가 큰 block-series 우선 | 공유 전체/계열별 W/O → CUT_LTH → BV_QTY gap |
| `cut_first_balanced` | CUT_LTH가 큰 block-series 우선 | 공유 전체/계열별 W/O → CUT_LTH → BV_QTY gap |
| `bevel_first_balanced` | BV_QTY가 큰 block-series 우선 | 공유 전체/계열별 W/O → CUT_LTH → BV_QTY gap |

모든 후보는 동일한 hard mask와 사전식 score로 평가된다. 가장 좋은 완성 계획의
action sequence가 self-label이 되며 CE/NLL update를 한 번 수행한다.

현재 Phase 1 state는 pair 19차원, environment 8차원이다. 과거 16+5 checkpoint는
입력 계약이 달라 재사용하지 않으며 새 output directory에서 처음부터 학습한다.

### 4.2 이어서 학습

```cmd
python main.py phase1-train-pair-self-labeling --config config_np_100.yaml --episodes 20000 --min-blocks 12 --max-blocks 80 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms all --hidden-dim 128 --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --seed 0 --device cuda --resume-checkpoint latest --output-dir output/phase1_mixed_shared_pool_v3
```

`--episodes`는 추가 학습 횟수가 아니라 최종 episode 번호다. 예를 들어 3,100에서
20,000까지 이어서 학습할 때 `20000`을 유지한다. 이미 20,000까지 완료한 모델을
10,000회 더 학습하려면 `30000`으로 지정한다. output 경로와 나머지 계약 인자는
기존 실행과 같아야 한다.

### 4.3 Loss 그래프

```cmd
python scripts/plot_phase1_loss_only.py output/phase1_mixed_shared_pool_v3 --window 100
```

`metrics.csv`만 읽어 `loss_curve.png`, `loss_curve_normalized.png`,
`loss_only_status.json`을 갱신한다.

Phase 1 validation은 공유 설비군 전체 gap인 `validation_wo_gap.png`,
`validation_cut_gap.png`, `validation_bevel_gap.png`와 계열별 gap인
`validation_series_wo_gap.png`, `validation_series_cut_gap.png`,
`validation_series_bevel_gap.png`를 함께 생성한다.

## 5. Phase 2 비교 실험 설계

Phase 2는 매 episode마다 MIXED 문제를 생성하고 Phase 1 결과로 block-series를 Bay에
배정한 뒤, Bay별 W/O batch-machine subproblem을 푼다. Phase 1은 Phase 2 학습 중
업데이트하지 않는다. 다음 네 실험을 **동일한 seed와 나머지 Phase 2 인자**로 각각
실행한다.

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
balanced_tact_load
spt_batch
lpt_batch
agent_greedy + agent_sample_*
```

### 5.1 W/O 우선 Phase 1 휴리스틱 고정

```cmd
python main.py phase2-train-batch-machine-self-labeling --config config_np_100.yaml --phase1-heuristic wo_first_balanced --episodes 20000 --min-blocks 12 --max-blocks 80 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms workload_makespan_dispatch,min_makespan,lookahead_min_makespan,best_fit_lth,balanced_tact_load,spt_batch,lpt_batch --phase2-score-mode raw --hidden-dim 128 --action-pool-limit None --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --seed 0 --device cuda --output-dir output/phase2_upstream_wo
```

### 5.2 절단장 우선 Phase 1 휴리스틱 고정

```cmd
python main.py phase2-train-batch-machine-self-labeling --config config_np_100.yaml --phase1-heuristic cut_first_balanced --episodes 20000 --min-blocks 12 --max-blocks 80 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms workload_makespan_dispatch,min_makespan,lookahead_min_makespan,best_fit_lth,balanced_tact_load,spt_batch,lpt_batch --phase2-score-mode raw --hidden-dim 128 --action-pool-limit None --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --seed 0 --device cuda --output-dir output/phase2_upstream_cut
```

### 5.3 베벨 우선 Phase 1 휴리스틱 고정

```cmd
python main.py phase2-train-batch-machine-self-labeling --config config_np_100.yaml --phase1-heuristic bevel_first_balanced --episodes 20000 --min-blocks 12 --max-blocks 80 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms workload_makespan_dispatch,min_makespan,lookahead_min_makespan,best_fit_lth,balanced_tact_load,spt_batch,lpt_batch --phase2-score-mode raw --hidden-dim 128 --action-pool-limit None --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --seed 0 --device cuda --output-dir output/phase2_upstream_bevel
```

### 5.4 학습된 Phase 1 agent 고정

Phase 1 장시간 학습이 끝난 뒤 실행한다.

```cmd
python main.py phase2-train-batch-machine-self-labeling --config config_np_100.yaml --phase1-checkpoint output/phase1_mixed_shared_pool_v3/phase1_pair_pointer_best.pt --phase1-samples 32 --phase1-temperature 1.0 --episodes 20000 --min-blocks 12 --max-blocks 80 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms workload_makespan_dispatch,min_makespan,lookahead_min_makespan,best_fit_lth,balanced_tact_load,spt_batch,lpt_batch --phase2-score-mode raw --hidden-dim 128 --action-pool-limit None --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --seed 0 --device cuda --output-dir output/phase2_upstream_agent_s32
```

이 모드에서 Phase 1 agent는 freeze 상태다. 매 Phase 2 episode마다 Phase 1 완성 계획
32개를 생성하고 Phase 1 사전식 score가 가장 좋은 계획 하나를 upstream 배정으로
사용한다. Phase 2 CE/NLL만 update되며 Phase 1 parameter는 바뀌지 않는다.

## 6. Phase 2 재개 학습

각 실험의 처음 학습 명령과 모든 RunSpec 인자를 그대로 유지하고
`--resume-checkpoint latest`만 추가한다. 예를 들어 frozen Phase 1 agent 실험은 다음과
같다.

```cmd
python main.py phase2-train-batch-machine-self-labeling --config config_np_100.yaml --phase1-checkpoint output/phase1_mixed_shared_pool_v3/phase1_pair_pointer_best.pt --phase1-samples 32 --phase1-temperature 1.0 --episodes 20000 --min-blocks 12 --max-blocks 80 --rollout-samples 64 --rollout-samples_validation 64 --heuristic-algorithms workload_makespan_dispatch,min_makespan,lookahead_min_makespan,best_fit_lth,balanced_tact_load,spt_batch,lpt_batch --phase2-score-mode raw --hidden-dim 128 --action-pool-limit None --checkpoint-every 100 --validation-every 100 --validation-episodes 20 --seed 0 --device cuda --resume-checkpoint latest --output-dir output/phase2_upstream_agent_s32
```

Phase 2도 `--episodes`는 최종 episode 번호다. score mode, batch limit, action pool,
heuristic bank, Phase 1 upstream 계약이 checkpoint와 다르면 fallback 없이 실패한다.

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
python main.py phase2-run-full-workflow --config config_np_100.yaml --phase1-heuristic wo_first_balanced --batch-machine-checkpoint output/phase2_upstream_wo/phase2_batch_machine_policy.pt --synthetic-blocks 30 --seed 0 --output-dir output/full_flow_upstream_wo
```

절단장·베벨 고정 실험은 `--phase1-heuristic`과 checkpoint/output 경로를 해당 실험
이름으로 함께 바꾼다.

### 8.2 학습된 Phase 1 agent + 학습된 Phase 2 agent

```cmd
python main.py phase2-run-full-workflow --config config_np_100.yaml --phase1-checkpoint output/phase1_mixed_shared_pool_v3/phase1_pair_pointer_best.pt --phase1-samples 32 --phase1-temperature 1.0 --batch-machine-checkpoint output/phase2_upstream_agent_s32/phase2_batch_machine_policy.pt --synthetic-blocks 30 --seed 0 --output-dir output/full_flow_agent_agent
```

Phase 2 checkpoint 실행은 저장된 RunSpec을 상속한다. 학습 때와 다른 Phase 1
checkpoint, sampling 수, score mode 또는 batch/action 계약을 섞으면 실행하지 않는다.

## 9. 권장 실행 순서

1. CUDA 인식 확인
2. Phase 1 `phase1_mixed_shared_pool_v3` 장시간 학습
3. Phase 1 validation과 loss 확인
4. Phase 2 H-WO, H-CUT, H-BV 세 실험 실행
5. Phase 2 A-P1 frozen-agent 실험 실행
6. 네 Phase 2 실험의 동일 holdout validation 비교
7. 가장 좋은 계약으로 full-flow checkpoint 평가
8. 신규 실제 W/O time-axis audit와 최종 full-scale 평가

네 Phase 2 실험은 upstream만 다르고 나머지 seed·episode·rollout·heuristic bank·score
mode가 같아야 한다. 그렇지 않으면 성능 차이를 Phase 1 upstream의 효과로 해석할 수
없다.
