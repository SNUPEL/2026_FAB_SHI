# implementation status

이 문서는 현재 `pmsp` 프로젝트의 구현 상태를 표로 정리한 문서입니다.

핵심 해석은 아래와 같습니다.

- `완료`
  - 지금 바로 실행 가능하며, 현재 구조 기준으로 기능이 실제 동작함
- `스켈레톤 완료`
  - 코드 뼈대와 연결 위치는 잡혀 있으나, 실제 현업 데이터/상세 로직은 아직 비어 있음
- `미구현`
  - 아직 자리만 없거나, 추후 설계가 더 필요함

즉, 현재 프로젝트는

> `학습과 휴리스틱을 붙일 수 있는 PMSP형 절단 스케줄링 환경의 뼈대는 완성`

된 상태이고,

> `현업 상세 데이터와 상세 물리 로직까지 채운 최종 환경`

은 아직 아닌 상태입니다.

---

## 1. 전체 상태 요약

| 구분 | 상태 | 설명 |
|---|---|---|
| 병렬 이기종 설비 스케줄링 환경 골격 | 완료 | `reset/step`, action 후보 생성, 시간 전진, 스케줄 반영까지 동작 |
| 제약 on/off 구조 | 완료 | `config.yaml`에서 category / hard / soft / override / calendar 제어 가능 |
| 휴리스틱 실행 | 완료 | `spt`, `priority`, `load_balance` 실행 가능 |
| 하드 제약 마스킹 | 완료 | 후보 action 생성 단계에서 제거 |
| 소프트 제약 penalty | 완료 | reward 및 휴리스틱 tie-break에 반영 |
| 캘린더 / 설비 고장 제약 | 완료 | 휴무일, 반일 가동, 점심시간, shutdown, 설비별 운영시간, 설비 고장 반영 |
| 택트타임 분석기 | 스켈레톤 완료 | 실제 실적 분석 로직은 아직 없음 |
| 학습 데이터 생성기 | 스켈레톤 완료 | template 기반 단순 생성만 구현 |
| PPO / REINFORCE / Self-labeling | 완료 | 실제 학습 루프, checkpoint 저장/로드 동작 |
| 2D 레이아웃 상세 물리 모델 | 미구현 | 위치 정보만 보관, 실제 이동/면적/충돌 미반영 |
| 현업 상세 제약 | 미구현 | 셋업, 정반 3분할, 시간대별 반입 제한 등은 추후 반영 |

---

## 2. 항목별 상세 표

| 항목 | 현재 상태 | 지금 되는 것 | 아직 비어 있는 것 | 주요 파일 |
|---|---|---|---|---|
| 문제 정의 | 완료 | 절단 공정을 `비관련 병렬기계 스케줄링` 형태로 표현 | 실제 데이터 기반 세부 목적식 조정 | `docs/problem_definition.md` |
| action 정의 | 완료 | `job-machine pair` 형태로 action 선택 | `machine-then-job` 모드 확장 | `config.yaml`, `docs/action_space_notes.md`, `Environment/simulation.py` |
| 환경 인터페이스 | 완료 | `reset()`, `step(action_id)`, observation 반환 | Gym 완전 호환 wrapper 정리 | `Environment/environment.py` |
| 시뮬레이션 시간 전진 | 완료 | 가능한 action이 없으면 다음 완료 시점 또는 calendar 해제 시점까지 자동 전진 | preemption / resume | `Environment/simulation.py` |
| 설비 부하 추적 | 완료 | 설비별 누적 부하 계산 | 실제 일자/교대/가동 계획 반영 | `Environment/state.py`, `Environment/simulation.py` |
| 후공정 적치 추적 | 완료 | bay별 적치량 누적 추적 | 시간대별 반출/감소 로직 | `Environment/state.py`, `Environment/simulation.py` |
| 하드 제약 구조 | 완료 | 계열, 두께, 길이, 단일 처리, 일일 용량, 적치 한계, calendar, 설비별 운영시간, 설비 고장 제약 적용 | 현업 상세 하드 제약 추가 | `Environment/constraints/machine_rules.py`, `Environment/constraints/downstream_rules.py`, `Environment/constraints/calendar_rules.py` |
| 소프트 제약 구조 | 완료 | 후공정 우선순위, 납기, 설비 선호, 적치 위험, 부하 평준화 penalty 적용 | 실제 penalty 계수 튜닝 | `Environment/constraints/soft_rules.py`, `Environment/reward.py` |
| 제약 registry | 완료 | 함수 1개 추가 후 config에서 on/off 가능 | 없음 | `Environment/constraints/registry.py` |
| reward 구조 | 스켈레톤 완료 | makespan, load balance, priority bonus, soft penalty 조합 | 실제 목적식 계수 검증 | `Environment/reward.py` |
| 휴리스틱 | 완료 | `spt`, `priority`, `load_balance` 사용 가능 | EDD, 적치 최소 초과 규칙, 후공정 우선 규칙 추가 | `Agent/heuristics.py` |
| 택트타임 분석 | 스켈레톤 완료 | 분석기 인터페이스와 demo 실행 가능 | 실적 데이터 기반 분석/회귀/보정계수 | `Utils/tact_time.py`, `main.py analyze-tact` |
| 시나리오 생성 | 스켈레톤 완료 | template 기반 job 복제 생성 | 실제 분포 기반 episode 생성 | `Utils/scenario_generator.py`, `main.py generate-scenario` |
| 학습 러너 | 완료 | `train`, `eval` 명령과 checkpoint 저장/로드 동작 | 현업 데이터 기반 성능 튜닝 | `Train/runner.py` |
| 알고리즘 | 완료 | REINFORCE / PPO / Self-labeling 실제 학습 루프 구현 | 보상/탐색/배치 전략 세부 튜닝 | `Train/algorithm/*.py` |
| 네트워크 | 완료 | GNN + MLP 기반 정책/가치 네트워크 구현 | feature 확정 후 입력 스키마 튜닝 | `Train/network/*.py` |
| 샘플 데이터 | 완료 | 샘플 설비/작업/베이 시나리오 존재 | 실제 현업 데이터 반영 | `input/sample_scenario.yaml` |
| config 기반 제어 | 완료 | category/hard/soft/override/calendar, reward, action 모드 제어 | 현업 룰 확정 후 항목 확장 | `config.yaml`, `docs/config_guide.md` |
| 2D 위치 정보 | 스켈레톤 완료 | 설비 위치 좌표만 보관 | 실제 레이아웃 제약, 이동거리, 면적/충돌 | `Environment/environment.py`, `input/sample_scenario.yaml` |
| 설비 교체 시나리오 | 스켈레톤 완료 | 시나리오 파일에서 설비 목록 수정 가능 | 월별 도입/철거 이벤트 반영 | `input/sample_scenario.yaml` |

---

## 3. 현재 구현된 하드 제약

| 제약 키 | 상태 | 설명 |
|---|---|---|
| `machine_enabled` | 완료 | 설비 사용 가능 여부 |
| `family_eligibility` | 완료 | 설비별 가능 계열 제약 |
| `thickness_range` | 완료 | 설비별 가능 두께 범위 |
| `table_length_limit` | 완료 | 정반 또는 장비 길이 초과 여부 |
| `machine_single_processing` | 완료 | 같은 시점에 한 설비는 하나의 작업만 처리 |
| `daily_capacity_limit` | 완료 | 일일 가용시간 초과 여부 |
| `daily_job_cap_limit` | 완료 | 일일 전체 작업 수 제한 |
| `downstream_capacity` | 완료 | 후공정 bay 적치량 한계 |
| `calendar_open` | 완료 | 휴무일 / 점심시간 / 공장 전체 shutdown |
| `machine_calendar_open` | 완료 | 설비별 운영시간 / 설비별 계획 정지 |
| `machine_breakdown` | 완료 | 특정 설비 시간대 고장 / 정지 |

---

## 4. 현재 구현된 소프트 제약

| 제약 키 | 상태 | 설명 |
|---|---|---|
| `downstream_priority` | 완료 | 후공정 우선순위가 높은 bay를 먼저 보내는 선호 |
| `due_date_urgency` | 완료 | 납기 여유가 적은 작업 우선 |
| `preferred_machine_type` | 완료 | 작업별 선호 설비 타입 |
| `downstream_buffer_warning` | 완료 | 적치 한계 직전 경고 penalty |
| `load_balance_preference` | 완료 | 과부하 설비 회피 선호 |

---

## 5. 지금 바로 실행 가능한 명령

| 명령 | 상태 | 설명 |
|---|---|---|
| `python3 main.py show-config --config config.yaml` | 완료 | 현재 활성 config 요약 출력 |
| `python3 main.py simulate --config config.yaml` | 완료 | 휴리스틱 1개로 전체 시뮬레이션 실행 |
| `python3 main.py trace --config config.yaml` | 완료 | step 단위 trace 출력 |
| `python3 main.py analyze-tact --config config.yaml` | 완료 | 택트타임 분석 스켈레톤 실행 |
| `python3 main.py generate-scenario --config config.yaml --duplicate-jobs 2 --output-path output/generated_scenario.yaml` | 완료 | 시나리오 생성 스켈레톤 실행 |
| `python3 main.py train --config config.yaml` | 완료 | 선택한 알고리즘으로 실제 학습 실행 |
| `python3 main.py eval --config config.yaml` | 완료 | 저장된 checkpoint 또는 휴리스틱 평가 |

---

## 6. 아직 실제 데이터가 오면 채워야 하는 항목

| 우선순위 | 항목 | 왜 필요한가 | 들어갈 파일 |
|---|---|---|---|
| 1 | 택트타임 산출 로직 | 처리시간 `p(i,m)`이 있어야 환경과 학습이 현실화됨 | `Utils/tact_time.py`, `Environment/data.py` |
| 2 | 현업 하드 제약 확정 | 마스킹이 잘못되면 불가능한 스케줄이 나옴 | `Environment/constraints/` |
| 3 | 현업 소프트 제약 확정 | penalty와 우선순위 방향이 맞아야 함 | `Environment/constraints/soft_rules.py`, `Environment/reward.py` |
| 4 | 셋업 / 장비 전환 상세 로직 | 실제 절단 생산성 차이를 반영해야 함 | `Environment/simulation.py` |
| 5 | 후공정 반입/반출 시간 로직 | 적치량이 단순 누적이 아니라 시간에 따라 변함 | `Environment/simulation.py` |
| 6 | 현업 feature / reward 튜닝 | 학습 성능을 실제 목표에 맞춰야 함 | `Train/network/*.py`, `Environment/reward.py`, `Train/algorithm/*.py` |

---

## 7. 결론

현재 상태를 한 문장으로 정리하면 아래와 같습니다.

> `연구계획서의 전체 흐름을 따라갈 수 있는 절단 공정 PMSP형 스케줄링 스켈레톤 환경은 구현 완료`  
> `다만 실제 현업 데이터를 반영한 최종 택트타임/제약/학습 로직은 아직 채워야 함`
