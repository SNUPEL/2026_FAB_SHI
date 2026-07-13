# 절단 QA 개발명세 v2 검토 정리

원본: `메일내용/절단_QA_개발명세_정리본_v2.docx`

이 문서는 Word 문서의 문단, 표, 글자색 표시를 기준으로 다시 정리한 개발 기준 문서이다. 원본 Word는 문단 434개, 표 21개로 구성되어 있으며, 색상으로 상태를 구분하고 있다.

---

## 1. 색상 해석

Word 문서에서 확인한 주요 글자색은 다음과 같다.

| 색상 코드 | 문서 내 의미 | 개발 처리 |
|---|---|---|
| `1F4E79` | 제목/섹션명 | 구조 구분용 |
| `70AD47` | 확정 | 코드에 직접 반영 |
| `C00000` | Hard | 위반 불가 제약, validator에서 fail 처리 |
| `7030A0` | 분석 필요 | 데이터 분석/상관분석/모델링 필요 |
| `C45911` | 배경 | 초기 회의 근거, 최종 Q&A와 충돌하면 Q&A 우선 |
| `666666` | 보류/제외 | 현재 NP 초안에서는 비활성화 |
| `FFFFFF` | 표 헤더 | 표 제목 셀 |

정리 기준은 명확하다. 초록색은 바로 구현, 빨간색은 hard constraint, 보라색은 분석 태스크, 회색은 1차 모델 제외, 갈색은 참고 배경이다.

---

## 2. 최종 개발 범위

현재 개발해야 하는 문제는 다음 문장으로 정리된다.

> NP 계열 W/O를 대상으로, Bay 22/23에 배치된 7대 PLS 설비에 대해 계획 착수일, Bay, 장비를 결정하고, 후공정 일정과 블록셋 동일 Bay 제약을 만족하면서 Bay/설비 부하를 평준화하는 초기 절단 스케줄링 문제

| 구분 | 최종 기준 |
|---|---|
| 대상 데이터 | `절단03~04_NP물량_마스킹.xlsx` |
| 대상 계열 | NP |
| 의사결정 | 계획 착수일, Bay, 장비 |
| 제외할 순서 정보 | `WK_SEQ` |
| 대상 Bay | Bay 22, Bay 23 |
| 대상 설비 | PLS21, PLS22, PLS23, PLS24, PLS31, PLS32, PLS33 |
| 문제 유형 | 병렬기계 스케줄링, 설비 할당, 일자 배정 |
| 핵심 Hard 제약 | 후공정 일정, 블록셋 동일 Bay, 설비당 W/O 3개, 길이합 55000 |
| 1차 제외 | NCG, 광폭, 크레인 과부하, 적재순서, 레이저 설비, 다계열 통합 |

---

## 3. 공장 설비 구조

### 3.1 회의 배경 기준 전체 레이아웃

초기 회의에서는 공장 전체 레이아웃을 넓게 봤다.

| 회의 기준 Bay | 설비 구성 | 현재 처리 |
|---|---|---|
| Bay 1 | 소부재 전용 | 과제 대상 아님 |
| Bay 2 | 신 PLS 4대, LM 3대 | 데이터 적용 시 Bay 22로 대응하여 사용 |
| Bay 3 | 신 PLS 3대, NCG 1대, LM 2대 | 데이터 적용 시 Bay 23으로 대응하여 사용 |
| Bay 4 | PLS 4대, LM 2대, 향후 LSR 3대 | 현재 초안 제외 |
| Bay 5 | PLS 2대, LM 2대, 향후 LSR 2대 | 판넬 전용 성격, 현재 초안 제외 |
| Bay 6 | 신 PLS 2대, LM 2대 | 판넬 전용, 현재 초안 제외 |

회의록에는 Bay 2, Bay 3처럼 표기되어 있고, Q&A와 Excel 데이터에는 `CUT_BAY = 22`, `CUT_BAY = 23`으로 확인된다. 개발 기준은 최신 Q&A와 실제 데이터에 맞춰 Bay 22/23으로 둔다.

주의: Bay 2/3와 CUT_BAY 22/23의 명명 관계는 문서상 대응으로 처리되어 있으나, 현업에 한번 확인하면 좋다.

### 3.2 현재 개발 기준 설비

| CUT_BAY | 장비명 | 설비 타입 | 초기 모델 | 비고 |
|---|---|---|---|---|
| 22 | PLS21 | PLS | 사용 | NP 1차 후보 |
| 22 | PLS22 | PLS | 사용 | NP 1차 후보 |
| 22 | PLS23 | PLS | 사용 | NP 1차 후보 |
| 22 | PLS24 | PLS | 사용 | NP 1차 후보 |
| 23 | PLS31 | PLS | 사용 | NP 1차 후보 |
| 23 | PLS32 | PLS | 사용 | NP 1차 후보 |
| 23 | PLS33 | PLS | 사용 | NP 1차 후보 |

현재 받은 Excel 데이터도 이 매핑과 일치한다.

| CUT_BAY | 설비 |
|---|---|
| 22 | PLS21, PLS22, PLS23, PLS24 |
| 23 | PLS31, PLS32, PLS33 |

### 3.3 설비 운영 규칙

| 항목 | 확정 규칙 | 개발 처리 |
|---|---|---|
| 복수 설비 | 한 Bay에 PLS가 여러 대 있어도 각 설비는 독립 병렬 처리 | PLS21 작업 종료가 PLS22 착수 조건이 아님 |
| 설비-Bay 관계 | Bay는 설비에 따라 매핑 | `machine_to_bay` dict를 단일 기준으로 사용 |
| 정반/라인 | 각 설비에 W/O 3개까지, 3개 길이 합 55000 이하 | 같은 설비의 동시 active W/O 묶음 기준으로 구현 |
| 작업 단위 | W/O 단위 | 블록/프로젝트 정보는 grouping constraint에 사용 |

---

## 4. 데이터 스키마

입력 파일은 `절단03~04_NP물량_마스킹.xlsx`이고, 주요 컬럼은 다음과 같다.

| 컬럼 | 의미 | 개발 처리 |
|---|---|---|
| `PROJ_NO` | 프로젝트/호선 식별자 | `block_set_id` 생성 |
| `BLK_NO` | 블록명 | `block_set_id` 생성 |
| `WK_ORD_NO` | W/O 명 | `job_id` 후보 |
| `GYEL_ACT_STDT` | 계획 착수일 | 기준 계획, 학습 label, 비교값 |
| `GYEL_ACT_EDDT` | 계획 종료일 | 기준 계획, 검증값 |
| `RT_CUT_ST_DTM` | 실적 착수시간 | 실적 소요시간 산정 |
| `RT_CUT_ED_DTM` | 실적 종료시간 | 종료 < 시작이면 삭제 |
| `GYEL` | 계열 | 현재 NP 중심 |
| `LTH` | 길이 | 55000 길이합 제약 후보 컬럼 |
| `THK` | 두께 | 처리시간 분석, 레이저 확장 시 사용 |
| `CUT_LTH` | 절단 길이 | `TACT_TIME` 분석 핵심 |
| `MARK_LTH` | 마킹 길이 | 처리시간 보조 feature |
| `BVL_LTH` | 베벨 길이 | 처리시간 보조 feature |
| `STL_QTY` | 강재 수량 | 처리시간/용량 분석 후보 |
| `RT_EQP_NM` | 장비명 | `machine_id` |
| `CUT_BAY` | 절단 Bay | Bay 매핑 검증 컬럼 |
| `TACT_TIME` | 택트타임 | 그대로 신뢰하지 말고 분석 필요 |
| `ASS_ST_DT` | 후공정 소요일/후공정 관련 일자 | due date 후보, 산식 config화 |
| `PTLST_QTY` | 부재 수량 | `TACT_TIME` 분석 후보 |

---

## 5. 전처리 규칙

| ID | 항목 | Q&A 기준 | Codex 구현 |
|---|---|---|---|
| P-01 | `WK_SEQ` | 제외, 실행 순서로 해석하지 않음 | 입력에 있어도 scheduler feature로 사용하지 않음 |
| P-02 | 실적시각 역전 | `RT_CUT_ED_DTM < RT_CUT_ST_DTM`이면 삭제 | 삭제 건수 로그 기록 |
| P-03 | `STL_REQ_DT` 결측 | 모두 삭제 | 현재 파일에 없으면 존재 시에만 적용 |
| P-04 | `GYEL` | 현재 NP만 사용, 타 계열은 후속 확장 | `target_series = [NP]` |
| P-05 | `TACT_TIME` | 부재 수량, 두께, 절단 길이, 실적 시각 차이와 상관분석 후 사용 | 처리시간 산정 모델 개발 |
| P-06 | `WK_DT` | 새로운 계획 착수일 컬럼 참고 | 실적일로 직접 해석 금지 |
| P-07 | 송선단계 코드 | 후공정 소요일 기준으로 착수일 결정 | routing code mapping은 보류 |

---

## 6. 제약조건 명세

### 6.1 Hard 제약

| ID | 제약 | Q&A 기준 | 검증 관점 |
|---|---|---|---|
| C-01 | 단일 설비 배정 | 각 W/O는 정확히 하나의 설비에 배정 | `sum_m,d x[i,m,d] = 1` |
| C-02 | 설비-Bay 일관성 | 배정 Bay는 설비의 Bay와 일치 | `bay[i] = machine_to_bay[machine[i]]` |
| C-03 | 블록셋 동일 Bay | 동일 프로젝트+동일 블록명은 동일 Bay, 설비까지는 묶지 않음 | `groupby(PROJ_NO, BLK_NO)` 내 bay unique count = 1 |
| C-04 | 후공정 일정 | 지연 절대 불가, 제약으로 설정 | completion 또는 planned start <= due date |
| C-05 | 정반/라인 용량 | 각 설비에 W/O 3개까지, 3개 길이 합 55000 이하 | 같은 설비에서 시간이 겹치는 active W/O count와 LTH 합 검증 |
| C-06 | 설비/계열 우선순위 | 1/2순위가 불가능할 때만 3/4순위 fallback | 다계열 단계에서 tiered candidate policy |
| C-07 | 레이저 20T | LSR은 20T 이하만 가능, PLS는 20T 초과 우선 | 현재 초안 제외, phase 2 |

### 6.2 목적함수

| ID | 목적 | 구현 메모 |
|---|---|---|
| O-01 | Bay 부하 평준화 | Bay 22/23 간 작업량 또는 작업시간 편차 최소화 |
| O-02 | 설비 부하 평준화 | PLS21~PLS33 간 가동시간 편차 최소화 |
| O-03 | 우선순위/fallback 최소화 | NP 단일 범위에서는 영향 작고, 다계열 확장 시 중요 |
| O-04 | 후공정 일정 위반 | 목적함수가 아니라 hard constraint로 처리 |

---

## 7. Codex 구현 명세

### 7.1 권장 프로젝트 구조

Word 문서가 제안한 구조는 다음과 같다.

```text
cutting_scheduler/
  config/
    factory_current.yaml
    columns.yaml
  data/
    raw/
    interim/
    processed/
  src/
    data_loader.py
    preprocessing.py
    takt_time.py
    factory.py
    constraints.py
    scheduler_heuristic.py
    scheduler_cp.py
    evaluator.py
    export.py
  tests/
    test_preprocessing.py
    test_factory_mapping.py
    test_constraints.py
    test_scheduler_smoke.py
  main.py
```

현재 저장소는 이미 `Utils/`, `Environment/`, `Agent/`, `Train/` 구조를 갖고 있으므로 새 폴더를 그대로 만들기보다 현재 구조에 맞춰 아래처럼 매핑하는 것이 현실적이다.

| Word 제안 | 현재 저장소 대응 |
|---|---|
| `src/data_loader.py` | `Utils/cutting_data_loader.py` |
| `src/preprocessing.py` | `Utils/cutting_data_loader.py` 또는 `Utils/preprocessing.py` |
| `src/takt_time.py` | `Utils/tact_time.py` |
| `src/factory.py` | `Environment/data.py`, `Utils/cutting_scenario_builder.py` |
| `src/constraints.py` | `Environment/constraints/` |
| `src/scheduler_heuristic.py` | `Agent/heuristics.py` |
| `src/evaluator.py` | 신규 `Utils/evaluator.py` 또는 `Environment/simulation.py` |
| `src/export.py` | 신규 `Utils/export.py` |

### 7.2 factory config 핵심

Word 문서의 `factory_current.yaml` 초안에서 중요한 내용은 다음이다.

```yaml
scope:
  target_series: ["NP"]
  decision_scope: ["planned_start_date", "bay", "machine"]
  exclude_wk_seq: true

factory:
  bays:
    "22":
      machines: ["PLS21", "PLS22", "PLS23", "PLS24"]
      active: true
    "23":
      machines: ["PLS31", "PLS32", "PLS33"]
      active: true

constraints:
  block_set_same_bay:
    enabled: true
    type: hard
    group_cols: ["PROJ_NO", "BLK_NO"]
  downstream_due_date:
    enabled: true
    type: hard
    due_col: "ASS_ST_DT"
  machine_day_capacity:
    enabled: true
    type: hard
    max_wo_count: 3
    max_total_length: 55000
    length_col: "LTH"
  sequence_from_wk_seq:
    enabled: false
  ncg:
    enabled: false
  wide_plate:
    enabled: false
  crane_lm:
    enabled: false
  laser_lsr:
    enabled: false
```

현재 `config_np_100.yaml`에는 일부만 반영되어 있다. 특히 `block_set_same_bay`, `machine_day_capacity.max_total_length`, `cut_bay`와 `downstream_due_date`는 별도 구현이 필요하다.

### 7.3 표준 데이터 모델

| 객체 | 필수 필드 |
|---|---|
| Job | `job_id`, `project_no`, `block_no`, `block_set_id`, `series`, `length`, `thickness`, `cut_length`, `mark_length`, `bevel_length`, `steel_qty`, `part_qty`, `due_date`, `planned_start_original`, `machine_original`, `bay_original`, `process_time` |
| Machine | `machine_id`, `bay_id`, `machine_type`, `active`, `capacity_wo_per_day`, `capacity_length_per_day` |
| Assignment | `job_id`, `scheduled_date`, `bay_id`, `machine_id`, `process_time`, `is_original_plan`, `validation_status` |
| ValidationReport | `num_due_violations`, `num_blockset_bay_violations`, `num_machine_count_violations`, `num_machine_length_violations`, `bay_load_std`, `machine_load_std` |

현재 코드의 `Job`, `Machine`, `ScheduledOperation`에는 위 필드가 전부 있지는 않다. 특히 `Machine.bay_id`, `Job.block_set_id`, `Job.cut_bay`, `ScheduledOperation.scheduled_date` 계열은 추가 검토가 필요하다.

---

## 8. 개발 필요 항목

Word 문서의 개발 항목을 현재 코드 기준으로 다시 정리하면 다음 순서가 맞다.

| 우선순위 | 항목 | 현재 코드 작업 |
|---:|---|---|
| 1 | 입력 데이터 로더 | `Utils/cutting_data_loader.py` 생성 |
| 2 | 전처리 파이프라인 | NP filtering, 날짜 파싱, 역전 시간 삭제, 제외 로그 |
| 3 | 공장 설비 config | Bay 22/23, PLS21~33, `machine_to_bay` |
| 4 | scenario builder | Excel/CSV를 현재 환경 YAML로 변환 |
| 5 | `TACT_TIME` 분석 | `Utils/tact_time.py` 확장 |
| 6 | 제약 검증기 | 후공정 일정, 블록셋 동일 Bay, 설비당 3 W/O, 길이합 55000 |
| 7 | Baseline 휴리스틱 | EDD, SPT, LPT, 최소부하 설비 |
| 8 | 평가 지표 | Bay 부하, 설비 부하, violation count, 기존계획 대비 변동량 |
| 9 | 결과 출력 | job schedule, machine-day summary, bay-day summary, violation report |
| 10 | 최적화 모델 | CP-SAT 또는 MILP, 후속 |
| 11 | RL 환경 | 휴리스틱/검증기 이후 확장 |
| 12 | 레이저/다계열 확장 | phase 2 |

---

## 9. 테스트 기준

| ID | 테스트 | 통과 기준 |
|---|---|---|
| T-01 | 설비 매핑 | PLS21~PLS24는 Bay 22, PLS31~PLS33은 Bay 23 |
| T-02 | `WK_SEQ` 미사용 | 입력에 있어도 스케줄 결정에 직접 사용되지 않음 |
| T-03 | 블록셋 동일 Bay | 동일 `PROJ_NO + BLK_NO` 그룹 내 Bay가 2개 이상이면 fail |
| T-04 | 후공정 일정 | due date 이후 배정 W/O가 있으면 fail |
| T-05 | 설비 용량 | 같은 설비에서 동시에 active인 W/O가 4개 이상 또는 LTH 합 55000 초과이면 fail |
| T-06 | 제외 항목 | NCG/LSR/광폭/크레인 제약이 초기 config에서 꺼져 있어야 함 |
| T-07 | 부하평준화 평가 | Bay/machine load std가 산출되어야 함 |
| T-08 | `TACT_TIME` 리포트 | `TACT_TIME`과 실적시간/절단길이/두께/부재수량 상관계수 산출 |

---

## 10. 산출물 포맷

| 산출물 | 필수 포함 내용 |
|---|---|
| `job_schedule.csv/xlsx` | `job_id`, `project_no`, `block_no`, `block_set_id`, `scheduled_date`, `bay_id`, `machine_id`, `process_time`, `due_date`, `validation_flags` |
| `machine_day_summary.csv/xlsx` | `date`, `machine_id`, `bay_id`, `num_jobs`, `total_length`, `total_process_time`, `capacity_violated` |
| `bay_day_summary.csv/xlsx` | `date`, `bay_id`, `num_jobs`, `total_length`, `total_process_time` |
| `validation_report.json/csv` | hard constraint별 위반 건수와 위반 W/O 리스트 |
| `metrics.json/csv` | `bay_load_std`, `machine_load_std`, `max_machine_load`, `min_machine_load`, `total_jobs`, `infeasible_count` |

---

## 11. 원문 Q&A 핵심 매핑

| Q&A 항목 | 최종 정리 |
|---|---|
| `WK_SEQ` | 제외, 착수일/Bay/장비를 결정 |
| 실적 시각 이상치 | 종료시각이 시작시각보다 이른 데이터는 삭제 |
| 송선단계 | 후공정 소요일 기준으로 착수일 결정, 필요 시 후공정 매핑 고려 |
| `STL_REQ_DT` 결측 | 모두 삭제 |
| `GYEL` | NP 알고리즘 초안 먼저 개발 |
| `TACT_TIME` | 부정확 가능성, 상관관계 분석 요청 |
| 설비/계열 우선순위 | penalty 아님, 1/2순위 불가 시 3/4순위 |
| 레이저 | LSR은 20T 이하만, PLS는 20T 초과 우선이나 현재 초안 제외 |
| 광폭 | 고려하지 않음 |
| NCG | 고려하지 않음 |
| 판넬 전용 Bay | 25, TRANS Bay가 판넬 전용, FL 1, FN 2, soft 제약 |
| 한 Bay 내 복수 설비 | 각 설비는 독립 운행 |
| 정반 라인 길이 | 각 설비 W/O 3개까지, 3개 길이 합 55000 이하 |
| 블록셋 | 동일 프로젝트+동일 블록명은 동일 Bay, 설비까지는 묶지 않음 |
| 크레인 | 현재 고려하지 않음 |
| 후공정 일정 | 제약으로 설정 |
| 목적함수 | Bay 부하, 설비 부하, 계열/물성 우선순위. 가중치는 결과 보며 조정 |
| 공정/설비 기준 | 먼저 22, 23 Bay 설비만 고려 |

---

## 12. 검토 의견

Word 정리본의 방향은 현재 코드 개발 기준으로 타당하다. 특히 “Bay 22/23 + PLS 7대 + NP”로 범위를 자른 점이 중요하다. 이 범위를 넘어서 Bay 1~6 전체, 레이저, NCG, 크레인까지 한번에 넣으면 문제 정의가 다시 흔들린다.

다만 현재 코드와 Word 명세 사이에 차이가 있다.

- 현재 `input/np_100_scenario.yaml`은 `CUT_BAY`를 임시로 `downstream_bay`에 넣고 있다. 명세 기준으로는 `cut_bay`와 `downstream_bay/due_date`를 분리해야 한다.
- 현재 `Machine`에는 `bay_id`가 없다. 명세 기준으로는 `machine_to_bay` 또는 `Machine.bay_id`가 필요하다.
- 현재 `parallel_capacity=3`은 “동시 슬롯 3개”처럼 동작하지만, 길이합 55000까지 막지는 못한다. 3 W/O/55000은 machine-date가 아니라 같은 설비의 동시 active W/O 묶음 제약으로 구현한다.
- 후공정 일정의 hard 기준은 확정이지만, `ASS_ST_DT`를 완료 마감일로 볼지 착수 기준으로 볼지 산식은 config화가 필요하다.
- CP-SAT/MILP는 바로 넣기보다, data loader, scenario builder, validator, heuristic baseline이 먼저다.

---

## 13. Codex 다음 작업 지시문

다음 코드 개발은 아래 순서로 진행하는 것이 맞다.

1. `Utils/cutting_data_loader.py` 생성
2. `Utils/cutting_scenario_builder.py` 생성
3. `Machine.bay_id` 또는 `machine_to_bay` 구조 추가
4. `Job.block_set_id`, `Job.cut_bay`, `Job.source_machine_id` 추가 검토
5. `CUT_BAY`와 `downstream_bay` 분리
6. `TACT_TIME` 분석 리포트 구현
7. 동일 블록셋 동일 Bay validator 구현
8. 동시 active W/O 묶음 기준 W/O 3개 및 LTH 합 55000 validator 구현
9. SPT/load_balance/EDD/최소부하 휴리스틱 결과를 공통 포맷으로 저장
10. 그 다음 CP-SAT/MILP 또는 Gymnasium wrapper 확장

---

## 14. 최종 정리

현재 공장 이해는 다음 한 문장으로 고정하면 된다.

> 지금 개발할 것은 전체 공장 DES가 아니라, NP W/O를 Bay 22/23의 PLS 7대에 배정하는 초기 절단 스케줄링 문제다. Q&A로 확정된 hard 제약은 후공정 일정, 블록셋 동일 Bay, 설비당 W/O 3개 및 길이합 55000이며, `TACT_TIME`은 분석 후 처리시간으로 써야 한다.
