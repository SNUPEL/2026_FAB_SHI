**절단 공정 Q&A 기반 개발 명세 정리본**

> 초기 NP 단계의 Q&A와 원본 자료를 통합한 근거 archive다. 현재 MIXED 실행 계약은
> 루트의 `다계열_Phase1_정책_및_합성데이터_생성_계약.md`를 우선한다.

근거 원본: `절단 문의사항 정리_05.14.docx`, `절단03~04_NP물량_마스킹.xlsx`

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>문서 목적<br />
</strong>Codex 또는 개발자가 바로 구현할 수 있도록 공장 설비 범위, 데이터 해석, 제약조건, 목적함수, 전처리 규칙, 구현 모듈, 검증 기준을 하나의 개발 명세로 재구성하였다.<br />
자료 간 내용이 충돌할 경우 최종 질의응답 결과인 「절단 문의사항 정리_05.14.docx」를 우선 기준으로 한다.</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

# **0. 색상 및 상태 구분**

| **상태**      | **의미**                                         | **개발 처리**                       |
|---------------|--------------------------------------------------|-------------------------------------|
| **확정**      | 현업 답변 또는 Q&A로 확정된 기준                 | 코드에 직접 반영                    |
| **Hard**      | 위반 불가 제약                                   | 검증기에서 반드시 fail 처리         |
| **개발 필요** | 구현해야 할 로직/모듈/검증 항목                  | Codex 작업 항목으로 분리            |
| **분석 필요** | 데이터 기반으로 산정/검증해야 하는 항목          | 상관분석, 이상치 분석, 회귀/룰 산출 |
| **보류/제외** | 현재 NP 초안에서는 제외하거나 후속 단계에서 반영 | config flag로 비활성화              |
| **배경**      | 초기 회의/협의자료의 참고 내용                   | 최종 Q&A와 충돌하면 Q&A 우선        |

# **1. 최종 결론: 현재 개발해야 하는 문제**

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>현재 기준 핵심 문장<br />
</strong>현재 개발 범위는 “NP 계열 W/O를 대상으로, Bay 22/23에 배치된 7대 PLS 설비에 대해 계획 착수일·Bay·장비를 결정하고, 후공정 일정과 블록셋 동일 Bay 제약을 만족하면서 Bay/설비 부하를 평준화하는 초기 절단 스케줄링 문제”이다.</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

| **구분**           | **내용**                                                        | **최종 처리**                                               |
|--------------------|-----------------------------------------------------------------|-------------------------------------------------------------|
| **대상 데이터**    | 절단03~04_NP물량_마스킹.xlsx                                    | **현재 NP 중심 데이터. Q&A 기준 NP 알고리즘 초안부터 개발** |
| **의사결정**       | 계획 착수일, Bay, 장비                                          | **WK_SEQ를 작업 순서로 쓰지 않는다. 작업 순서 제외**        |
| **대상 Bay**       | Bay 22, Bay 23                                                  | **Q&A: 먼저 22, 23 Bay 설비만 고려**                        |
| **대상 설비**      | PLS21, PLS22, PLS23, PLS24, PLS31, PLS32, PLS33                 | **총 7대 플라즈마 절단 설비**                               |
| **문제 유형**      | 병렬기계 스케줄링 / 설비 할당 + 일자 배정                       | **각 설비는 독립적으로 운행됨**                             |
| **핵심 Hard 제약** | 후공정 일정 준수, 블록셋 동일 Bay, 설비당 W/O 3개/길이합 55,000 | **검증기에서 위반 여부를 별도로 출력**                      |
| **초기 제외**      | NCG, 광폭, 크레인 과부하, 적재순서, 레이저 설비, 다계열 통합    | **보류 항목은 config에서 비활성화**                         |

# **2. 기본 공장 설비 내용**

## **2.1 전체 회의 기준 레이아웃: 배경 정보**

| **회의 기준 Bay** | **설비 구성**                  | **현재 문서상 처리**                                                       |
|-------------------|--------------------------------|----------------------------------------------------------------------------|
| **Bay 1**         | 소부재 전용                    | **과제 대상 아님**                                                         |
| **Bay 2**         | 신 PLS 4대, LM 3대             | **초기 회의상 소조/중조/대조 대상. 데이터 적용 시 Bay 22로 대응하여 사용** |
| **Bay 3**         | 신 PLS 3대, NCG 1대, LM 2대    | **초기 회의상 NCG→LSR 예정. 데이터 적용 시 Bay 23으로 대응하여 사용**      |
| **Bay 4**         | PLS 4대, LM 2대 / 향후 LSR 3대 | **현재 초안 제외**                                                         |
| **Bay 5**         | PLS 2대, LM 2대 / 향후 LSR 2대 | **판넬 전용 성격. 현재 초안 제외**                                         |
| **Bay 6**         | 신 PLS 2대, LM 2대             | **판넬 전용. 현재 초안 제외**                                              |

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>표기 정리<br />
</strong>회의록/협의자료에는 Bay 2, Bay 3처럼 표기된 부분이 있고, Q&amp;A와 엑셀 데이터에서는 CUT_BAY 22, 23으로 확인된다. 개발 명세에서는 실제 데이터와 최종 Q&amp;A를 기준으로 Bay 22/23 표기를 사용한다.</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## **2.2 현재 개발 기준 설비: Codex가 사용할 확정 매핑**

| **CUT_BAY** | **장비명** | **설비 타입** | **초기 모델** | **비고**      |
|-------------|------------|---------------|---------------|---------------|
| **22**      | PLS21      | PLS           | **사용**      | NP 1순위 후보 |
| **22**      | PLS22      | PLS           | **사용**      | NP 1순위 후보 |
| **22**      | PLS23      | PLS           | **사용**      | NP 1순위 후보 |
| **22**      | PLS24      | PLS           | **사용**      | NP 1순위 후보 |
| **23**      | PLS31      | PLS           | **사용**      | NP 1순위 후보 |
| **23**      | PLS32      | PLS           | **사용**      | NP 1순위 후보 |
| **23**      | PLS33      | PLS           | **사용**      | NP 1순위 후보 |

## **2.3 설비 운영 규칙**

| **항목**          | **확정 규칙**                                              | **개발 주석**                                   |
|-------------------|------------------------------------------------------------|-------------------------------------------------|
| **복수 설비**     | **한 Bay에 PLS가 여러 대 있어도 각 설비는 독립 병렬 처리** | PLS21 작업 종료가 PLS22 착수 조건이 아님        |
| **설비-Bay 관계** | **Bay는 설비에 따라 매핑**                                 | machine_to_bay 딕셔너리를 단일 기준으로 사용    |
| **정반/라인**     | **각 설비에 W/O 3개까지, 3개 길이 합 55,000 이하**         | 같은 설비의 동시 active W/O 묶음 기준으로 구현              |
| **작업 단위**     | **W/O 단위**                                               | 블록/프로젝트 정보는 grouping constraint에 사용 |

# **3. 데이터 스키마와 전처리 기준**

## **3.1 입력 파일 및 주요 컬럼**

| **컬럼**          | **의미**                       | **개발 처리**                           |
|-------------------|--------------------------------|-----------------------------------------|
| **PROJ_NO**       | 프로젝트/호선 식별자           | block_set_id 생성에 사용                |
| **BLK_NO**        | 블록명                         | block_set_id 생성에 사용                |
| **WK_ORD_NO**     | W/O 명                         | job_id로 사용 가능                      |
| **GYEL_ACT_STDT** | 계획 착수일                    | 기준 계획/학습 label/비교값             |
| **GYEL_ACT_EDDT** | 계획 종료일                    | 기준 계획/검증값                        |
| **RT_CUT_ST_DTM** | 실적 착수시간                  | 실적 소요시간 산정용                    |
| **RT_CUT_ED_DTM** | 실적 종료시간                  | **RT_CUT_ED \< RT_CUT_ST이면 삭제**     |
| **GYEL**          | 계열                           | 현재 NP 중심. 타 계열은 후속            |
| **LTH**           | 길이                           | 정반 55,000 길이합 제약 후보 컬럼       |
| **THK**           | 두께                           | 레이저 20T 제약/처리시간 분석 후보      |
| **CUT_LTH**       | 절단 길이                      | TACT_TIME/실적시간 상관분석 핵심        |
| **MARK_LTH**      | 마킹 길이                      | 처리시간 보조 피처                      |
| **BVL_LTH**       | 베벨 길이                      | 처리시간 보조 피처                      |
| **STL_QTY**       | 강재 수량                      | 처리시간/용량 분석 후보                 |
| **RT_EQP_NM**     | 장비명                         | machine_id                              |
| **CUT_BAY**       | 절단 Bay                       | Bay 매핑 검증 컬럼                      |
| **TACT_TIME**     | 택트타임                       | **그대로 신뢰하지 말고 상관분석 필요**  |
| **ASS_ST_DT**     | 후공정 소요일/후공정 관련 일자 | **due-date 후보. 최종 산식은 config화** |
| **PTLST_QTY**     | 부재 수량                      | TACT_TIME/실적시간 분석 후보            |

## **3.2 전처리 규칙**

| **ID**   | **항목**            | **Q&A 기준**                                                   | **Codex 구현**                                    |
|----------|---------------------|----------------------------------------------------------------|---------------------------------------------------|
| **P-01** | **WK_SEQ**          | **제외. 실행 순서로 해석하지 않음**                            | 입력에 있더라도 scheduler feature로 사용하지 않음 |
| **P-02** | **실적시각 역전**   | **RT_CUT_ED_DTM \< RT_CUT_ST_DTM이면 삭제**                    | 데이터 클리닝 로그에 삭제 건수 기록               |
| **P-03** | **STL_REQ_DT 결측** | **모두 삭제**                                                  | 현재 파일에 해당 컬럼이 없다면 존재 시에만 적용   |
| **P-04** | **GYEL**            | **현재 NP만 사용. 타 계열은 후속 데이터 전달 후 확장**         | config.target_series=\[NP\]                       |
| **P-05** | **TACT_TIME**       | **부재 수량·두께·절단 길이·실적 시각 차이와 상관분석 후 사용** | p_i 산정 모델 개발 필요                           |
| **P-06** | **WK_DT**           | **새로운 계획 착수일 컬럼 참고**                               | 실적일로 직접 해석 금지                           |
| **P-07** | **송선단계 코드**   | **후공정 소요일 기준으로 착수일 결정. 필요 시 추후 매핑**      | routing code mapping은 보류 모듈로 분리           |

# **4. 제약조건 및 목적함수 명세**

| **ID**   | **제약**               | **Q&A 기준**                                                          | **수식/검증 관점**                                              |
|----------|------------------------|-----------------------------------------------------------------------|-----------------------------------------------------------------|
| **C-01** | **단일 설비 배정**     | **각 W/O는 정확히 하나의 설비에 배정**                                | sum_m,d x\[i,m,d\] = 1                                          |
| **C-02** | **설비-Bay 일관성**    | **배정 Bay는 설비의 Bay와 일치**                                      | bay\[i\] = machine_to_bay\[machine\[i\]\]                       |
| **C-03** | **블록셋 동일 Bay**    | **동일 프로젝트 + 동일 블록명은 동일 Bay 배정. 설비까지 묶지는 않음** | groupby(PROJ_NO, BLK_NO) 내 bay unique count = 1                |
| **C-04** | **후공정 일정**        | **지연 절대 불가. 제약으로 설정**                                     | planned_start 또는 completion \<= due_date. due 산식은 config화 |
| **C-05** | **정반/라인 용량**     | **각 설비에 W/O 3개까지 작업 가능, 3개 길이 합 55,000 이하**          | for each concurrent machine interval: active_count\<=3 and active_sum(LTH)\<=55000            |
| **C-06** | **설비/계열 우선순위** | **페널티가 아니라 1·2순위가 가득 찬 경우 3·4순위 fallback**           | 후속 다계열 단계에서 tiered candidate policy로 구현             |
| **C-07** | **레이저 20T**         | **LSR은 20T 이하만 가능. PLS는 20T 초과 우선**                        | 현재 초안 제외. phase-2에서 활성화                              |

| **ID**   | **목적**                     | **정리**                                          | **구현 메모**                                            |
|----------|------------------------------|---------------------------------------------------|----------------------------------------------------------|
| **O-01** | **Bay 부하 평준화**          | **Bay 22/23 간 작업량 또는 작업시간 편차 최소화** | variance 또는 max-min 사용                               |
| **O-02** | **설비 부하 평준화**         | **PLS21~PLS33 간 가동시간 편차 최소화**           | machine load variance 사용                               |
| **O-03** | **우선순위/fallback 최소화** | **1·2순위 불가 시에만 3·4순위 사용**              | 현재 NP 단일 범위에서는 영향 작음. 후속 다계열 확장 필요 |
| **O-04** | **후공정 일정 위반**         | **목적함수 페널티가 아니라 hard constraint**      | objective에 넣지 말고 validator에서 hard fail            |

# **5. Codex용 구현 명세**

## **5.1 권장 프로젝트 구조**

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th>cutting_scheduler/<br />
config/<br />
factory_current.yaml # Bay 22/23, PLS21~PLS33, constraints 활성화 상태<br />
columns.yaml # 원천 컬럼명, 표준 컬럼명, dtype<br />
data/<br />
raw/ # 원천 xlsx<br />
interim/ # 전처리 중간 산출물<br />
processed/ # scheduler 입력 parquet/csv<br />
src/<br />
data_loader.py # xlsx 로딩, 컬럼 표준화, 날짜 파싱<br />
preprocessing.py # 이상치 삭제, NP filtering, block_set_id 생성<br />
takt_time.py # TACT_TIME/실적시간 상관분석 및 처리시간 산정<br />
factory.py # Machine, Bay, FactoryConfig dataclass<br />
constraints.py # hard constraint validator<br />
scheduler_heuristic.py # rule-based baseline<br />
scheduler_cp.py # CP-SAT/MILP 구현 선택지<br />
evaluator.py # 부하평준화, 위반 건수, fallback 사용량 계산<br />
export.py # 결과 csv/xlsx/plot 저장<br />
tests/<br />
test_preprocessing.py<br />
test_factory_mapping.py<br />
test_constraints.py<br />
test_scheduler_smoke.py<br />
main.py</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## **5.2 factory_current.yaml 초안**

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th>project:<br />
name: cutting_np_initial<br />
source_priority: ["절단 문의사항 정리_05.14.docx", "절단03~04_NP물량_마스킹.xlsx", "회의록/협의자료"]<br />
<br />
scope:<br />
target_series: ["NP"]<br />
decision_scope: ["planned_start_date", "bay", "machine"]<br />
exclude_wk_seq: true<br />
<br />
factory:<br />
bays:<br />
"22":<br />
machines: ["PLS21", "PLS22", "PLS23", "PLS24"]<br />
active: true<br />
"23":<br />
machines: ["PLS31", "PLS32", "PLS33"]<br />
active: true<br />
machine_type:<br />
PLS21: "PLS"<br />
PLS22: "PLS"<br />
PLS23: "PLS"<br />
PLS24: "PLS"<br />
PLS31: "PLS"<br />
PLS32: "PLS"<br />
PLS33: "PLS"<br />
<br />
constraints:<br />
block_set_same_bay:<br />
enabled: true<br />
type: hard<br />
group_cols: ["PROJ_NO", "BLK_NO"]<br />
downstream_due_date:<br />
enabled: true<br />
type: hard<br />
due_col: "ASS_ST_DT"<br />
assumption: "same-day cutting unless processing-time model says otherwise"<br />
machine_day_capacity:<br />
enabled: true<br />
type: hard<br />
max_wo_count: 3<br />
max_total_length: 55000<br />
length_col: "LTH"<br />
sequence_from_wk_seq:<br />
enabled: false<br />
ncg:<br />
enabled: false<br />
wide_plate:<br />
enabled: false<br />
crane_lm:<br />
enabled: false<br />
laser_lsr:<br />
enabled: false<br />
<br />
objective:<br />
bay_load_balance:<br />
enabled: true<br />
metric: "variance"<br />
machine_load_balance:<br />
enabled: true<br />
metric: "variance"<br />
fallback_priority:<br />
enabled: false # NP 초기 범위에서는 영향 작음. 다계열 확장 시 활성화</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## **5.3 표준 데이터 모델**

| **객체**             | **필수 필드**                                                                                                                                                                                                     |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Job**              | job_id, project_no, block_no, block_set_id, series, length, thickness, cut_length, mark_length, bevel_length, steel_qty, part_qty, due_date, planned_start_original, machine_original, bay_original, process_time |
| **Machine**          | machine_id, bay_id, machine_type, active, capacity_wo_per_day, capacity_length_per_day                                                                                                                            |
| **Assignment**       | job_id, scheduled_date, bay_id, machine_id, process_time, is_original_plan, validation_status                                                                                                                     |
| **ValidationReport** | num_due_violations, num_blockset_bay_violations, num_machine_count_violations, num_machine_length_violations, bay_load_std, machine_load_std                                                                      |

## **5.4 스케줄러 구현 흐름**

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th>def run_scheduler(input_xlsx, config):<br />
raw = load_xlsx(input_xlsx)<br />
df = standardize_columns(raw)<br />
df = clean_records(df,<br />
remove_reversed_actual_time=True,<br />
remove_missing_stl_req_dt_if_exists=True,<br />
target_series=["NP"],<br />
ignore_wk_seq=True,<br />
)<br />
df["block_set_id"] = df["PROJ_NO"].astype(str) + "__" + df["BLK_NO"].astype(str)<br />
df["due_date"] = derive_due_date(df, config.constraints.downstream_due_date)<br />
df["process_time"] = estimate_process_time(df) # 초기: TACT_TIME 또는 실적 기반 실험값<br />
<br />
factory = FactoryConfig.from_yaml(config)<br />
candidates = build_candidate_machines(df, factory) # 초기 NP: 7대 PLS 모두 후보<br />
<br />
schedule = solve_assignment(<br />
jobs=df,<br />
candidates=candidates,<br />
hard_constraints=[<br />
one_machine_per_job,<br />
machine_bay_consistency,<br />
block_set_same_bay,<br />
downstream_due_date,<br />
machine_day_capacity_count,<br />
machine_day_capacity_length,<br />
],<br />
objective=[bay_load_balance, machine_load_balance],<br />
)<br />
<br />
report = validate_schedule(schedule, df, factory, config)<br />
export_outputs(schedule, report)<br />
return schedule, report</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

# **6. 개발 필요 내용**

| **ID**     | **개발 항목**          | **내용**                                                                                 | **우선순위**  |
|------------|------------------------|------------------------------------------------------------------------------------------|---------------|
| **DEV-01** | **입력 데이터 로더**   | **xlsx 로딩, 컬럼명 표준화, 날짜/시간 파싱, 필수 컬럼 검증**                             | **필수**      |
| **DEV-02** | **전처리 파이프라인**  | **역전 실적시각 삭제, NP 필터링, 결측 처리, block_set_id 생성**                          | **필수**      |
| **DEV-03** | **공장 설비 config**   | **Bay 22/23, PLS21~PLS33, machine_to_bay, 활성화/비활성화 제약 관리**                    | **필수**      |
| **DEV-04** | **TACT_TIME 분석**     | **TACT_TIME vs 실적시간, CUT_LTH, THK, PTLST_QTY, LTH 상관분석 및 처리시간 산정식 도출** | **필수/분석** |
| **DEV-05** | **제약 검증기**        | **후공정 일정, 블록셋 동일 Bay, 설비당 3 W/O, 길이합 55,000 위반 검출**                  | **필수**      |
| **DEV-06** | **Baseline 휴리스틱**  | **EDD/SPT/LPT/최소부하 설비 배정 등 비교 기준 구현**                                     | **필수**      |
| **DEV-07** | **최적화 모델**        | **CP-SAT 또는 MILP로 x\[i,m,d\] 할당 변수 기반 모델 구현**                               | **권장**      |
| **DEV-08** | **평가 지표**          | **Bay 부하 표준편차, 설비 부하 표준편차, hard violation count, 기존계획 대비 변동량**    | **필수**      |
| **DEV-09** | **결과 출력**          | **job-level schedule, machine-day summary, bay-day summary, violation report 저장**      | **필수**      |
| **DEV-10** | **시각화**             | **Bay/설비별 부하 그래프, 일자별 투입량, 위반 리포트 테이블**                            | **권장**      |
| **DEV-11** | **RL 환경**            | **휴리스틱/CP 기준 이후, state/action/reward 환경으로 확장**                             | **후속**      |
| **DEV-12** | **레이저/다계열 확장** | **LSR 20T hard, PLS 20T 초과 우선, 계열별 우선순위 fallback**                            | **후속**      |

# **7. 검증 기준 및 테스트 케이스**

| **ID**   | **테스트**                 | **통과 기준**                                                         | **상태** |
|----------|----------------------------|-----------------------------------------------------------------------|----------|
| **T-01** | **설비 매핑 테스트**       | **PLS21~PLS24는 Bay 22, PLS31~PLS33은 Bay 23으로 매핑되어야 함**      |          |
| **T-02** | **WK_SEQ 미사용 테스트**   | **입력에 WK_SEQ가 있어도 출력 스케줄 결정에 직접 사용되지 않아야 함** |          |
| **T-03** | **블록셋 동일 Bay 테스트** | **동일 PROJ_NO+BLK_NO 그룹 내 배정 Bay가 2개 이상이면 fail**          |          |
| **T-04** | **후공정 일정 테스트**     | **due_date 이후 배정된 W/O가 있으면 fail**                            |          |
| **T-05** | **설비 용량 테스트**       | **같은 설비에서 동시에 active인 W/O가 4개 이상 또는 LTH 합 55,000 초과이면 fail** |          |
| **T-06** | **제외 항목 테스트**       | **NCG/LSR/광폭/크레인 제약이 초기 config에서 활성화되지 않아야 함**   |          |
| **T-07** | **부하평준화 평가**        | **기존계획 또는 단순 rule 대비 Bay/machine load std가 산출되어야 함** |          |
| **T-08** | **TACT_TIME 리포트**       | **TACT_TIME과 실적시간/절단길이/두께/부재수량 상관계수 테이블 산출**  |          |

# **8. 산출물 포맷**

| **산출물**                       | **필수 포함 내용**                                                                                                       |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| **job_schedule.csv/xlsx**        | job_id, project_no, block_no, block_set_id, scheduled_date, bay_id, machine_id, process_time, due_date, validation_flags |
| **machine_day_summary.csv/xlsx** | date, machine_id, bay_id, num_jobs, total_length, total_process_time, capacity_violated                                  |
| **bay_day_summary.csv/xlsx**     | date, bay_id, num_jobs, total_length, total_process_time                                                                 |
| **validation_report.json/csv**   | hard constraint별 위반 건수와 위반 W/O 리스트                                                                            |
| **metrics.json/csv**             | bay_load_std, machine_load_std, max_machine_load, min_machine_load, total_jobs, infeasible_count                         |

# **9. 원문 Q&A 핵심 매핑**

| **Q&A 항목**            | **최종 정리**                                                                            |
|-------------------------|------------------------------------------------------------------------------------------|
| **WK_SEQ**              | **WK_SEQ 컬럼 제외. 현재 작업 순서는 제외하고 착수일, Bay, 장비를 결정**                 |
| **실적 시각 이상치**    | **RT_CUT_ED_DTM이 RT_CUT_ST_DTM보다 이른 데이터는 일단 모두 삭제**                       |
| **송선단계**            | **후공정 소요일 기준으로 착수일 결정. 필요 시 추후 후공정 매핑 고려**                    |
| **STL_REQ_DT 결측**     | **모두 삭제**                                                                            |
| **GYEL**                | **현재 NP 계열 하나에 대해서만 포함. NP 알고리즘 초안 개발 먼저 진행**                   |
| **TACT_TIME**           | **모수도 적고 부정확. 부재 수량, 두께, 절단 길이와 상관관계 분석 요청**                  |
| **설비/계열 우선순위**  | **페널티 아님. 1·2순위가 가득 차서 불가할 때 3·4순위 배정**                              |
| **레이저**              | **LSR은 20T 이하만. PLS는 20T 초과 우선이나 현재 초안 제외**                             |
| **광폭**                | **광폭 고려 X**                                                                          |
| **NCG**                 | **NCG 고려 X**                                                                           |
| **판넬 전용 Bay**       | **25, TRANS Bay가 판넬 전용. FL 1, FN 2. soft 제약**                                     |
| **한 Bay 내 복수 설비** | **각 설비가 독립적으로 운행됨**                                                          |
| **정반 라인 길이**      | **각 설비에 W/O 3개까지, 3개 길이 합 55,000 이하**                                       |
| **블록셋**              | **동일 프로젝트, 동일 블록명. 동일 Bay로 배정, 설비까지는 X**                            |
| **크레인**              | **현재 고려 X**                                                                          |
| **후공정 일정**         | **제약으로 설정**                                                                        |
| **목적함수**            | **Bay별 부하 평준화, 설비별 부하 평준화, 계열/물성 우선순위. 가중치는 결과 보면서 조정** |
| **공정/설비 기준**      | **먼저 22, 23 Bay 설비만 고려. 추후 회의로 정리**                                        |

# **10. Codex 작업 지시문 예시**

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th>작업 목표:<br />
절단03~04_NP물량_마스킹.xlsx를 입력으로 받아 NP 계열 W/O에 대해<br />
Bay 22/23 및 PLS21~PLS33 설비 배정 스케줄을 생성한다.<br />
<br />
반드시 지킬 것:<br />
1) WK_SEQ는 사용하지 않는다.<br />
2) 초기 factory scope는 Bay 22/23과 PLS21,22,23,24,31,32,33으로 제한한다.<br />
3) 동일 PROJ_NO + BLK_NO는 동일 Bay에 배정한다. 단, 동일 설비까지 강제하지 않는다.<br />
4) 후공정 일정은 hard constraint로 둔다.<br />
5) 같은 설비에서 동시에 active인 W/O는 3개 이하, LTH 합 55,000 이하를 만족한다.<br />
6) NCG, LSR, 광폭, 크레인, 적재순서는 초기 버전에서 구현하지 않는다.<br />
7) TACT_TIME은 원천값을 맹신하지 말고 분석 리포트를 생성한다.<br />
<br />
우선 구현 순서:<br />
A. data_loader.py / preprocessing.py<br />
B. factory_current.yaml / factory.py<br />
C. constraints.py validator<br />
D. scheduler_heuristic.py baseline<br />
E. evaluator.py / export.py<br />
F. CP-SAT 또는 MILP 모델</th>
</tr>
</thead>
<tbody>
</tbody>
</table>
