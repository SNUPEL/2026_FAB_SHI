# AGENTS.md

이 저장소에서 Codex가 작업할 때 반드시 따를 규칙이다. 목표는 빠른 임시 구현이 아니라, 실제 데이터로 검증 가능한 절단 공장 DES 환경을 안정적으로 만드는 것이다.

---

## 1. 최우선 기준

- 작업 전 반드시 `현재과제_진행체크리스트.md`를 먼저 읽는다.
- `현재과제_진행체크리스트.md`를 단일 master checklist로 사용한다.
- 새 메일, 새 데이터, 새 Q&A가 들어오면 코드보다 master checklist를 먼저 갱신한다.
- 기존 md 문서는 근거 확인용 archive다. 진행 상태와 다음 작업은 master checklist에만 기록한다.
- 새 md 파일은 원문 정리, 외부 근거, 긴 분석 산출물이 아니면 만들지 않는다.
- 역할분담 문서는 사용하지 않는다.

---

## 2. 개발 태도

- 대충 구현하지 않는다.
- 코드를 바꾸기 전에 왜 바꾸는지, 무엇이 깨질 수 있는지, 더 단순한 방법이 있는지 최소 두 번 검토한다.
- 변경 이유를 코드 구조와 데이터 기준으로 설명할 수 있어야 한다.
- 현재 데이터로 검증할 수 없는 기능은 “구현 완료”로 표시하지 않는다.
- 없는 데이터를 임의 생성하거나 추정해서 정상처럼 보이게 하지 않는다.
- 결과가 이상하면 성능 개선보다 원인 추적을 먼저 한다.
- 기존 동작을 바꿀 때는 100건 기준 회귀검증을 먼저 통과시킨다.

---

## 3. 작업 루프와 스킬셋 적용

두 외부 글의 핵심을 이 저장소 방식에 맞게 적용한다. 원문 방식처럼 별도 `plan/` 폴더를 늘리지 않고, 이 프로젝트의 단일 master checklist인 `현재과제_진행체크리스트.md`에 계획과 진행 상태를 통합한다.

### 3.1 기본 작업 루프

- 기본은 현재 Codex 세션 하나에서 처리한다.
- 작업은 `이해 -> 체크리스트 갱신 -> 구현 -> 검증 -> 리뷰 -> 체크리스트 반영 -> 보고` 순서로 진행한다.
- 사용자가 코드 개발을 요구하면 단순 설명에서 멈추지 않고 가능한 범위까지 직접 구현하고 검증한다.
- 목표가 끝나지 않았으면 같은 루프를 반복한다.
- 반복은 무한정 돌리지 않는다. 성공 조건, 차단 조건, 시간/비용/권한 한계, 사용자 중단 지시 중 하나가 발생하면 멈추고 상태를 보고한다.
- 중간 결과가 이상하면 새 기능을 더하지 말고 원인 분석 루프로 되돌아간다.

### 3.2 Plan-first 적용

- 큰 변경은 구현 전에 현재 이해, 변경 범위, 깨질 수 있는 지점, 검증 명령을 먼저 정리한다.
- 계획 파일은 새로 만들지 않고 `현재과제_진행체크리스트.md`의 해당 섹션에 반영한다.
- 계획이 불명확해도 멈추지 말고, 로컬 파일과 데이터로 확인 가능한 것부터 조사한다.
- 사용자의 명시 승인이 필요한 위험 변경은 먼저 설명한다. 예: 대규모 삭제, 데이터 재생성, 제약 의미 변경, output 정리.
- 승인된 뒤에는 채팅 코드블록만 남기지 말고 실제 파일을 수정한다.

### 3.3 서브에이전트 사용 기준

- 서브에이전트는 사용자가 명시적으로 `/sub`, 서브에이전트, 병렬 에이전트, 역할 분담 실행을 요청한 경우에만 사용한다.
- 서브에이전트를 쓰더라도 현재 세션이 전체 책임자다.
- 병렬화할 수 있는 조사/검증/독립 코드 변경만 분리한다.
- 즉시 다음 작업이 막히는 핵심 판단은 메인 세션에서 직접 한다.
- 서브에이전트 결과는 그대로 믿고 끝내지 말고, 최소한 산출물 경로와 핵심 검증 결과를 확인한다.

### 3.4 목표 달성 루프

- 목표는 항상 검증 가능한 형태로 쪼갠다. 예: `identity_error_count=0`, `scheduled_jobs=100`, `hard violation=0`, `특정 위반 CSV 생성`.
- 각 루프가 끝날 때 `완료`, `진행 중`, `차단`, `재검토 필요` 중 하나로 상태를 정리한다.
- 실패하면 fallback으로 성공처럼 보이게 하지 않는다. 원인, 입력, 영향 범위, 다음 시도를 출력한다.
- 테스트가 실패하면 실패 로그를 먼저 읽고, 실패 원인을 고친 뒤 같은 테스트를 다시 실행한다.
- 성공 기준을 만족하지 못했는데도 “완료”라고 보고하지 않는다.

---

## 4. Fallback 금지

- silent fallback 금지.
- 조용히 기본값으로 대체하는 함수 금지.
- 컬럼이 없으면 대체 컬럼을 몰래 쓰지 않는다. alias로 명시된 컬럼만 허용한다.
- 날짜 파싱 실패를 현재 날짜, 0분, 평균값으로 대체하지 않는다.
- 처리시간 산식 실패를 `TACT_TIME`, 평균값, 1분으로 대체하지 않는다.
- machine/bay 매핑 실패를 임의 Bay 또는 첫 번째 machine으로 대체하지 않는다.
- 제약 평가 실패를 pass로 처리하지 않는다.
- fallback이 정말 필요한 경우는 config에 명시된 옵션이어야 하며, 출력물에 fallback 사용 여부와 이유를 반드시 남긴다.
- fallback이 없는 상태에서 진행 불가능하면 print로 원인을 출력하고 예외를 발생시킨다.

---

## 5. 오류 출력 규칙

- 실패 가능성이 있는 데이터 로딩, 파싱, scenario 변환, 환경 step, replay, validator에는 원인 출력이 있어야 한다.
- 예외를 잡는 경우 반드시 `print()`로 함수명, 입력 키, 원인, 영향 범위를 출력한 뒤 다시 raise 한다.
- broad `except Exception`을 쓰면 반드시 원인 출력 후 `raise RuntimeError(...) from exc`로 연결한다.
- 실패를 숨기기 위해 `pass`, 빈 list 반환, None 반환을 사용하지 않는다.
- CLI 명령은 시작 시 입력 config/scenario/output 경로를 출력한다.
- CLI 명령은 종료 시 핵심 count와 검증 결과를 출력한다.
- 디버그 출력은 최소한 아래 형식을 따른다.

```text
[ERROR][module.function] cause=<reason> key=<id> input=<summary>
[CHECK][module.function] rows=<n> excluded=<n> output=<path>
[VALIDATION][module.function] passed=<true/false> violations=<n>
```

---

## 6. 데이터 기준

- 전체 데이터로 바로 개발하거나 성능/제약 검증 기준으로 삼지 않는다.
- 1차 개발과 회귀검증은 `config_np_100.yaml` 기준으로 한다.
- `config_np_100.yaml`은 `input/`의 거대 YAML을 읽지 않고 `input/절단03~04_NP물량_마스킹_WO_수정_260618.xlsx`에서 100건 scenario를 메모리로 만든다.
- 전체 NP 실행 기준은 `input/절단03~04_NP물량_마스킹_WO_수정_260618.xlsx`이며, 블록 수정 파일은 `WK_ORD_NO`가 없으므로 scheduling/replay의 주 입력으로 쓰지 않는다.
- 현재 planning factory는 Bay 22/23/24/25/trans와 PLS/PLP 총 15대를 포함한다. machine identity는 `Utils/data/multi_series_cutting_data.py`의 `MIXED_PLANNING_MACHINE_IDS_BY_BAY`가 단일 기준이다.
- `config_np_100.yaml`과 `config_np_full.yaml`의 NP DES/replay factory는 해당 실적 scope에 존재하는 13대다. 이 회귀용 factory와 MIXED 학습·계획의 15대 topology를 혼동하지 않는다.
- `EQP_3`은 실제 데이터에서 NC/trans로만 관측된 actual-only 설비다. 실적 identity는 보존하되 planning machine 후보에는 넣지 않는다.
- `input/`은 사용하지 않는 생성 산출물 보관소로 두지 않는다. 필요 산출물은 `output/generated/` 아래에 만든다.
- 전체 NP는 100건 회귀검증이 통과한 뒤 스케일 확인과 데이터 품질 audit 용도로만 실행한다.
- 동일 장비/동일 실적 착수/종료 timestamp 다중 W/O는 제약 위반 근거가 아니라 현업 확인된 데이터 오류 후보로 분리한다.
- MIXED 학습은 `변경사항/절단블록_데이터.xlsx`와 `변경사항/절단WO_데이터.xlsx`의 NP/FN/FL/NC 계약을 사용한다.
- NP100 DES/replay 회귀와 MIXED Phase 1/2 학습을 같은 입력 계약으로 혼동하지 않는다.
- 지원하지 않는 계열, duration 이상치, 파싱 실패 row는 삭제하지 않고 제외 로그에 남긴다.
- `TACT_TIME`은 기계가 아크로 절단하는 시간이며 단위는 분이다.
- generated simulation, 휴리스틱, RL 검증의 기본 처리시간은 `TACT_TIME`이다.
- 실적 착수-종료 elapsed time은 처리시간 검증값이 아니라 actual replay identity, 시간축 가시화, 데이터 품질 audit 용도로만 쓴다.
- `CUT_BAY`와 `downstream_bay`는 섞지 않는다.
- `Job.cut_bay`는 절단 Bay, `Job.downstream_bay`는 후공정/적치 Bay다.
- NP 장척은 동일 `PROJ_NO+GYEL+BLK_NO` W/O의 `CUT_LTH` 합이 1,000 이상인지로 판정한다. 개별 W/O 최댓값으로 판정하지 않는다.
- NP 광폭·CNT·장척 Bay 제약과 계열별 Bay/machine eligibility는 planning action mask에 적용한다.
- NCG, LSR, LM/크레인, Bay IN/OUT은 현재 모델에서 제외한다.

---

## 7. 환경/DES 기준

- 최종 목표는 event-generating DES core다.
- actual replay는 DES 검증 모드다.
- generated simulation과 actual replay는 같은 event schema를 사용해야 한다.
- Phase 1 action은 `(block-series, Bay)` pair 선택이다.
- merged Phase 2 policy action은 `SELECT_WO` 하나다. 환경은 현재 전역 시각에 실행 가능한 유휴 설비를 `machine_id` 오름차순으로 선택하고, 없을 때만 전체 설비의 다음 최소 완료 시각으로 event jump한다.
- batch close는 설비의 미래 완료 시각만 예약하며 전역 `current_time`을 이동하지 않는다. 모든 설비가 점유된 경우에만 완료 이벤트로 이동하고, 모든 W/O 배정 후 남은 완료 이벤트를 drain한다.
- Phase 2 teacher score는 `hard violation -> makespan -> Bay 내부 CUT_LTH gap 합 -> W/O 수 gap 합 -> BV_QTY gap 합 -> 점유시간 gap 합`의 사전식 순서다.
- DES runtime은 open batch에 W/O를 추가하고 close할 때 machine에 투입하는 동일 batch 계약을 사용한다.
- batch는 W/O 1~3개를 묶고, W/O `길이(LTH)` 합 55,000 이하일 때만 close되어 같은 시점에 투입/완료되는 구조다.
- 같은 `PROJ_NO+GYEL+BLK_NO`의 W/O는 같은 절단 Bay에 배정한다. 같은 물리 블록이라도 계열이 다르면 다른 Bay로 갈 수 있다.
- 한 batch 안에는 서로 다른 블록의 W/O가 섞일 수 있다.
- 실적 데이터와 알고리즘 결과 비교는 투입 순서나 실적 착수/종료시간 오차가 아니라 Bay별/설비별 부하평준화 중심으로 한다.
- DES 구현 표현은 `open_batch:job_id@machine_id`, `add_to_batch:batch_id:job_id`, `close_batch:batch_id@machine_id`를 사용한다.
- `Machine.bay_id`가 있으므로 machine 선택 순간 절단 Bay도 확정된다.
- `job_id@machine_id` 단건 dispatch는 baseline/debug 전용이며, 기본 환경으로 사용하지 않는다.
- Gymnasium은 DES core 위에 얇게 붙이는 wrapper다.
- SimPy는 queue, transfer, resource-flow 데이터가 들어온 뒤 내부 엔진 후보로 검토한다.

---

## 8. 공장/Bay/설비 모델링 기준

- Bay에 설비 수량을 적으면 해당 Bay 안에 그 수만큼 machine instance가 생성되어야 한다.
- 예: `Bay 6: PLS 3, LSR 2`는 PLS machine 3대와 LSR machine 2대를 생성해야 한다.
- 여러 Bay가 추가되어도 코드 수정 없이 config만 바꿔 machine 목록이 생성되어야 한다.
- 현업 machine id가 있으면 `machine_ids`를 우선 사용한다.
- 현업 machine id가 없으면 `id_pattern`으로 자동 생성한다.
- `Machine.bay_id`는 필수다.
- machine과 Bay 관계는 제약과 event log에서 모두 추적 가능해야 한다.

---

## 9. 제약 추가 기준

- 새 제약은 `Environment/constraints/` 아래에 작은 함수로 추가한다.
- 새 제약 함수는 `ConstraintContext`를 입력으로 받고 `ConstraintResult`를 반환한다.
- 새 제약은 `Environment/constraints/registry.py`의 `RULES`에 등록한다.
- 새 제약은 `RULE_CATEGORIES`에 category를 등록한다.
- 새 제약은 config의 `hard_enabled` 또는 `soft_enabled`에서 켜고 끈다.
- category는 기존 `machine`, `capacity`, `downstream`, `priority`, `preference`, `layout`, `calendar` 중 하나를 우선 사용한다.
- 새 제약에 필요한 정보가 `ConstraintContext`에 없으면 context와 dataclass를 먼저 확장한다.
- hard 제약 평가 중 오류가 나면 fail-open 하지 않는다. 원인을 print하고 실패시킨다.
- planning hard violation은 정상 결과로 보여주지 않는다.

---

## 10. 코드 작업 기준

- 새 기능은 작은 함수 단위로 나눈다.
- 한 함수는 한 책임만 갖게 한다.
- 원본 데이터 로딩, 전처리, scenario 변환, 시뮬레이션 실행, replay, validation, export를 분리한다.
- CSV/Excel 파싱 로직은 환경 코드에 직접 넣지 말고 `Utils/` 아래로 분리한다.
- 환경은 가능하면 scenario YAML만 입력으로 받게 유지한다.
- manual edit은 `apply_patch`를 사용한다.
- 사용자 변경을 되돌리지 않는다.
- 임의 cleanup, 대규모 refactor, 파일 삭제는 master checklist에 근거가 있을 때만 한다.
- 기존 구현을 바꿀 때는 변경 전후 output count가 어떻게 달라지는지 확인한다.

---

## 11. 가시화/디버깅 기준

- 휴리스틱 또는 학습 결과는 사람이 직접 확인 가능한 report와 playback으로 남긴다.
- 정적 리포트는 추가 서버 없이 브라우저에서 여는 `index.html`을 사용한다.
- 동적 playback은 `playback.html`을 사용한다.
- playback은 시간 흐름, play/pause, speed control, timeline slider를 제공해야 한다.
- playback은 machine별 현재 작업, W/O 상태, 진행률, simulation clock, W/O 검색을 제공해야 한다.
- hard validation 실패 시 playback을 정상 결과처럼 보여주지 않는다.
- event log가 생긴 뒤 playback은 event log를 우선 입력으로 사용해야 한다.
- 회사 공유용 summary는 actual replay와 generated schedule의 Bay별/설비별 부하편차를 함께 보여줘야 한다.

---

## 12. 실행 확인

코드 수정 후 최소 아래를 확인한다.

```bash
python3 -X pycache_prefix=/tmp/pmsp_pycache -m py_compile main.py Agent/*.py Phase1/*.py Phase2/*.py Environment/*.py Environment/constraints/*.py Utils/data/*.py Utils/learning/*.py Utils/phase1/*.py Utils/reporting/*.py Train/network/*.py scripts/*.py
python3 -m unittest discover -s tests -p 'test_*.py'
python3 main.py show-config --config config_np_100.yaml
python3 main.py factory-summary --config config_np_100.yaml
python3 main.py simulate --config config_np_100.yaml --heuristic spt
python3 main.py simulate --config config_np_100.yaml --heuristic load_balance
python3 main.py factory-replay --config config_np_100.yaml
git diff --check
```

DES event 전환 후에는 추가로 아래를 확인한다.

```bash
python3 main.py playback --config config_np_100.yaml --heuristic spt
```

확인 기준:

- 100건 SPT scheduled_jobs 100
- 100건 load_balance scheduled_jobs 100
- unscheduled_jobs 없음
- actual replay identity error 0
- hard validation 실패 시 실패 원인 출력
- event log 생성 시 event count가 예상 count와 일치

---

## 13. 완료 보고 기준

- 최종 답변에는 변경한 핵심 파일을 짧게 말한다.
- 실행한 검증 명령과 결과를 말한다.
- 실행하지 못한 검증이 있으면 명확히 말한다.
- 남은 위험이나 다음 작업이 있으면 숨기지 않는다.
