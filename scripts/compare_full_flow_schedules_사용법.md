# full-flow 스케줄 비교 도구 사용법 (`compare_full_flow_schedules.py`)

특정 문제 인스턴스 하나에 대해 **Phase 2 휴리스틱**과 **제안 방법(Phase 2 RL batch-machine 에이전트)**의
full-flow 실행 결과를 **Gantt 차트 · 배치 구성 표 · 비교 요약**으로 나란히 보여주는 시각화 전용 도구다.

- 스크립트: [scripts/compare_full_flow_schedules.py](compare_full_flow_schedules.py)
- 이 도구는 **그림/표만 그린다.** full-flow 실행은 기존 CLI(`python main.py phase2-run-full-workflow ...`)로
  미리 수행하고, 이 도구는 그 결과 output 디렉터리 2개를 읽는다. 파이프라인 코드는 전혀 건드리지 않는다.
- 관련 문서: [다계열 Phase 1·Phase 2 학습 실행 가이드](../다계열_Phase1_Phase2_학습_실행_가이드.md)

---

## 1. 핵심 개념

- **비교 단위**: 방법 A vs 방법 B. 두 방법은 **동일 `--seed` + `--synthetic-blocks` + 동일 Phase 1 upstream**을
  써야 한다. 그래야 작업 집합·작업시간·순위가 동일하고 **batch 구성·설비 배치·timeline·makespan만** 달라져,
  차이를 방법의 효과로 해석할 수 있다.
- **batch는 병렬 실행**: 한 batch의 소요시간 `batch_duration = max(멤버 작업 processing_time)`이다
  (합이 아님). 그래서 Gantt에서 가장 긴 멤버 작업이 batch 폭을 결정한다.
- **순위(rank)**: "같은 배치를 형성할 수 있는 동종 계열" = **같은 Bay·같은 family** 안에서
  `processing_time` 내림차순 등수(1등 = 가장 김). 기본 기준은 `--rank-scope bay_family`.
- **family**: `block_set_id = PROJ_NO::GYEL::BLK_NO`의 가운데 토큰(NP/NC/FN/FL).

---

## 2. 선행 조건 — 입력 디렉터리 2개 만들기

제안 방법이 **Phase 2 RL 에이전트**이므로 학습된 체크포인트 `phase2_batch_machine_policy.pt`가 있어야 한다.

### 2-1. (제안 방법을 쓸 때만) Phase 2 학습

가이드 5절 참고. 예:

```cmd
python main.py phase2-train-batch-machine-self-labeling --config config_mixed.yaml ^
  --phase1-heuristic wo_first_balanced --episodes 20000 --seed 0 --device cuda ^
  --output-dir output/phase2_upstream_wo
```

→ `output/phase2_upstream_wo/phase2_batch_machine_policy.pt` 생성.

### 2-2. 동일 인스턴스로 full-flow 2회

두 실행에서 `--phase1-*` upstream, `--seed`, `--synthetic-blocks`를 **똑같이** 두고
Phase 2 방법만 바꾼다(휴리스틱 ↔ 체크포인트, 상호 배타).

```cmd
:: 방법 A — Phase 2 휴리스틱
python main.py phase2-run-full-workflow --config config_mixed.yaml ^
  --phase1-heuristic wo_first_balanced --batch-machine-heuristic min_makespan ^
  --synthetic-blocks 30 --seed 0 --output-dir output/cmp_heur

:: 방법 B — 제안(학습된 Phase 2 에이전트)
python main.py phase2-run-full-workflow --config config_mixed.yaml ^
  --phase1-heuristic wo_first_balanced ^
  --batch-machine-checkpoint output/phase2_upstream_wo/phase2_batch_machine_policy.pt ^
  --synthetic-blocks 30 --seed 0 --output-dir output/cmp_agent
```

> **학습 없이 도구만 먼저 검증**하려면 방법 B도 `--batch-machine-heuristic`으로 두고
> 두 휴리스틱을 비교한다(예: A=`min_makespan`, B=`lpt_batch`).

**Phase 2 휴리스틱 뱅크(실제 6종)**: `workload_makespan_dispatch`, `min_makespan`,
`lookahead_min_makespan`, `best_fit_lth`, `spt_batch`, `lpt_batch`. (`balanced_tact_load`는 코드에 없음.)

**Phase 1 upstream 선택**: `--phase1-heuristic {wo_first_balanced|cut_first_balanced|bevel_first_balanced}`
또는 학습된 Phase 1 사용 시 `--phase1-checkpoint <...>.pt --phase1-samples 32`.
(두 full-flow에서 동일해야 함.)

각 full-flow는 output 디렉터리에 다음을 쓴다(이 도구가 읽는 파일 굵게):
**`wo_machine_assignment.csv`**, **`machine_timeline.csv`**, **`phase2_workflow_report.json`**,
`batch_list.csv`, `machine_load_summary.csv`, `makespan_summary.csv`, `bay_metric_summary.csv`,
`phase2_constraint_audit.csv`, `phase2_event_log.json`.

---

## 3. 비교 시각화 실행

```cmd
python scripts/compare_full_flow_schedules.py ^
  --method-a-dir output/cmp_heur  --label-a heuristic ^
  --method-b-dir output/cmp_agent --label-b proposed ^
  --output-dir output/analysis/schedule_comparison
```

### CLI 인자

| 인자 | 필수 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `--method-a-dir` | O | — | 방법 A full-flow output 디렉터리 |
| `--method-b-dir` | O | — | 방법 B full-flow output 디렉터리 |
| `--label-a` | | `heuristic` | 방법 A 라벨(파일명·제목에 사용) |
| `--label-b` | | `proposed` | 방법 B 라벨 |
| `--output-dir` | | `output/analysis/schedule_comparison` | 산출물 저장 폴더 |
| `--rank-scope` | | `bay_family` | 순위 집합: `bay_family`(Bay·family) / `family`(계열 전체) / `machine`(설비) |
| `--font-path` | | 자동 탐색 | 한글 폰트 경로(미지정 시 `malgun.ttf` 탐색, 없으면 기본 폰트) |

성공하면 마지막에 `[VALIDATION][compare_full_flow_schedules.main] passed=true ...`를 출력한다.

---

## 4. 산출물

`--output-dir`에 저장된다. `<A>`/`<B>`는 `--label-a`/`--label-b` 값.

| 파일 | 내용 |
| --- | --- |
| `gantt_<A>.png`, `gantt_<B>.png` | 방법별 Gantt. Y축=설비(Bay별 묶음), X축=시간. batch 외곽선=`batch_duration`, 그 안에 멤버 작업을 각자 `processing_time` 길이 서브바로 표시. 서브바 색=family, 주석=`#rank/size` |
| `gantt_compare.png` | A(위)·B(아래)를 **동일 X축**으로 스택 → makespan 차이 직접 비교 |
| `batch_composition_<A>.csv` / `.png` | batch별 구성. CSV는 작업 1행씩 정확값, PNG는 열람용 요약 |
| `batch_composition_<B>.csv` / `.png` | 〃 |
| `comparison_summary.csv` / `.png` | A vs B 지표와 Δ(B−A) |

### 4-1. `batch_composition_<label>.csv` 컬럼

| 컬럼 | 의미 |
| --- | --- |
| `batch_id`, `bay_id`, `machine_id` | batch 식별·위치 |
| `start_time`, `finish_time`, `batch_duration` | batch 시작/끝/소요(=멤버 최장 시간) |
| `job_id`, `family` | 멤버 작업·계열 |
| `processing_time` | 이 작업의 처리시간 |
| `rank_in_group`, `group_size` | 동종계열(기본 Bay·family) 내 작업시간 등수 / 그룹 크기 |
| `is_longest_in_batch` | 이 작업이 해당 batch의 최장(=`batch_duration`을 정함)이면 1 |

PNG 요약의 멤버 셀 표기: `★WO<번호>-<seq> <family> #<rank>/<size> (<proc>)` — `★`는 batch 최장 작업.

### 4-2. `comparison_summary.csv` 지표

`metric, <A>, <B>, delta(B-A)` 형식.

- score 6성분(**작을수록 우수**): `hard_violation → makespan → cut_gap → wo_gap → bevel_gap → occupancy_gap`
  (Phase 2 사전식 score 순서. PNG에서 이 6개만 Δ 색상 표시: 녹색=개선, 적색=악화)
- 진단 지표: `batch_count`, `avg_batch_wo_count`, `avg_wo_fill_ratio`(평균 W/O 채움률=평균 wo수/3), `total_jobs`

---

## 5. 그림 읽는 법

- **Gantt 서브바 길이** = 그 작업의 `processing_time`. **검은 외곽선** = batch 전체(=최장 멤버).
  → batch 안에서 가장 오른쪽까지 뻗은 작업이 `batch_duration`을 결정한다.
- **서브바 옆 숫자** `4/52` = 이 작업이 동종계열(Bay·family) 52개 중 4번째로 길다는 뜻.
- **색상** = family(NP·NC·FN·FL). 범례는 우측 바깥.
- **Bay 밴드/라벨** = 설비를 Bay별로 묶어 배경색으로 구분.
- `gantt_compare.png`는 두 방법을 같은 X축으로 보여주므로 makespan(오른쪽 끝) 차이가 한눈에 보인다.

---

## 6. 자체 검증(그림 그리기 전 실패 시 중단)

각 방법을 로드할 때 다음을 확인하고, 어긋나면 `[ERROR]...cause=<사유>`를 찍고 중단한다.

1. 모든 assignment 작업이 정확히 한 batch에만 등장(누락/중복 스케줄 없음)
2. `batch_duration == max(멤버 processing_time)` (오차 1e-3)
3. `makespan == max(finish_time)` == report.json의 makespan
4. 각 (Bay, family) 그룹의 rank가 1..n 유일

---

## 7. 재현 예 (학습 없이 스모크)

```cmd
python main.py phase2-run-full-workflow --config config_mixed.yaml --phase1-heuristic wo_first_balanced ^
  --batch-machine-heuristic min_makespan --synthetic-blocks 20 --seed 0 --output-dir output/_smoke_min
python main.py phase2-run-full-workflow --config config_mixed.yaml --phase1-heuristic wo_first_balanced ^
  --batch-machine-heuristic lpt_batch    --synthetic-blocks 20 --seed 0 --output-dir output/_smoke_lpt
python scripts/compare_full_flow_schedules.py ^
  --method-a-dir output/_smoke_min --label-a min_makespan ^
  --method-b-dir output/_smoke_lpt --label-b lpt_batch ^
  --output-dir output/analysis/_smoke_compare
```

---

## 8. 주의사항

- **동일 인스턴스·동일 Phase 1 upstream 필수.** seed/블록수/Phase 1 방법이 다르면 비교가 무의미하다.
- **제안(RL) 비교에는 Phase 2 체크포인트가 선행**되어야 한다(현재 저장소에 학습된 `.pt` 없음 → 먼저 학습).
- Windows `cmd` 기준 예시다. PowerShell/bash에서는 줄바꿈 문자(`^`)를 각 셸 문법으로 바꾼다.
- 한글 폰트가 없으면 제목/범례가 기본 폰트로 나온다(그림은 정상 생성). `--font-path`로 지정 가능.
