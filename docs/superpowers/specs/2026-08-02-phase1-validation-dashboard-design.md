# Phase 1 validation 대시보드 자동 갱신 — 설계

- 작성일: 2026-08-02
- 브랜치: 21th
- 상태: 승인됨 (구현 대기)

## 1. 목적

현재 학습 중인 Phase 1 `series_only` 런(`output/phase1_mixed_resource_pool_v2`)의
진행 현황을, 기존 참고 대시보드(18th-merge Phase 2 모니터, self-contained HTML)와
같은 형태로 보여주고 **validation이 찍힐 때마다 자동으로 갱신**한다.

핵심 제약: 대상 런은 **이미 실행 중**(PID 70768)이므로 학습 프로세스의 코드는
바꿀 수 없다. 따라서 학습 코드에 훅을 넣지 않고, **바깥에서 output dir를 감시하는
워처**로 대시보드를 갱신한다.

## 2. 확정된 결정

| 항목 | 결정 |
| --- | --- |
| 대상 범위 | Phase 1 현재 런 1개만 (`output/phase1_mixed_resource_pool_v2`) |
| 갱신 방식 | 백그라운드 워처 → 로컬 self-contained HTML 재생성 |
| 재사용 전략 | A안 — 기존 `scripts/build_training_dashboard.py`의 데이터 계층 재사용 + 단일 런 HTML 템플릿 신규 |
| 차트 ④ | episode당 소요시간(sec/ep) — 10일 런의 I/O 누적 감시용 |
| 폴링 주기 | 60초 |
| 출력 위치 | `output/dashboard/phase1_v2/dashboard.html` (+ `dashboard_data.json`) |

명시적으로 **제외**: 참고 artifact의 "제약 위반(hard)" 차트(Phase 1은 배정 문제라
`metrics.csv`에 `hard_violation_count`가 없음), "temperature / greedy takeover" 지표
(현재 Phase 1은 고정 temperature, annealing 아님).

## 3. 데이터 계약 (근거)

Phase 1 런이 이미 쓰고 있는 파일만 읽는다. **학습 코드 변경 없음.**

- `metrics.csv`: `episode, loss, best_source(NP_NC:x|FN_FL:y), subproblem_count, objective_scope, ...`
  (hard_violation 컬럼 없음)
- `validation_summary.csv`: `train_episode, validation_view(NP/NC/NP_NC/FN/FL/FN_FL),
  agent_is_best, agent_rank, best_source, np_*_gap ... fn_fl_*_gap`.
  validation 1회당 여러 view row가 쌓인다. `--validation-every 100`이므로 100 ep마다 새 `train_episode`.
- `checkpoints/*.pt`: mtime 간격으로 sec/ep 계산 (metrics.csv에 시각 컬럼이 없어서 필요).
  checkpoint가 2개 미만이면 sec/ep는 `–`.
- CSV는 학습이 append 중이라 마지막 줄이 잘릴 수 있음 → 기존 `_read_csv_rows`가
  잘린 행을 버리므로 그대로 재사용한다.

## 4. 구성요소

### 4.1 생성기 `scripts/build_phase1_dashboard.py`

- CLI: `--run-dir`, `--target-episodes`(기본 20000), `--output`(기본
  `output/dashboard/phase1_v2/dashboard.html`), `--window`(이동평균, 기본 25).
- `build_training_dashboard.py`에서 `collect_phase, _read_csv_rows, _to_int, _to_float`
  등을 import한다. (같은 `scripts/` 안이라 스크립트 실행 시 sys.path[0]으로 import 가능.)
- `collect_phase(run_dir, target, window, now)`로 단일 런을 집계 → 기존이 이미 주는 값:
  current_ep, target, progress_pct, running, sec_per_ep, eta_h, loss/loss_avg,
  agent_rate_recent/all, source_counts, speed_intervals, validation_rate, validation_rows.
- **추가 추출(신규 helper)**: holdout best-rate를 자원군 view별로 분리하는
  `phase1_best_rate_by_view(validation_rows, view)` — `validation_view == view`인 행만
  골라 `train_episode`별 `100 * mean(agent_is_best)` 시계열. view = `NP_NC`, `FN_FL`,
  그리고 두 view 합친 `overall`.
- self-contained HTML + `dashboard_data.json`을 쓴다. 외부 의존성 0, 라이트/다크 테마,
  손수 짠 SVG 차트/툴팁/테마토글 JS는 기존 `_TEMPLATE`에서 가져와 재사용하되 DOM은
  단일 런 레이아웃으로 재구성한다.
- strict: `--run-dir`가 없으면 `[ERROR][build_phase1_dashboard] cause=... path=...`
  출력 후 `raise RuntimeError(...)`. (AGENTS.md no-silent-fallback)

### 4.2 워처 `scripts/watch_phase1_dashboard.py`

- CLI: `--run-dir`, `--target-episodes`, `--output`, `--interval`(기본 60), `--window`.
- 순수 함수 `latest_validation_episode(run_dir) -> int | None`:
  `validation_summary.csv`의 최대 `train_episode` (없으면 None).
- 루프:
  1. 기동 즉시 1회 렌더.
  2. `interval`초마다 `latest_validation_episode`를 확인 → 직전 값보다 커지면(=새 validation) 재렌더.
  3. 매 폴링에서 `metrics.csv` 마지막 `episode`(current_ep)를 확인 → `>= target`이면
     마지막 렌더 1회 후 종료(exit 0).
- strict: run dir 없으면 진단 출력 후 `raise`.
- Windows에서 detached(`Start-Process`)로 실행. 로그·PID는 출력 폴더에 기록
  (`watch_stdout.log`, `watch_stderr.log`, `watch.pid`). 지금 돌고 있는 v2 런에 즉시 적용.

## 5. 화면 구성 (참고 artifact 형태, 단일 Phase 1 런)

- **헤더**: eyebrow / 제목 / 부제 / 테마 토글.
- **타일 4개**:
  ① 진행률 `current_ep / target` + 진행 바 + `progress% · s/ep · ETA h` + 실행중 점(live dot)
  ② holdout best-rate(최신 validation)
  ③ agent 최상위율(최근 200 train episode, best_source에서 agent 승률)
  ④ episode당 소요시간(최근 sec/ep, 없으면 `–`)
- **차트 4개** (모두 x=episode, 기존 lineChart 재사용):
  ① 학습 loss — raw(옅게) + 이동평균, 하한 0
  ② holdout best-rate(%) — `NP_NC`, `FN_FL`, `overall` 3선, 0~100 고정 눈금, validation 지점 dot
  ③ agent 채택률(%) — train best_source에서 `agent_greedy` vs `agent_sample_*`(묶음) vs 휴리스틱, 이동평균
  ④ episode당 소요시간(s/ep) — checkpoint 구간, dot
- **막대**: 승리 후보 분포(최근 200 ep, subproblem별 best source; `agent_sample_*` 묶음). Phase 1은 자원군 2개라 막대 합 = 2 × episode 수.
- **표**: holdout validation 상세(최근 8회 — train ep, agent 최상위율, 평균 rank).
- **푸터**: run 경로 + 스냅샷 시각 + 이동평균 창.

## 6. 에러 처리

- 필수 파일/디렉터리 부재는 fallback 없이 진단 출력 후 예외 종료.
- 잘린 CSV 마지막 행은 버림(기존 로직 재사용). loss/gap의 빈값·NaN은 점에서 제외.
- 없는 계열(예: 특정 validation view row 부재)은 0으로 채우지 않고 해당 시계열에서 미존재로 둔다.

## 7. 테스트 (unittest, `tests/`)

- `tests/test_phase1_dashboard.py`:
  - 생성기 스모크: tmp에 최소 `metrics.csv`(episode/loss/best_source/subproblem_count 포함)와
    `validation_summary.csv`(train_episode/validation_view/agent_is_best/agent_rank) fixture를 만들어
    `build`를 돌려 HTML 파일 생성·핵심 DOM id 포함·`dashboard_data.json` 파싱·기대 키 존재 검증.
  - `phase1_best_rate_by_view`: NP_NC/FN_FL view 필터·비율 계산 단위 검증.
  - 워처 `latest_validation_episode`: tmp CSV로 최대 train_episode 반환·부재 시 None 검증.
- 컴파일 체크 목록에 두 신규 스크립트 추가.

## 8. 범위 밖 (YAGNI)

- Phase 2 / dual-phase 대시보드(기존 `build_training_dashboard.py`가 담당) 변경.
- claude.ai Artifact 자동 게시(자동화 불가 — 요청 시 수동 게시).
- 계열별 gap 추세 차트(차트 ④를 sec/ep로 확정, gap 추세는 도입하지 않음).
- 학습 코드(main.py, Phase1/*) 수정.

## 9. 파일 목록

- 신규: `scripts/build_phase1_dashboard.py`, `scripts/watch_phase1_dashboard.py`, `tests/test_phase1_dashboard.py`
- 재사용(변경 없음): `scripts/build_training_dashboard.py`
- 산출물(gitignore된 output/): `output/dashboard/phase1_v2/{dashboard.html,dashboard_data.json,watch.*}`
