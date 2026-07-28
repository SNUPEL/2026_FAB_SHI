# Phase 1 학습 지표 I/O를 append 방식으로 전환

작성일: 2026-07-28
대상 브랜치: `16th`
관련 파일: `Phase1/pair_self_labeling.py`, `main.py`

## 1. 문제

Phase 1 학습(`phase1-train-pair-self-labeling`)은 **매 에피소드마다 네 개의 산출 파일을 전체 재작성**한다
(`Phase1/pair_self_labeling.py:782-787`).

```python
_write_metrics(output_path / "metrics.csv", metrics_rows)
_write_subproblem_metrics(output_path / "subproblem_metrics.csv", subproblem_metric_rows)
_write_candidate_summary(output_path / "candidate_summary.csv", candidate_rows)
_write_jsonl(output_path / "best_action_table.jsonl", best_action_rows)
```

`_write_csv`/`_write_jsonl`은 모두 `path.open("w")`로 파일을 처음부터 다시 쓴다
(`Phase1/pair_self_labeling.py:2923`, `:2933`). 누적 행이 늘수록 에피소드당 쓰기량이 선형으로 증가하고,
따라서 학습 총소요는 에피소드 수에 대해 **2차(quadratic)**로 늘어난다.

Phase 2는 같은 위치에서 `_append_rows(...)`로 새 행만 덧붙인다(`Phase2/merged.py:417-418`). 즉 동일 문제가
Phase 2에서는 이미 해결되어 있고, Phase 1에만 남아 있다.

### 1.1 실측 근거

기존 학습 `output/phase1_pair_20k`(에피소드 4,627에서 중단)의 체크포인트 간격에서 구간 속도를 계산했다.

| 에피소드 | s/ep |
|---|---|
| 200 | 20.1 |
| 1,100 | 35.7 |
| 2,100 | 54.5 |
| 3,100 | 74.2 |
| 4,600 | 98.8 |

같은 시점 재작성 대상 파일 크기:

| 파일 | 크기 | 행 수 |
|---|---|---|
| `candidate_summary.csv` | 1.9 GB | 620,019 |
| `best_action_table.jsonl` | 529 MB | 46,813 |
| `subproblem_metrics.csv` | 1.9 MB | 9,255 |
| `metrics.csv` | 1.4 MB | 4,628 |
| **합계 (에피소드당 재작성량)** | **약 2.4 GB** | |

구간 속도를 직선으로 적합하면 `sec_per_ep(N) ≈ 16.5 + 0.0179 · N`이다. 이를 0..4,627 구간에서 적분하면
**74.4시간**으로, 실제 소요 **74.15시간**과 일치한다. 즉 이 진단은 실측으로 검증된 것이다.

대조군으로 Phase 2(`output/phase2_upstream_wo`, append 방식)는 100 에피소드당 체크포인트 간격이 학습 내내
56~60분으로 평평했다.

### 1.2 영향

20,000 에피소드 기준 추정:

| | 총 소요 | 마지막 에피소드 속도 |
|---|---|---|
| 현재 코드 | 약 45일 | 374 s/ep |
| 수정 후 | 약 3.8일 | 16.5 s/ep (일정) |

부수적으로, 재작성 대상 행을 전부 메모리에 들고 있으므로(`candidate_rows` 620k 행) RAM 사용도 함께 증가한다.

## 2. 목표와 비목표

**목표**
- 에피소드당 지표 I/O 비용을 O(누적 행 수)에서 O(신규 행 수)로 낮춘다.
- 누적 행의 불필요한 메모리 보유를 제거한다.
- 학습 결과(모델, 점수, 로그 내용)는 **변경하지 않는다**. 순수 I/O 변경이다.

**비목표 (이번 범위 밖)**
- Phase 1/Phase 2 동시 실행 및 자원 배분 측정 (`--device`, `--candidate-workers`)
- 학습 현황 대시보드 생성기
- 학습 알고리즘·하이퍼파라미터 변경

## 3. 설계

### 3.1 누적 리스트별 처리

각 리스트가 파일 쓰기 외에 어디서 읽히는지 조사한 결과에 따라 처리를 달리한다.

| 리스트 | 쓰기 외 사용처 | 처리 |
|---|---|---|
| `candidate_rows` | 없음 (`:695` append만) | 메모리 누적 제거, 파일 append |
| `best_action_rows` | 없음 (`:710` extend만) | 메모리 누적 제거, 파일 append |
| `metrics_rows` | 마지막 행(`:879-885`), `_source_counts`(`:920`) | 메모리 유지, 파일 append |
| `subproblem_metric_rows` | 길이만(`:913`) | 메모리 유지, 파일 append |

`metrics_rows`/`subproblem_metric_rows`는 20,000 에피소드에서도 각각 수 MB 수준이므로 메모리 유지 비용이
무시할 만하고, 요약(`summary.json`) 생성에 필요하므로 그대로 둔다.

### 3.2 append 헬퍼

`Phase2/merged.py:3010`의 `_append_rows`와 동일한 의미의 헬퍼를 Phase 1에 둔다.

- 파일이 없으면 헤더를 쓰고 행을 덧붙인다.
- 파일이 있으면 헤더 없이 행만 덧붙인다.
- 인코딩은 기존과 동일하게 CSV는 `utf-8-sig`, JSONL은 `utf-8`을 유지한다.
  (기존 산출물과 플롯 스크립트가 `utf-8-sig`로 읽으므로 바꾸지 않는다.)

### 3.3 resume 처리

현재는 시작 시 `_read_csv_rows(path, "episode", start_episode)`로 이전 행을 메모리에 적재하고, 이후 매
에피소드 전체 재작성으로 파일이 갱신된다. append 전환 후에는 **시작 시 1회만** 파일을 잘라낸다.

- 시작(또는 resume) 시점에 각 파일을 읽어 `episode < start_episode`인 행만 남기고 **한 번** 다시 쓴다.
- 이후 에피소드는 append만 수행한다.
- 새 학습(resume 아님)일 때 기존 파일이 남아 있으면 동일 절차로 잘라내므로, 이어붙기 오염이 발생하지 않는다.

### 3.4 `--write-candidate-summary` 플래그

Phase 2에 이미 있는 인터페이스(`main.py:1304-1306`, 기본 `False`, 도움말 "Disabled by default for long
training speed")를 Phase 1에도 추가한다.

- 기본 `False`: `candidate_summary.csv`를 생성하지 않는다.
- `True`: 기존과 동일하게 후보별 audit 행을 append로 기록한다.
- 이 파일이 전체 재작성량의 대부분(1.9 GB)이므로, 플래그만으로도 효과가 크다.

플래그 이름·기본값·도움말 문구는 Phase 2와 맞춰 두 phase의 CLI 사용법이 어긋나지 않게 한다.

## 4. 검증

1. **동등성** — 동일 시드(`--seed 0`)로 수정 전후 각각 짧은 학습을 돌려 `metrics.csv`와
   `subproblem_metrics.csv`의 내용이 동일한지 비교한다. 학습 결과가 바뀌지 않아야 한다.
2. **속도 곡선** — 이것이 이 작업의 성공 기준이다. "수정 전" 곡선은 다시 측정하지 않는다. 기존
   `output/phase1_pair_20k`가 동일 시드·동일 설정으로 돌았으므로 그 구간 속도를 기준선으로 쓴다
   (ep 200 → 20.1 s/ep, ep 300 → 21.9 s/ep, 이후 계속 상승).

   수정 후 동일 설정으로 300 에피소드를 돌려 100 에피소드 구간별 s/ep를 계산하고, 다음을 확인한다.
   - 구간 속도가 **에피소드가 늘어도 증가하지 않는다** (평평하거나 문제 크기에 따른 무작위 변동만 존재).
   - 절대값이 기준선 이하다.

   시드가 고정이라 두 실행의 에피소드별 문제 크기가 동일하므로 짝지은 비교가 성립한다.
3. **resume** — 학습을 중간에 중단하고 `--resume-checkpoint latest`로 재개했을 때 각 산출 파일에 행
   중복·누락·헤더 중복이 없는지 확인한다.
4. **플래그** — `--write-candidate-summary` 미지정 시 `candidate_summary.csv`가 생성되지 않고, 지정 시
   기존과 같은 스키마로 생성되는지 확인한다.
5. **기존 테스트** — Phase 1 관련 기존 테스트가 계속 통과해야 한다.

검증 2와 3은 자동 테스트로 만들기 어려우므로 수동 실행으로 확인하고 결과를 기록한다. 검증 1·4·5는 테스트로
남긴다.

## 5. 위험 요소

- **resume 회귀** — 절단 시점이 1회로 바뀌므로, 잘라내기 조건(`episode < start_episode`)이 기존과 정확히
  같아야 한다. 다르면 재개 시 행이 중복되거나 사라진다. 검증 3으로 확인한다.
- **`candidate_summary.csv` 소비자** — 기본 비활성으로 바뀌므로, 이 파일을 읽는 도구가 있으면 영향을 받는다.
  `scripts/plot_phase1_pair_training.py`가 이 파일을 읽으므로, 플래그 없이 학습한 run에 대해서는 해당 스크립트가
  후보 관련 그래프를 생성할 수 없다. 이 동작 차이를 스크립트에서 명시적으로 처리한다(파일이 없으면 해당 그래프를
  건너뛰고 이유를 출력).
- **부분 쓰기** — append 중 프로세스가 죽으면 마지막 행이 잘릴 수 있다. 기존 전체 재작성 방식도 같은 위험이
  있었고(오히려 파일 전체가 손상될 수 있었다) append가 더 안전하므로, 추가 대응은 하지 않는다.
