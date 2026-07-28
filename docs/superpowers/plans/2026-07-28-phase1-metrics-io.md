# Phase 1 학습 지표 I/O append 전환 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 1 학습이 매 에피소드마다 산출 파일 전체를 다시 쓰는 동작을 append로 바꿔, 에피소드당 I/O를 O(누적 행 수)에서 O(신규 행 수)로 낮춘다.

**Architecture:** 필드 목록을 모듈 상수로 뽑고, Phase 2의 `_append_rows` 패턴을 이식한 append 헬퍼를 추가한다. 학습 시작 시 한 번만 파일을 `start_episode` 기준으로 절단하고, 이후 에피소드는 신규 행만 덧붙인다. 아무도 읽지 않는 두 누적 리스트(`candidate_rows`, `best_action_rows`)는 메모리 보유를 중단한다. 가장 큰 `candidate_summary.csv`는 Phase 2와 동일한 `--write-candidate-summary` 플래그로 기본 비활성화한다.

**Tech Stack:** Python 3, 표준 라이브러리 `csv`/`json`, 테스트는 `unittest`.

## Global Constraints

- 대상 브랜치는 `16th`이다. 작업 트리에 이미 다른 미커밋 변경(`Phase2/merged.py`, `main.py` staged / `CLAUDE.md` unstaged)이 있으므로, **이 계획의 커밋에는 그 파일들의 무관한 변경을 포함하지 않는다.** 각 커밋에서 파일을 명시적으로 `git add`한다.
- Python 인터프리터는 **`/home/temp_id/anaconda3/envs/accord_env/bin/python`** 이다. conda base의 `python`에는 `torch`가 없어 import가 실패한다.
- 테스트는 **unittest**로 작성한다 (`python -m unittest tests.test_x`). 저장소 관례이며 `CLAUDE.md`에 명시되어 있다.
- CSV 인코딩은 **`utf-8-sig`**, JSONL은 **`utf-8`** 을 유지한다. 기존 산출물과 `scripts/plot_*.py`가 이 인코딩으로 읽는다.
- **학습 결과는 바뀌면 안 된다.** 이 작업은 순수 I/O 변경이다. 점수 계산, 후보 생성, 업데이트 로직에 손대지 않는다.
- 오류 처리는 저장소 관례를 따른다: 진단을 `print`하고 `raise`한다. 조용한 폴백(기본값 대체, `pass`)을 만들지 않는다 (`AGENTS.md` §4–5).
- 첫 커밋 전에 사용자에게 커밋 진행 여부를 한 번 확인한다.

---

### Task 1: 필드 목록 상수화 + append 헬퍼

`_write_*` 함수 안에 인라인으로 박혀 있는 필드 목록을 모듈 상수로 뽑는다. 쓰기와 append가 같은 필드 정의를 공유해야 헤더와 행 순서가 어긋나지 않는다.

**Files:**
- Modify: `Phase1/pair_self_labeling.py` (`_write_metrics` `:2488`, `_write_subproblem_metrics` `:2516`, `_write_candidate_summary` `:2540`, `_write_csv` `:2923`, `_write_jsonl` `:2933`)
- Test: `tests/test_phase1_metrics_io.py` (신규)

**Interfaces:**
- Consumes: 없음 (첫 태스크)
- Produces:
  - `_METRICS_FIELDS: tuple[str, ...]`
  - `_SUBPROBLEM_METRICS_FIELDS: tuple[str, ...]`
  - `_CANDIDATE_SUMMARY_FIELDS: tuple[str, ...]`
  - `_append_csv_rows(path: Path, fields: Sequence[str], rows: Sequence[Mapping]) -> None`
  - `_append_jsonl_rows(path: Path, rows: Sequence[Mapping]) -> None`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_phase1_metrics_io.py` 를 새로 만든다.

```python
"""Phase 1 학습 지표 파일의 append 기록 계약을 검증한다."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from Phase1.pair_self_labeling import (
    _CANDIDATE_SUMMARY_FIELDS,
    _METRICS_FIELDS,
    _SUBPROBLEM_METRICS_FIELDS,
    _append_csv_rows,
    _append_jsonl_rows,
)


class AppendCsvRowsTests(unittest.TestCase):
    def test_creates_header_once_and_appends_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            fields = ("episode", "loss")

            _append_csv_rows(path, fields, [{"episode": 1, "loss": 0.5}])
            _append_csv_rows(path, fields, [{"episode": 2, "loss": 0.4}])

            with path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual([row["episode"] for row in rows], ["1", "2"])
            self.assertEqual([row["loss"] for row in rows], ["0.5", "0.4"])
            self.assertEqual(path.read_text(encoding="utf-8-sig").count("episode,loss"), 1)

    def test_missing_key_becomes_empty_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            _append_csv_rows(path, ("episode", "loss"), [{"episode": 1}])

            with path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(rows[0]["loss"], "")

    def test_empty_rows_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            with self.assertRaises(RuntimeError):
                _append_csv_rows(path, ("episode",), [])


class AppendJsonlRowsTests(unittest.TestCase):
    def test_appends_one_line_per_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_action_table.jsonl"

            _append_jsonl_rows(path, [{"episode": 1, "step": 0}])
            _append_jsonl_rows(path, [{"episode": 2, "step": 0}, {"episode": 2, "step": 1}])

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual([json.loads(line)["episode"] for line in lines], [1, 2, 2])

    def test_empty_rows_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_action_table.jsonl"
            with self.assertRaises(RuntimeError):
                _append_jsonl_rows(path, [])


class FieldConstantTests(unittest.TestCase):
    def test_metrics_fields_match_documented_schema(self) -> None:
        self.assertEqual(_METRICS_FIELDS[0], "episode")
        self.assertIn("loss", _METRICS_FIELDS)
        self.assertIn("score_json", _METRICS_FIELDS)
        self.assertIn("subproblem_count", _METRICS_FIELDS)

    def test_subproblem_fields_carry_subproblem_id(self) -> None:
        self.assertIn("subproblem_id", _SUBPROBLEM_METRICS_FIELDS)

    def test_candidate_fields_carry_is_best(self) -> None:
        self.assertIn("is_best", _CANDIDATE_SUMMARY_FIELDS)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `/home/temp_id/anaconda3/envs/accord_env/bin/python -m unittest tests.test_phase1_metrics_io -v`
Expected: FAIL — `ImportError: cannot import name '_METRICS_FIELDS'`

- [ ] **Step 3: 필드 상수 추출**

`Phase1/pair_self_labeling.py` 의 `_write_metrics` 바로 앞(현재 `:2488` 위치)에 상수를 정의하고, 세 `_write_*` 함수가 그 상수를 쓰도록 바꾼다. 필드 순서와 내용은 기존과 **완전히 동일**해야 한다.

```python
_METRICS_FIELDS: tuple[str, ...] = (
    "episode",
    "problem_id",
    "block_count",
    "problem_seed",
    "case_type",
    "hard_case_mode",
    "hard_case_corr_steel_cut_before",
    "hard_case_corr_steel_cut_after",
    "best_source",
    "score_mode",
    "objective_scope",
    "loss",
    "score_json",
    "phase2_feedback_score_json",
    "learning_score_json",
    "candidate_count",
    "subproblem_count",
)

_SUBPROBLEM_METRICS_FIELDS: tuple[str, ...] = (
    "episode",
    "problem_id",
    "subproblem_id",
    "series",
    "block_count",
    "job_count",
    "best_source",
    "objective_scope",
    "loss",
    "score_json",
    "phase2_feedback_score_json",
    "learning_score_json",
    "candidate_count",
)

_CANDIDATE_SUMMARY_FIELDS: tuple[str, ...] = (
    "episode",
    "problem_id",
    "block_count",
    "problem_seed",
    "subproblem_id",
    "candidate_index",
    "source",
    "is_best",
    "score_mode",
    "objective_scope",
    "score_json",
    "phase2_feedback_score_json",
    "learning_score_json",
    *PHASE1_SCORE_FIELD_NAMES,
    "assignment_count",
    "transition_count",
    "bay_loads_json",
)


def _write_metrics(path: Path, rows: Sequence[Mapping]) -> None:
    """Write per-episode train metrics."""

    _write_csv(path, _METRICS_FIELDS, rows)


def _write_subproblem_metrics(path: Path, rows: Sequence[Mapping]) -> None:
    """Write one row for every independent resource-pool optimizer update."""

    _write_csv(path, _SUBPROBLEM_METRICS_FIELDS, rows)


def _write_candidate_summary(path: Path, rows: Sequence[Mapping]) -> None:
    """Write all complete candidates for audit."""

    _write_csv(path, _CANDIDATE_SUMMARY_FIELDS, rows)
```

- [ ] **Step 4: append 헬퍼 추가**

`_write_jsonl` 바로 뒤(현재 `:2938` 다음)에 추가한다.

```python
def _append_csv_rows(path: Path, fields: Sequence[str], rows: Sequence[Mapping]) -> None:
    """Append rows without rewriting the accumulated training history."""

    if not rows:
        print(f"[ERROR][phase1_pair_self_labeling._append_csv_rows] cause=no_rows path={path}")
        raise RuntimeError(f"no rows to append: {path}")
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _append_jsonl_rows(path: Path, rows: Sequence[Mapping]) -> None:
    """Append JSONL rows without rewriting the accumulated table."""

    if not rows:
        print(f"[ERROR][phase1_pair_self_labeling._append_jsonl_rows] cause=no_rows path={path}")
        raise RuntimeError(f"no rows to append: {path}")
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `/home/temp_id/anaconda3/envs/accord_env/bin/python -m unittest tests.test_phase1_metrics_io -v`
Expected: PASS (9 tests)

- [ ] **Step 6: 기존 Phase 1 테스트 회귀 확인**

Run: `/home/temp_id/anaconda3/envs/accord_env/bin/python -m unittest discover -s tests -p 'test_phase1*.py' -v`
Expected: 기존과 동일하게 전부 PASS

- [ ] **Step 7: 커밋**

```bash
git add Phase1/pair_self_labeling.py tests/test_phase1_metrics_io.py
git commit -m "refactor(phase1): extract metrics field constants and add append helpers"
```

---

### Task 2: 학습 루프의 네 파일을 append로 전환

에피소드 루프에서 누적 리스트 전체를 다시 쓰던 것을 신규 행 append로 바꾸고, 절단은 루프 진입 전 1회만 수행한다.

**Files:**
- Modify: `Phase1/pair_self_labeling.py` — resume 블록 `:582-590`, 에피소드 루프 시작 `:622`, 서브문제 루프 내부 `:695`/`:710`/`:721`, 에피소드 말미 쓰기 `:782-787`
- Test: `tests/test_phase1_metrics_io.py` (Task 1에서 만든 파일에 추가)

**Interfaces:**
- Consumes: Task 1의 `_METRICS_FIELDS`, `_SUBPROBLEM_METRICS_FIELDS`, `_CANDIDATE_SUMMARY_FIELDS`, `_append_csv_rows`, `_append_jsonl_rows`
- Produces: `_truncate_history_files(output_path: Path, start_episode: int, write_candidate_summary: bool) -> tuple[List[Dict], List[Dict]]` — 절단 후 `(metrics_rows, subproblem_metric_rows)` 를 돌려준다. 나머지 두 파일은 절단만 하고 행을 돌려주지 않는다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_phase1_metrics_io.py` 에 아래 클래스를 추가한다. import 줄에 `_truncate_history_files` 를 더한다.

```python
class TruncateHistoryFilesTests(unittest.TestCase):
    def _write_csv(self, path: Path, fields, rows) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(fields))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_keeps_only_rows_before_start_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write_csv(
                out / "metrics.csv",
                _METRICS_FIELDS,
                [{"episode": 1}, {"episode": 2}, {"episode": 3}],
            )
            self._write_csv(
                out / "subproblem_metrics.csv",
                _SUBPROBLEM_METRICS_FIELDS,
                [{"episode": 1}, {"episode": 2}],
            )

            metrics_rows, subproblem_rows = _truncate_history_files(
                out, start_episode=3, write_candidate_summary=False
            )

            self.assertEqual([row["episode"] for row in metrics_rows], ["1", "2"])
            self.assertEqual([row["episode"] for row in subproblem_rows], ["1", "2"])
            with (out / "metrics.csv").open(encoding="utf-8-sig", newline="") as file:
                on_disk = list(csv.DictReader(file))
            self.assertEqual([row["episode"] for row in on_disk], ["1", "2"])

    def test_fresh_run_removes_stale_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write_csv(out / "metrics.csv", _METRICS_FIELDS, [{"episode": 1}])
            (out / "best_action_table.jsonl").write_text(
                json.dumps({"episode": 1}) + "\n", encoding="utf-8"
            )

            metrics_rows, subproblem_rows = _truncate_history_files(
                out, start_episode=1, write_candidate_summary=False
            )

            self.assertEqual(metrics_rows, [])
            self.assertEqual(subproblem_rows, [])
            self.assertFalse((out / "metrics.csv").exists())
            self.assertFalse((out / "best_action_table.jsonl").exists())

    def test_missing_files_are_tolerated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_rows, subproblem_rows = _truncate_history_files(
                Path(tmp), start_episode=5, write_candidate_summary=True
            )
            self.assertEqual(metrics_rows, [])
            self.assertEqual(subproblem_rows, [])
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `/home/temp_id/anaconda3/envs/accord_env/bin/python -m unittest tests.test_phase1_metrics_io -v`
Expected: FAIL — `ImportError: cannot import name '_truncate_history_files'`

- [ ] **Step 3: `_truncate_history_files` 구현**

Task 1에서 추가한 `_append_jsonl_rows` 뒤에 둔다.

```python
def _truncate_history_files(
    output_path: Path,
    start_episode: int,
    write_candidate_summary: bool,
) -> tuple[List[Dict], List[Dict]]:
    """Trim per-episode history files once, before the training loop appends to them.

    새 학습(start_episode == 1)이면 이전 산출물을 지운다. resume이면 start_episode 이전 행만 남긴다.
    metrics/subproblem 행은 summary 집계에 필요하므로 돌려주고, candidate/best-action 행은
    학습 중 읽는 곳이 없으므로 메모리에 남기지 않는다.
    """

    metrics_path = output_path / "metrics.csv"
    subproblem_path = output_path / "subproblem_metrics.csv"
    candidate_path = output_path / "candidate_summary.csv"
    best_action_path = output_path / "best_action_table.jsonl"

    if start_episode <= 1:
        for path in (metrics_path, subproblem_path, candidate_path, best_action_path):
            if path.exists():
                path.unlink()
        return [], []

    metrics_rows = _read_csv_rows(metrics_path, "episode", start_episode)
    subproblem_rows = _read_csv_rows(subproblem_path, "episode", start_episode)
    if metrics_rows:
        _write_metrics(metrics_path, metrics_rows)
    elif metrics_path.exists():
        metrics_path.unlink()
    if subproblem_rows:
        _write_subproblem_metrics(subproblem_path, subproblem_rows)
    elif subproblem_path.exists():
        subproblem_path.unlink()

    if write_candidate_summary:
        candidate_rows = _read_csv_rows(candidate_path, "episode", start_episode)
        if candidate_rows:
            _write_candidate_summary(candidate_path, candidate_rows)
        elif candidate_path.exists():
            candidate_path.unlink()
        del candidate_rows
    elif candidate_path.exists():
        candidate_path.unlink()

    best_action_rows = _read_jsonl_rows(best_action_path, "episode", start_episode)
    if best_action_rows:
        _write_jsonl(best_action_path, best_action_rows)
    elif best_action_path.exists():
        best_action_path.unlink()
    del best_action_rows

    return metrics_rows, subproblem_rows
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/temp_id/anaconda3/envs/accord_env/bin/python -m unittest tests.test_phase1_metrics_io -v`
Expected: PASS (12 tests)

- [ ] **Step 5: resume 블록을 새 헬퍼로 교체**

`Phase1/pair_self_labeling.py:581-590` 의 여섯 줄(주석 포함)을 아래로 바꾼다. `validation_rows` / `validation_candidate_rows` 를 읽는 `:591-594` 는 **그대로 둔다** — validation 파일은 `validation_every` 주기에만 쓰이므로 재작성 비용이 문제가 되지 않는다.

```python
    # LINE-BY-LINE: 진행 기록 파일은 학습 시작 시 한 번만 start_episode 기준으로 잘라내고, 이후에는 append합니다.
    metrics_rows, subproblem_metric_rows = _truncate_history_files(
        output_path,
        start_episode,
        write_candidate_summary,
    )
```

`write_candidate_summary` 는 Task 4에서 파라미터로 들어온다. 이 태스크에서는 함수 시그니처에 아직 없으므로, **Task 4를 먼저 적용하지 않았다면** 이 단계에서 `train_phase1_pair_self_labeling` 시그니처(`:479`)에 `write_candidate_summary: bool = False,` 를 `objective_scope` 파라미터 뒤에 추가한다.

- [ ] **Step 6: 에피소드 루프에 에피소드 단위 수집 리스트 도입**

`:622` 의 `for episode in range(start_episode, episodes + 1):` 바로 다음 줄들 앞에 추가한다.

```python
        # LINE-BY-LINE: 이 에피소드에서 새로 생긴 행만 모읍니다. 파일에는 이 행들만 덧붙입니다.
        episode_subproblem_rows: List[Dict] = []
        episode_candidate_rows: List[Dict] = []
        episode_best_action_rows: List[Dict] = []
```

- [ ] **Step 7: 서브문제 루프의 append 대상을 에피소드 리스트로 변경**

세 곳을 바꾼다.

`:695` `candidate_rows.append(` → `episode_candidate_rows.append(`
`:710` `best_action_rows.extend(` → `episode_best_action_rows.extend(`
`:721` `subproblem_metric_rows.append(` → `episode_subproblem_rows.append(`

- [ ] **Step 8: 에피소드 말미 쓰기를 append로 교체**

`:781-787` 의 네 줄(주석 포함)을 아래로 바꾼다. `metrics_rows.append(...)` (`:760`) 는 그대로 둔다 — `_source_counts` 와 phase2 feedback 검사에 필요하다.

```python
        # LINE-BY-LINE: 이번 에피소드에서 새로 생긴 행만 덧붙입니다. 누적 재작성을 하지 않습니다.
        _append_csv_rows(output_path / "metrics.csv", _METRICS_FIELDS, [metrics_rows[-1]])
        if episode_subproblem_rows:
            subproblem_metric_rows.extend(episode_subproblem_rows)
            _append_csv_rows(
                output_path / "subproblem_metrics.csv",
                _SUBPROBLEM_METRICS_FIELDS,
                episode_subproblem_rows,
            )
        if write_candidate_summary and episode_candidate_rows:
            _append_csv_rows(
                output_path / "candidate_summary.csv",
                _CANDIDATE_SUMMARY_FIELDS,
                episode_candidate_rows,
            )
        if episode_best_action_rows:
            _append_jsonl_rows(output_path / "best_action_table.jsonl", episode_best_action_rows)
```

- [ ] **Step 9: 사용하지 않게 된 누적 리스트 제거 확인**

`candidate_rows` 와 `best_action_rows` 라는 이름이 `train_phase1_pair_self_labeling` 함수 본문에 더 이상 없어야 한다 (`validation_candidate_rows` 는 다른 변수이므로 남아 있는 것이 정상).

Run: `grep -n "^\s*candidate_rows\b\|^\s*best_action_rows\b" Phase1/pair_self_labeling.py`
Expected: 출력 없음

- [ ] **Step 10: 짧은 학습으로 실제 동작 확인**

Run:
```bash
/home/temp_id/anaconda3/envs/accord_env/bin/python main.py phase1-train-pair-self-labeling \
  --config config_mixed.yaml --episodes 3 --rollout-samples 4 \
  --heuristic-algorithms wo_first_balanced,cut_first_balanced,bevel_first_balanced \
  --min-blocks 12 --max-blocks 20 --device cpu --seed 0 \
  --output-dir output/bench/append_smoke
```
Expected: 정상 종료. 이어서 확인:
```bash
wc -l output/bench/append_smoke/metrics.csv          # 4 (헤더 1 + 에피소드 3)
grep -c "^episode," output/bench/append_smoke/metrics.csv   # 1 (헤더 중복 없음)
ls output/bench/append_smoke/candidate_summary.csv   # 없어야 함 (기본 비활성)
```

- [ ] **Step 11: 기존 테스트 회귀 확인**

Run: `/home/temp_id/anaconda3/envs/accord_env/bin/python -m unittest discover -s tests -p 'test_phase*.py' -v`
Expected: 전부 PASS

- [ ] **Step 12: 커밋**

```bash
git add Phase1/pair_self_labeling.py tests/test_phase1_metrics_io.py
git commit -m "perf(phase1): append per-episode metrics instead of rewriting history"
```

---

### Task 3: `--write-candidate-summary` CLI 플래그

Phase 2와 같은 이름·기본값·도움말로 Phase 1에도 후보 audit 파일 옵트인 플래그를 붙인다.

**Files:**
- Modify: `main.py` — 파서 `:1180-1240` 구간, 설정 에코 `:236-259` 구간, 호출부 `:280`
- Test: `tests/test_phase1_metrics_io.py`

**Interfaces:**
- Consumes: Task 2의 `train_phase1_pair_self_labeling(..., write_candidate_summary: bool = False)`
- Produces: CLI 플래그 `--write-candidate-summary` (`args.write_candidate_summary: bool`, 기본 `False`)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_phase1_metrics_io.py` 에 추가한다.

```python
class Phase1CliFlagTests(unittest.TestCase):
    def test_write_candidate_summary_defaults_to_false(self) -> None:
        from main import build_parser

        args = build_parser().parse_args(
            [
                "phase1-train-pair-self-labeling",
                "--config", "config_mixed.yaml",
                "--output-dir", "output/tmp",
            ]
        )
        self.assertFalse(args.write_candidate_summary)

    def test_write_candidate_summary_can_be_enabled(self) -> None:
        from main import build_parser

        args = build_parser().parse_args(
            [
                "phase1-train-pair-self-labeling",
                "--config", "config_mixed.yaml",
                "--output-dir", "output/tmp",
                "--write-candidate-summary",
            ]
        )
        self.assertTrue(args.write_candidate_summary)
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `/home/temp_id/anaconda3/envs/accord_env/bin/python -m unittest tests.test_phase1_metrics_io.Phase1CliFlagTests -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'write_candidate_summary'`

- [ ] **Step 3: 파서에 플래그 추가**

`main.py` 의 Phase 1 파서에서 `--resume-checkpoint` 인자 정의 바로 뒤에 추가한다. 문구는 Phase 2(`:1304-1306`)와 맞춘다.

```python
    phase1_train_pair_self_labeling_parser.add_argument(
        "--write-candidate-summary",
        action="store_true",
        help="Write per-training-candidate audit rows. Disabled by default for long training speed.",
    )
```

- [ ] **Step 4: 설정 에코와 호출부 연결**

`main.py` 설정 에코 블록에서 `print(f"- device: {args.device}")` 바로 앞에 추가한다.

```python
    print(f"- write_candidate_summary: {args.write_candidate_summary}")
```

`:280` 의 `train_phase1_pair_self_labeling(` 호출에서 `objective_scope=args.objective_scope,` 다음 줄에 추가한다.

```python
        write_candidate_summary=args.write_candidate_summary,
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `/home/temp_id/anaconda3/envs/accord_env/bin/python -m unittest tests.test_phase1_metrics_io -v`
Expected: PASS (14 tests)

- [ ] **Step 6: 플래그 켠 상태의 동작 확인**

Run:
```bash
/home/temp_id/anaconda3/envs/accord_env/bin/python main.py phase1-train-pair-self-labeling \
  --config config_mixed.yaml --episodes 2 --rollout-samples 2 \
  --heuristic-algorithms wo_first_balanced --min-blocks 12 --max-blocks 16 \
  --device cpu --seed 0 --write-candidate-summary \
  --output-dir output/bench/append_smoke_cand
```
Expected: `output/bench/append_smoke_cand/candidate_summary.csv` 가 생성되고, 헤더가 1줄만 있어야 한다.
```bash
grep -c "^episode," output/bench/append_smoke_cand/candidate_summary.csv   # 1
```

- [ ] **Step 7: 커밋**

```bash
git add main.py tests/test_phase1_metrics_io.py
git commit -m "feat(phase1): add --write-candidate-summary opt-in flag"
```

---

### Task 4: 플롯 스크립트의 `candidate_summary.csv` 부재 대응

`candidate_summary.csv` 가 기본적으로 생성되지 않게 되었으므로, 이 파일을 요구하던 스크립트가 죽지 않고 해당 그래프만 건너뛰게 한다.

**Files:**
- Modify: `scripts/plot_phase1_pair_training.py` — `_read_candidate_summary` `:76-92`, 호출부 `:49`
- Test: `tests/test_plot_phase1_pair_training.py` (기존 파일에 추가)

**Interfaces:**
- Consumes: Task 3의 플래그 동작 (파일이 없을 수 있음)
- Produces: `_read_candidate_summary(path: Path) -> list[dict[str, str]]` 가 파일 부재 시 `RuntimeError` 대신 `[]` 를 돌려준다. 빈 파일과 컬럼 누락은 기존대로 `RuntimeError` 를 유지한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_plot_phase1_pair_training.py` 에 추가한다. 파일 상단 import 에 `_read_candidate_summary` 를 더하고, `tempfile`/`pathlib.Path` 를 import 한다.

```python
class ReadCandidateSummaryTests(unittest.TestCase):
    def test_missing_file_returns_empty_list(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            rows = _read_candidate_summary(Path(tmp) / "candidate_summary.csv")

        self.assertEqual(rows, [])

    def test_empty_file_still_raises(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate_summary.csv"
            path.write_text("episode,source,score_json,learning_score_json\n", encoding="utf-8-sig")
            with self.assertRaises(RuntimeError):
                _read_candidate_summary(path)
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `/home/temp_id/anaconda3/envs/accord_env/bin/python -m unittest tests.test_plot_phase1_pair_training.ReadCandidateSummaryTests -v`
Expected: FAIL — `test_missing_file_returns_empty_list` 에서 `RuntimeError: missing candidate summary file`

- [ ] **Step 3: 부재 시 빈 목록 반환으로 변경**

`scripts/plot_phase1_pair_training.py:76-79` 를 바꾼다. 빈 파일·컬럼 누락 검사는 그대로 둔다 — 그것은 파일이 손상됐다는 뜻이므로 조용히 넘어가면 안 된다.

```python
def _read_candidate_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(
            "[CHECK][plot_phase1_pair_training._read_candidate_summary] "
            f"cause=candidate_summary_disabled path={path} "
            "hint=rerun training with --write-candidate-summary to get candidate plots"
        )
        return []
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
```

- [ ] **Step 4: 후보 기반 집계가 빈 목록을 견디는지 확인하고 안내 출력 추가**

`candidate_rows` 는 `main()` 안에서만 쓰인다 (`:49` 읽기 → `:52-55` objective_scope 필터 → `:56` `_best_agent_rows_by_episode(...)`). 따라서 감쌀 곳은 한 군데다.

`:56` 의 `proposed_rows = _best_agent_rows_by_episode(candidate_rows)` 를 아래로 바꾼다.

```python
    proposed_rows = _best_agent_rows_by_episode(candidate_rows)
    if not proposed_rows:
        print(
            "[CHECK][plot_phase1_pair_training] "
            "cause=skipped_candidate_plots reason=no_candidate_summary_rows "
            "hint=rerun training with --write-candidate-summary for candidate plots"
        )
```

그다음 `_write_plots` 가 빈 `proposed_rows` 로 죽지 않는지 **직접 확인한다**.

Run: `grep -n "def _write_plots" -A 40 scripts/plot_phase1_pair_training.py`

`proposed_rows` 를 인덱싱하거나 `max()`/`min()` 에 빈 시퀀스로 넘기는 지점이 있으면, 해당 그래프 생성만 `if proposed_rows:` 로 감싼다. 이미 빈 목록에서 안전하면 추가 변경 없이 다음 단계로 간다. `rows`(metrics) 기반 그래프는 항상 생성되어야 하므로 감싸지 않는다.

- [ ] **Step 5: 테스트 통과 확인**

Run: `/home/temp_id/anaconda3/envs/accord_env/bin/python -m unittest tests.test_plot_phase1_pair_training -v`
Expected: 전부 PASS

- [ ] **Step 6: Task 2에서 만든 run 으로 실제 실행 확인**

Run: `/home/temp_id/anaconda3/envs/accord_env/bin/python scripts/plot_phase1_pair_training.py output/bench/append_smoke`
Expected: 오류 없이 종료하고, `loss_curve.png` 와 `training_quick_status.json` 이 생성되며, 후보 그래프는 건너뛴다는 `[CHECK]` 메시지가 출력된다.

- [ ] **Step 7: 커밋**

```bash
git add scripts/plot_phase1_pair_training.py tests/test_plot_phase1_pair_training.py
git commit -m "fix(scripts): tolerate missing candidate_summary in phase1 plot script"
```

---

### Task 5: 속도·동등성·resume 검증

이 작업의 성공 기준을 실제로 확인한다. 자동 테스트로 만들기 어려운 항목이므로 수동 실행하고 결과를 기록한다.

**Files:**
- Create: `scripts/bench_phase1_append.sh`
- Create: `docs/superpowers/plans/2026-07-28-phase1-metrics-io-results.md` (측정 결과 기록)

**Interfaces:**
- Consumes: Task 2·3의 완성된 학습 경로
- Produces: 없음 (검증 산출물)

- [ ] **Step 1: 검증 스크립트 작성**

```bash
#!/usr/bin/env bash
# Phase 1 append 전환의 속도 곡선을 측정한다.
# 기존 output/phase1_pair_20k 와 동일한 설정·시드로 300 에피소드를 돌리고,
# 100 에피소드 구간별 s/ep 가 증가하지 않는지 확인한다.
set -eu

PY="${PY:-/home/temp_id/anaconda3/envs/accord_env/bin/python}"
OUT="${OUT:-output/bench/phase1_append_300}"

rm -rf "${OUT}"
start=$(date +%s)
"${PY}" main.py phase1-train-pair-self-labeling \
  --config config_mixed.yaml \
  --episodes 300 \
  --rollout-samples 64 \
  --heuristic-algorithms wo_first_balanced,cut_first_balanced,bevel_first_balanced \
  --objective-scope shared_and_series \
  --min-blocks 12 --max-blocks 80 \
  --device cuda \
  --seed 0 \
  --checkpoint-every 100 \
  --output-dir "${OUT}" \
  > "${OUT}.log" 2>&1
end=$(date +%s)

echo "총 소요: $(( end - start ))초 / 300 에피소드"
echo "구간별 s/ep (체크포인트 간격 기준):"
ls -l --time-style='+%s' "${OUT}"/checkpoints/*.pt 2>/dev/null | awk '{print $6, $7}' | sort -n | \
  awk '{ep=$2; gsub(/.*_ep0*/,"",ep); gsub(/\.pt/,"",ep);
        if(prev){printf "  ep %5d: %5.1f s/ep\n", ep, ($1-prev)/100} prev=$1}'
```

- [ ] **Step 2: 실행**

Run: `chmod +x scripts/bench_phase1_append.sh && bash scripts/bench_phase1_append.sh`
Expected: 정상 종료. 300 에피소드 소요는 기준선(ep 100~300 구간 약 20~22 s/ep)보다 짧아야 한다.

- [ ] **Step 3: 속도 곡선 판정**

기준선과 비교한다.

| 구간 | 기준선 (수정 전, `output/phase1_pair_20k`) | 합격 조건 |
|---|---|---|
| ep 200 | 20.1 s/ep | 이하 |
| ep 300 | 21.9 s/ep | 이하이며 **ep 200보다 크지 않을 것** |

ep 300 구간이 ep 200 구간보다 크면 아직 누적 비용이 남아 있는 것이므로, 남은 전체 재작성 지점을 찾아야 한다.
Run: `grep -n "_write_csv\|_write_jsonl" Phase1/pair_self_labeling.py` 로 학습 루프 안에서 호출되는 것이 없는지 확인한다.

- [ ] **Step 4: 동등성 확인**

수정 전 산출물과 같은 시드의 앞 3 에피소드를 비교한다. `output/phase1_pair_20k/metrics.csv` 의 처음 3행이 기준이다.

Run:
```bash
/home/temp_id/anaconda3/envs/accord_env/bin/python - <<'EOF'
import csv
def head(path, n=3):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)][:n]
before = head("output/phase1_pair_20k/metrics.csv")
after = head("output/bench/phase1_append_300/metrics.csv")
keys = ["episode", "problem_id", "block_count", "problem_seed", "best_source", "loss", "score_json"]
for i, (b, a) in enumerate(zip(before, after), start=1):
    diff = {k: (b.get(k), a.get(k)) for k in keys if b.get(k) != a.get(k)}
    print(f"ep{i}: {'동일' if not diff else diff}")
EOF
```
Expected: 세 에피소드 모두 `동일`. 다르면 I/O 외의 동작이 바뀐 것이므로 되돌아가 원인을 찾는다.

- [ ] **Step 5: resume 확인**

Run:
```bash
/home/temp_id/anaconda3/envs/accord_env/bin/python main.py phase1-train-pair-self-labeling \
  --config config_mixed.yaml --episodes 350 --rollout-samples 64 \
  --heuristic-algorithms wo_first_balanced,cut_first_balanced,bevel_first_balanced \
  --objective-scope shared_and_series --min-blocks 12 --max-blocks 80 \
  --device cuda --seed 0 --checkpoint-every 100 \
  --resume-checkpoint latest \
  --output-dir output/bench/phase1_append_300 \
  > output/bench/phase1_append_300_resume.log 2>&1
```
이어서 확인:
```bash
/home/temp_id/anaconda3/envs/accord_env/bin/python - <<'EOF'
import csv
with open("output/bench/phase1_append_300/metrics.csv", encoding="utf-8-sig", newline="") as fh:
    eps = [int(r["episode"]) for r in csv.DictReader(fh)]
print("행 수:", len(eps))
print("중복:", len(eps) != len(set(eps)))
print("연속:", eps == list(range(1, max(eps) + 1)))
EOF
grep -c "^episode," output/bench/phase1_append_300/metrics.csv
```
Expected: 중복 `False`, 연속 `True`, 헤더 개수 `1`.

- [ ] **Step 6: 결과 기록 후 커밋**

측정값을 `docs/superpowers/plans/2026-07-28-phase1-metrics-io-results.md` 에 표로 적는다 (구간별 s/ep 기준선 대비, 동등성 판정, resume 판정, 20,000 에피소드 재추정치).

```bash
git add scripts/bench_phase1_append.sh docs/superpowers/plans/2026-07-28-phase1-metrics-io-results.md
git commit -m "test(phase1): record append-mode speed, equivalence and resume verification"
```

---

## 자체 검토 결과

**스펙 커버리지**

| 스펙 항목 | 담당 태스크 |
|---|---|
| §3.1 리스트별 처리 | Task 2 (Step 6~9) |
| §3.2 append 헬퍼 | Task 1 |
| §3.3 resume 1회 절단 | Task 2 (`_truncate_history_files`) |
| §3.4 `--write-candidate-summary` | Task 3 |
| §4.1 동등성 | Task 5 Step 4 |
| §4.2 속도 곡선 | Task 5 Step 2~3 |
| §4.3 resume | Task 5 Step 5 |
| §4.4 플래그 | Task 3 Step 6 |
| §4.5 기존 테스트 | Task 1 Step 6, Task 2 Step 11 |
| §5 `plot_phase1_pair_training.py` 대응 | Task 4 |

누락 없음.

**타입 일관성**

- `_METRICS_FIELDS` / `_SUBPROBLEM_METRICS_FIELDS` / `_CANDIDATE_SUMMARY_FIELDS`: Task 1에서 `tuple[str, ...]` 로 정의하고 Task 2에서 같은 이름으로 사용한다.
- `_append_csv_rows(path, fields, rows)`: Task 1 정의와 Task 2 호출의 인자 순서가 일치한다.
- `_truncate_history_files(output_path, start_episode, write_candidate_summary)`: Task 2에서 정의하고 같은 태스크 Step 5에서 호출한다. 반환은 `(metrics_rows, subproblem_metric_rows)` 2-튜플로 일관된다.
- `write_candidate_summary`: Task 2 Step 5에서 시그니처에 추가하고 Task 3에서 CLI로 연결한다. 기본값 `False` 로 두 곳이 같다.

**주의**

Task 2 Step 5는 Task 3의 CLI 플래그보다 먼저 함수 파라미터를 추가한다. 태스크를 순서대로 실행하면 문제가 없으나, Task 3을 먼저 실행하면 `train_phase1_pair_self_labeling` 에 아직 파라미터가 없어 `TypeError` 가 난다. **Task 순서를 지킬 것.**
