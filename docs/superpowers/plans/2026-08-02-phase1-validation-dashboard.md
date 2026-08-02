# Phase 1 validation 대시보드 자동 갱신 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 현재 학습 중인 Phase 1 런(`output/phase1_mixed_resource_pool_v2`)의 진행 현황을 self-contained HTML 대시보드로 보여주고, validation이 찍힐 때마다 백그라운드 워처가 자동 재생성한다.

**Architecture:** 기존 `scripts/build_training_dashboard.py`의 데이터 계층(`collect_phase` 등)을 재사용하는 단일 런 생성기 `scripts/build_phase1_dashboard.py`와, `validation_summary.csv`의 최대 `train_episode`를 폴링해 생성기를 호출하는 워처 `scripts/watch_phase1_dashboard.py`를 만든다. 학습 코드는 건드리지 않는다(이미 실행 중인 프로세스에 바깥에서 붙는다).

**Tech Stack:** Python 3(표준 라이브러리만: `csv`, `json`, `argparse`, `pathlib`, `datetime`, `time`), self-contained HTML+인라인 SVG/JS(외부 의존성 0), unittest.

## Global Constraints

- No silent fallback: 필수 파일/디렉터리 부재는 `print` 진단 후 `raise RuntimeError(...)`. 평균/0/빈값으로 대체 금지 (AGENTS.md §4–5).
- 학습 코드(`main.py`, `Phase1/*`) 및 실행 중인 프로세스 미변경.
- 대상 런: `output/phase1_mixed_resource_pool_v2`, target episodes `20000`.
- 출력: `output/dashboard/phase1_v2/dashboard.html` (+ 같은 폴더 `dashboard_data.json`). `output/`는 gitignore됨 — 산출물은 커밋하지 않는다.
- 재사용 원본: `scripts/build_training_dashboard.py` (변경 없음). 차트 엔진/CSS/헬퍼는 이 파일에서 그대로 복사.
- Windows conda env `simenv` 사용: `C:\Users\hani0\anaconda3\envs\simenv\python.exe` (테스트는 stdlib만 쓰므로 어떤 python이든 되지만 일관성 위해 simenv 사용).
- 테스트 러너: unittest (`python -m unittest ...`).
- 커밋 메시지 끝에 `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Structure

- `scripts/build_phase1_dashboard.py` (신규) — 단일 Phase 1 런 → self-contained HTML + JSON 생성기. 데이터 함수 + HTML 템플릿 + CLI.
- `scripts/watch_phase1_dashboard.py` (신규) — validation 폴링 워처. 생성기를 import해 재렌더.
- `tests/test_phase1_dashboard.py` (신규) — 생성기 데이터 함수 + 워처 폴링 함수 + 생성기 스모크 unittest.
- `scripts/build_training_dashboard.py` (재사용, 변경 없음) — `collect_phase`, `_read_csv_rows`, `_to_int`, `_to_float`, `_subproblem_sources`, `_moving_average`, `_downsample`, `DEFAULT_WINDOW`, `_TEMPLATE`.

---

## Task 1: 생성기 데이터 계층 (`build_phase1_dashboard.py` 데이터 함수)

**Files:**
- Create: `scripts/build_phase1_dashboard.py`
- Test: `tests/test_phase1_dashboard.py`

**Interfaces:**
- Consumes (from `build_training_dashboard.py`): `collect_phase(run_dir: Path, target_episodes: int, window: int, now: float) -> dict`, `_read_csv_rows(path: Path) -> list[dict]`, `_to_int(v) -> int|None`, `_subproblem_sources(s: str) -> list[str]`, `_moving_average(points, window)`, `_downsample(points)`, `DEFAULT_WINDOW: int`.
- Produces (later tasks rely on these exact names):
  - `PARENT_VIEWS = ("NP_NC", "FN_FL")`
  - `phase1_best_rate_by_view(rows: list[dict], view: str) -> list[list[float]]`
  - `phase1_adoption_series(metrics_rows: list[dict], window: int) -> dict[str, list[list[float]]]` (keys: `"greedy"`, `"sample"`, `"heuristic"`)
  - `build_payload(run_dir: Path, target: int, window: int) -> dict` (shape: `{"generated_at": str, "window": int, "run": dict}` where `run` is `collect_phase` output plus keys `best_rate_np_nc`, `best_rate_fn_fl`, `best_rate_overall`, `adoption`, `best_rate_latest`)

- [ ] **Step 1: Write the failing test**

`tests/test_phase1_dashboard.py`:

```python
import json
import sys
import tempfile
import unittest
from pathlib import Path

# scripts/ 를 import 경로에 올린다 (build_phase1_dashboard 와 그 의존 build_training_dashboard 가 여기 있다).
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import build_phase1_dashboard as dash  # noqa: E402


_METRICS = (
    "episode,loss,best_source\n"
    "1,2.5,NP_NC:wo_first_balanced|FN_FL:agent_greedy\n"
    "2,2.1,NP_NC:agent_sample_5|FN_FL:cut_first_balanced\n"
    "3,1.9,NP_NC:agent_greedy|FN_FL:agent_sample_9\n"
)

_VALIDATION = (
    "train_episode,validation_view,agent_is_best,agent_rank\n"
    "100,NP_NC,1,1\n"
    "100,FN_FL,0,2\n"
    "200,NP_NC,1,1\n"
    "200,FN_FL,1,1\n"
)


def _make_run(tmp: Path) -> Path:
    run = tmp / "run"
    run.mkdir()
    (run / "metrics.csv").write_text(_METRICS, encoding="utf-8")
    (run / "validation_summary.csv").write_text(_VALIDATION, encoding="utf-8")
    return run


class TestPhase1BestRateByView(unittest.TestCase):
    def test_np_nc_view_series(self):
        rows = dash._read_csv_rows_for_test(_VALIDATION)
        series = dash.phase1_best_rate_by_view(rows, "NP_NC")
        self.assertEqual(series, [[100.0, 100.0], [200.0, 100.0]])

    def test_fn_fl_view_series(self):
        rows = dash._read_csv_rows_for_test(_VALIDATION)
        series = dash.phase1_best_rate_by_view(rows, "FN_FL")
        self.assertEqual(series, [[100.0, 0.0], [200.0, 100.0]])

    def test_overall_pools_rows_not_view_average(self):
        rows = dash._read_csv_rows_for_test(_VALIDATION)
        series = dash.phase1_best_rate_by_view(rows, "overall")
        # ep100: rows NP_NC=1, FN_FL=0 -> 50%. ep200: 1,1 -> 100%.
        self.assertEqual(series, [[100.0, 50.0], [200.0, 100.0]])


class TestPhase1AdoptionSeries(unittest.TestCase):
    def test_per_episode_fractions_window_one(self):
        rows = dash._read_csv_rows_for_test(_METRICS)
        adoption = dash.phase1_adoption_series(rows, window=1)
        # ep1: greedy(FN_FL) + heuristic(NP_NC) -> greedy 50, sample 0, heur 50
        self.assertEqual(adoption["greedy"][0], [1.0, 50.0])
        self.assertEqual(adoption["sample"][0], [1.0, 0.0])
        self.assertEqual(adoption["heuristic"][0], [1.0, 50.0])
        # ep2: sample(NP_NC) + heuristic(FN_FL) -> greedy 0, sample 50, heur 50
        self.assertEqual(adoption["greedy"][1], [2.0, 0.0])
        self.assertEqual(adoption["sample"][1], [2.0, 50.0])
        # ep3: greedy + sample -> greedy 50, sample 50, heur 0
        self.assertEqual(adoption["greedy"][2], [3.0, 50.0])
        self.assertEqual(adoption["heuristic"][2], [3.0, 0.0])


class TestBuildPayload(unittest.TestCase):
    def test_payload_shape(self):
        with tempfile.TemporaryDirectory() as td:
            run = _make_run(Path(td))
            payload = dash.build_payload(run, target=20000, window=25)
        self.assertEqual(payload["window"], 25)
        run_data = payload["run"]
        self.assertEqual(run_data["current_ep"], 3)
        self.assertEqual(run_data["target"], 20000)
        for key in ("best_rate_np_nc", "best_rate_fn_fl", "best_rate_overall", "adoption", "best_rate_latest"):
            self.assertIn(key, run_data)
        self.assertEqual(run_data["best_rate_latest"], 100.0)  # overall 최신(ep200)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\hani0\anaconda3\envs\simenv\python.exe" -m unittest tests.test_phase1_dashboard -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_phase1_dashboard'` (파일 미생성).

- [ ] **Step 3: Write minimal implementation (data layer)**

`scripts/build_phase1_dashboard.py`:

```python
"""Phase 1 단일 런 학습 대시보드 (validation 마다 재생성).

collect_phase 로 기존 지표를 뽑고, holdout best-rate 를 NP_NC/FN_FL/overall view 로
분리해 참고 artifact 스타일의 self-contained HTML 한 장을 쓴다. 외부 요청이 없으므로
파일만 열면 그대로 보인다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 같은 scripts/ 폴더의 dual-phase 대시보드에서 데이터 계층을 재사용한다.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_training_dashboard import (  # noqa: E402
    DEFAULT_WINDOW,
    collect_phase,
    _downsample,
    _moving_average,
    _read_csv_rows,
    _subproblem_sources,
    _to_int,
)

PARENT_VIEWS = ("NP_NC", "FN_FL")


def _read_csv_rows_for_test(text: str) -> list[dict[str, str]]:
    """테스트 편의: CSV 문자열을 임시 파일 없이 행 목록으로 푼다."""

    import csv

    lines = text.splitlines()
    return [dict(row) for row in csv.DictReader(lines)]


def phase1_best_rate_by_view(rows: list[dict[str, str]], view: str) -> list[list[float]]:
    """holdout best-rate 를 validation_view 로 분리한다.

    view 가 'overall' 이면 NP_NC/FN_FL 행을 pool 한다(두 view 비율의 산술평균이 아님).
    train_episode 별 100 * mean(agent_is_best) 시계열을 돌려준다.
    """

    grouped: dict[int, list[int]] = {}
    for row in rows:
        current_view = row.get("validation_view")
        if view == "overall":
            if current_view not in PARENT_VIEWS:
                continue
        elif current_view != view:
            continue
        episode = _to_int(row.get("train_episode"))
        if episode is None:
            continue
        flag = str(row.get("agent_is_best", "")).strip().lower()
        if flag in {"true", "1"}:
            grouped.setdefault(episode, []).append(1)
        elif flag in {"false", "0"}:
            grouped.setdefault(episode, []).append(0)
    series: list[list[float]] = []
    for episode in sorted(grouped):
        flags = grouped[episode]
        series.append([float(episode), 100.0 * sum(flags) / len(flags)])
    return series


def _classify_source(source: str) -> str:
    if source == "agent_greedy":
        return "greedy"
    if source.startswith("agent_sample_"):
        return "sample"
    return "heuristic"


def phase1_adoption_series(
    metrics_rows: list[dict[str, str]], window: int
) -> dict[str, list[list[float]]]:
    """train best_source 를 greedy / sample / heuristic 비율로 나눠 이동평균한다."""

    greedy: list[list[float]] = []
    sample: list[list[float]] = []
    heuristic: list[list[float]] = []
    for row in metrics_rows:
        episode = _to_int(row.get("episode"))
        if episode is None:
            continue
        sources = _subproblem_sources(row.get("best_source", ""))
        if not sources:
            continue
        total = len(sources)
        counts = {"greedy": 0, "sample": 0, "heuristic": 0}
        for source in sources:
            counts[_classify_source(source)] += 1
        greedy.append([float(episode), 100.0 * counts["greedy"] / total])
        sample.append([float(episode), 100.0 * counts["sample"] / total])
        heuristic.append([float(episode), 100.0 * counts["heuristic"] / total])
    return {
        "greedy": _downsample(_moving_average(greedy, window)),
        "sample": _downsample(_moving_average(sample, window)),
        "heuristic": _downsample(_moving_average(heuristic, window)),
    }


def build_payload(run_dir: Path, target: int, window: int) -> dict[str, Any]:
    now = datetime.now(timezone.utc).timestamp()
    run = collect_phase(run_dir, target, window, now)
    metrics_rows = _read_csv_rows(run_dir / "metrics.csv")
    validation_rows = _read_csv_rows(run_dir / "validation_summary.csv")
    run["best_rate_np_nc"] = phase1_best_rate_by_view(validation_rows, "NP_NC")
    run["best_rate_fn_fl"] = phase1_best_rate_by_view(validation_rows, "FN_FL")
    run["best_rate_overall"] = phase1_best_rate_by_view(validation_rows, "overall")
    run["adoption"] = phase1_adoption_series(metrics_rows, window)
    overall = run["best_rate_overall"]
    run["best_rate_latest"] = overall[-1][1] if overall else None
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "window": window,
        "run": run,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\hani0\anaconda3\envs\simenv\python.exe" -m unittest tests.test_phase1_dashboard -v`
Expected: PASS (7 tests: 3 best-rate, 1 adoption, 1 payload — 나머지는 후속 task에서 추가).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_phase1_dashboard.py tests/test_phase1_dashboard.py
git commit -m "$(printf 'feat(21th): Phase 1 대시보드 데이터 계층(best-rate/adoption/payload)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 2: 생성기 HTML 템플릿 + CLI (`build_phase1_dashboard.py` 렌더)

**Files:**
- Modify: `scripts/build_phase1_dashboard.py` (add `_TEMPLATE`, `render_html`, `write_dashboard`, `main`)
- Test: `tests/test_phase1_dashboard.py` (add smoke test)

**Interfaces:**
- Consumes: `build_payload` (Task 1).
- Produces: `render_html(payload: dict) -> str`, `write_dashboard(run_dir: Path, target: int, output: str|Path, window: int) -> dict` (writes `output` HTML + sibling `dashboard_data.json`, returns payload), `main() -> int`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_phase1_dashboard.py`)

```python
class TestWriteDashboard(unittest.TestCase):
    def test_writes_self_contained_html_and_json(self):
        with tempfile.TemporaryDirectory() as td:
            run = _make_run(Path(td))
            out = Path(td) / "dash" / "dashboard.html"
            payload = dash.write_dashboard(run, target=20000, output=out, window=25)
        self.assertTrue(out.exists())
        html = out.read_text(encoding="utf-8")
        self.assertNotIn("__DATA__", html)          # 데이터가 실제로 치환됐다
        self.assertIn('id="app"', html)
        self.assertIn('id="chart-val"', html)
        self.assertIn('id="chart-adopt"', html)
        self.assertIn(str(run), html)               # run 경로가 푸터에 박힌다
        data_file = out.parent / "dashboard_data.json"
        self.assertTrue(data_file.exists())
        reloaded = json.loads(data_file.read_text(encoding="utf-8"))
        self.assertEqual(reloaded["run"]["current_ep"], payload["run"]["current_ep"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\hani0\anaconda3\envs\simenv\python.exe" -m unittest tests.test_phase1_dashboard.TestWriteDashboard -v`
Expected: FAIL — `AttributeError: module 'build_phase1_dashboard' has no attribute 'write_dashboard'`.

- [ ] **Step 3: Write implementation — template + render + CLI**

3a. Add `render_html`, `write_dashboard`, `main` to `scripts/build_phase1_dashboard.py` (아래를 파일 끝, `if __name__` 위에 추가):

```python
def render_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return _TEMPLATE.replace("__DATA__", data_json)


def write_dashboard(run_dir: Path, target: int, output: str | Path, window: int) -> dict[str, Any]:
    payload = build_payload(run_dir, target, window)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload), encoding="utf-8")
    (out.parent / "dashboard_data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--target-episodes", type=int, default=20000)
    parser.add_argument("--output", default="output/dashboard/phase1_v2/dashboard.html")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"[ERROR][build_phase1_dashboard] cause=missing_run_dir path={run_dir}")
        raise RuntimeError(f"missing run directory: {run_dir}")

    payload = write_dashboard(run_dir, args.target_episodes, args.output, args.window)
    output = Path(args.output)
    print(
        f"[CHECK][build_phase1_dashboard] wrote={output} bytes={output.stat().st_size} "
        f"current_ep={payload['run']['current_ep']} "
        f"best_rate_latest={payload['run']['best_rate_latest']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

3b. Add the `_TEMPLATE` constant (place above `render_html`). **Build it by copying `scripts/build_training_dashboard.py` `_TEMPLATE` (lines 392–848) verbatim, then applying these exact edits:**

- Keep verbatim (do not re-type): the `<style>...</style>` block (build_training_dashboard.py:393–507), and JS helpers `css`/`fmt`/`fmtInt` (600–604), `tile` (606–613), `legend` (638–642), `lineChart` (644–761), the theme-toggle listener (839–844), and the trailing `renderAll();` call.
- Change the first line to: `<title>Phase 1 학습 진행 모니터</title>`.
- Replace the `<div id="app"> ... </div>` body (build_training_dashboard.py:509–594) with this single-run DOM:

```html
<div id="app">
  <header class="hdr">
    <div>
      <p class="eyebrow">MIXED 다계열 · Phase 1 self-labeling (series_only)</p>
      <h1>Phase 1 학습 진행 모니터</h1>
      <p class="sub">부모 episode를 <strong>NP_NC(22/23/24)</strong>·<strong>FN_FL(25/trans)</strong> 두 자원군으로
        나눠 pair pointer를 self-labeling으로 학습한다. episode마다 다른 무작위 문제라 raw 점수는 노이즈가 크므로,
        학습 신호는 <strong>CE loss 추세</strong>와 <strong>holdout best-rate</strong>로 읽는다.</p>
    </div>
    <button id="themeToggle" class="theme-btn" aria-label="테마 전환">◐</button>
  </header>

  <section class="tiles" aria-label="요약 지표"></section>

  <section class="card">
    <div class="card-head">
      <div><h2>학습 loss <span class="unit">(teacher 모방 CE, 이동평균)</span></h2>
        <p class="card-sub">raw(옅은 선)와 이동평균(굵은 선). 문제 크기에 따라 흔들리지만 추세가 학습을 나타낸다.</p></div>
      <div class="legend" id="loss-legend"></div>
    </div>
    <div class="chart-wrap"><svg id="chart-loss" class="chart"></svg></div>
  </section>

  <section class="card">
    <div class="card-head">
      <div><h2>holdout best-rate <span class="unit">(agent가 최선 후보, %)</span></h2>
        <p class="card-sub">고정 holdout에서 100 episode마다 측정. NP_NC/FN_FL 자원군별과 두 view를 pool한 overall.</p></div>
      <div class="legend" id="val-legend"></div>
    </div>
    <div class="chart-wrap"><svg id="chart-val" class="chart"></svg></div>
  </section>

  <section class="card">
    <div class="card-head">
      <div><h2>agent 채택률 <span class="unit">(train best-source, 이동평균)</span></h2>
        <p class="card-sub">매 episode teacher-best가 agent_greedy / agent_sample_* / 휴리스틱 중 무엇인지. agent 비중이 늘수록 정책이 teacher를 추월.</p></div>
      <div class="legend" id="adopt-legend"></div>
    </div>
    <div class="chart-wrap"><svg id="chart-adopt" class="chart"></svg></div>
  </section>

  <section class="card">
    <div class="card-head">
      <div><h2>episode당 소요 시간 <span class="unit">(checkpoint 구간별, 초)</span></h2>
        <p class="card-sub">평평해야 정상. 계속 우상향하면 누적 I/O가 쌓이는 것. checkpoint가 2개 미만이면 아직 값이 없다.</p></div>
      <div class="legend" id="speed-legend"></div>
    </div>
    <div class="chart-wrap"><svg id="chart-speed" class="chart"></svg></div>
  </section>

  <section class="card">
    <div class="card-head"><div><h2>승리 후보 분포 <span class="unit">(최근 200 episode, subproblem별 best source)</span></h2>
      <p class="card-sub">agent 샘플은 rollout 수만큼 이름이 갈리므로 agent_sample_*로 묶어 센다. 막대 합 = 자원군 2개 × episode 수.</p></div></div>
    <div class="bars" id="src-bars"></div>
  </section>

  <section class="card">
    <div class="card-head"><h2>holdout validation 상세</h2></div>
    <div class="table-wrap">
      <table id="val-table">
        <thead><tr><th>train ep</th><th>agent 최상위율</th><th>agent 평균 rank</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

  <footer class="ftr">
    <span>출처: <code id="src-run"></code></span>
    <span id="snap-note"></span>
  </footer>
</div>
<div id="tooltip" class="tooltip" role="status" aria-live="polite"></div>
```

- Replace `renderTiles` (build_training_dashboard.py:615–636) with:

```javascript
function renderTiles() {
  const r = DATA.run;
  const eta = (r.eta_h === null || r.eta_h === undefined)
    ? `${fmt(r.progress_pct)}% · 속도 측정 대기`
    : `${fmt(r.progress_pct)}% · ${fmt(r.sec_per_ep)} s/ep · 남은 ${fmt(r.eta_h)}h`;
  const html = [
    tile("p1", "학습 진행", `${fmtInt(r.current_ep)} / ${fmtInt(r.target)}`, eta, r.progress_pct, r.running),
    tile("p1", "holdout best-rate", r.best_rate_latest === null ? "–" : `${fmt(r.best_rate_latest)}%`, "최신 validation (NP_NC+FN_FL)"),
    tile("p1", "agent 최상위율", r.agent_rate_recent === null ? "–" : `${fmt(r.agent_rate_recent)}%`, "최근 200 episode (train)"),
    tile("good", "episode당 소요", r.sec_per_ep === null ? "–" : `${fmt(r.sec_per_ep)} s`, "최근 checkpoint 구간"),
  ].join("");
  document.querySelector(".tiles").innerHTML = html;
}
```

- Replace `renderBars` (763–783) with:

```javascript
function renderBars() {
  const counts = DATA.run.source_counts || [];
  const box = document.getElementById("src-bars");
  if (!counts.length) { box.innerHTML = `<p class="card-sub">아직 집계할 데이터가 없다.</p>`; return; }
  const total = DATA.run.source_total || counts.reduce((s, c) => s + c[1], 0);
  const max = counts.reduce((m, c) => Math.max(m, c[1]), 0) || 1;
  const agentWins = counts.filter((c) => c[0].startsWith("agent_")).reduce((s, c) => s + c[1], 0);
  const c = css("--p1"), good = css("--good");
  const rows = [`<p class="t-label" style="margin:0 0 2px">${total.toLocaleString()}건 중 agent 승리 ${agentWins.toLocaleString()}건 (${(100 * agentWins / total).toFixed(1)}%)</p>`];
  for (const [name, count] of counts) {
    const isAgent = name.startsWith("agent_");
    rows.push(`<div class="bar-row"><span class="nm">${name}</span>
      <span class="track"><span style="width:${100 * count / max}%;background:${isAgent ? good : c}"></span></span>
      <span class="ct">${count}</span></div>`);
  }
  box.innerHTML = rows.join("");
}
```

- Replace `renderTable` (785–799) with:

```javascript
function renderTable() {
  const body = document.querySelector("#val-table tbody");
  const rows = (DATA.run.validation_rows || []).map((r) =>
    `<tr><td>${fmtInt(r.episode)}</td>
      <td>${r.rate === null ? "–" : fmt(r.rate) + "%"}</td>
      <td>${fmt(r.rank, 2)}</td></tr>`);
  body.innerHTML = rows.join("") ||
    `<tr><td colspan="3" style="text-align:left;color:var(--ink-3)">validation은 100 episode마다 생성된다 — 아직 없음</td></tr>`;
}
```

- Replace `renderAll` (801–837) with:

```javascript
function renderAll() {
  const c = css("--p1"), c2 = css("--p2"), good = css("--good"), ink3 = css("--ink-3");
  renderTiles();

  legend("loss-legend", [{ color: c, label: `이동평균 ${DATA.window}ep` }]);
  lineChart("chart-loss", [
    { points: DATA.run.loss, color: c, width: 1, opacity: .22 },
    { points: DATA.run.loss_avg, color: c, width: 2.4, label: `loss (평균 ${DATA.window}ep)` },
  ], { zeroBase: true, yDigits: 1, tipDigits: 3 });

  legend("val-legend", [{ color: c, label: "NP_NC" }, { color: c2, label: "FN_FL" }, { color: good, label: "overall" }]);
  lineChart("chart-val", [
    { points: DATA.run.best_rate_np_nc, color: c, width: 2.2, dots: true, label: "NP_NC" },
    { points: DATA.run.best_rate_fn_fl, color: c2, width: 2.2, dots: true, label: "FN_FL" },
    { points: DATA.run.best_rate_overall, color: good, width: 2.6, dots: true, label: "overall" },
  ], { yMin: 0, yMax: 100, tickStep: 20, yDigits: 0, height: 240, tipDigits: 1, tipSuffix: "%" });

  legend("adopt-legend", [{ color: good, label: "agent_greedy" }, { color: c, label: "agent_sample_*" }, { color: ink3, label: "휴리스틱" }]);
  lineChart("chart-adopt", [
    { points: DATA.run.adoption.greedy, color: good, width: 2.4, label: "agent_greedy" },
    { points: DATA.run.adoption.sample, color: c, width: 2.4, label: "agent_sample_*" },
    { points: DATA.run.adoption.heuristic, color: ink3, width: 2.4, label: "휴리스틱" },
  ], { yMin: 0, yMax: 100, tickStep: 20, yDigits: 0, height: 240, tipDigits: 1, tipSuffix: "%" });

  legend("speed-legend", [{ color: c, label: "s/ep" }]);
  lineChart("chart-speed", [
    { points: DATA.run.speed_intervals, color: c, width: 2.4, dots: true, label: "s/ep" },
  ], { zeroBase: true, yDigits: 0, height: 220, tipDigits: 1, tipSuffix: " s/ep" });

  renderBars();
  renderTable();
  document.getElementById("src-run").textContent = DATA.run.dir;
  document.getElementById("snap-note").textContent = `스냅샷 ${DATA.generated_at} · 이동평균 창 ${DATA.window}`;
}
```

> 주의: 복사한 원본에서 Phase 2 전용인 `chart-hv` / `hv-card` / `chart-speed` 중 **hard-violation 차트 관련 DOM·JS는 위 새 DOM/renderAll에 이미 없다** — 원본의 hard-violation `lineChart("chart-hv", ...)` 호출도 renderAll 교체로 사라진다. speed 차트는 유지된다.

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\hani0\anaconda3\envs\simenv\python.exe" -m unittest tests.test_phase1_dashboard -v`
Expected: PASS (모든 Task 1 + Task 2 테스트).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_phase1_dashboard.py tests/test_phase1_dashboard.py
git commit -m "$(printf 'feat(21th): Phase 1 단일 런 대시보드 HTML 템플릿·CLI\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 3: 워처 (`watch_phase1_dashboard.py`)

**Files:**
- Create: `scripts/watch_phase1_dashboard.py`
- Test: `tests/test_phase1_dashboard.py` (add watcher tests)

**Interfaces:**
- Consumes: `build_phase1_dashboard.write_dashboard`, `build_phase1_dashboard.DEFAULT_WINDOW`; `build_training_dashboard._read_csv_rows`, `_to_int`.
- Produces: `latest_validation_episode(run_dir: Path) -> int | None`, `current_train_episode(run_dir: Path) -> int | None`, `main() -> int`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_phase1_dashboard.py`)

```python
import build_phase1_dashboard  # noqa: E402  (ensures scripts/ on path already)
import watch_phase1_dashboard as watch  # noqa: E402


class TestWatcherPolling(unittest.TestCase):
    def test_latest_validation_episode(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            (run / "validation_summary.csv").write_text(_VALIDATION, encoding="utf-8")
            self.assertEqual(watch.latest_validation_episode(run), 200)

    def test_latest_validation_episode_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(watch.latest_validation_episode(Path(td)))

    def test_current_train_episode(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            (run / "metrics.csv").write_text(_METRICS, encoding="utf-8")
            self.assertEqual(watch.current_train_episode(run), 3)

    def test_current_train_episode_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(watch.current_train_episode(Path(td)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\hani0\anaconda3\envs\simenv\python.exe" -m unittest tests.test_phase1_dashboard.TestWatcherPolling -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'watch_phase1_dashboard'`.

- [ ] **Step 3: Write implementation**

`scripts/watch_phase1_dashboard.py`:

```python
"""Phase 1 대시보드를 validation 마다 재생성하는 워처.

validation_summary.csv 의 최대 train_episode 를 폴링하다가 값이 커지면(=새 validation)
build_phase1_dashboard 로 HTML 을 다시 쓴다. 학습이 target 에 도달하면 마지막으로 한 번
렌더하고 종료한다. 학습 코드는 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_phase1_dashboard as dash  # noqa: E402
from build_training_dashboard import _read_csv_rows, _to_int  # noqa: E402


def latest_validation_episode(run_dir: Path) -> int | None:
    rows = _read_csv_rows(run_dir / "validation_summary.csv")
    episodes = [ep for ep in (_to_int(r.get("train_episode")) for r in rows) if ep is not None]
    return max(episodes) if episodes else None


def current_train_episode(run_dir: Path) -> int | None:
    rows = _read_csv_rows(run_dir / "metrics.csv")
    if not rows:
        return None
    return _to_int(rows[-1].get("episode"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--target-episodes", type=int, default=20000)
    parser.add_argument("--output", default="output/dashboard/phase1_v2/dashboard.html")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--window", type=int, default=dash.DEFAULT_WINDOW)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"[ERROR][watch_phase1_dashboard] cause=missing_run_dir path={run_dir}")
        raise RuntimeError(f"missing run directory: {run_dir}")

    def render() -> int:
        payload = dash.write_dashboard(run_dir, args.target_episodes, args.output, args.window)
        return payload["run"]["current_ep"]

    ep = render()
    last_val = latest_validation_episode(run_dir)
    print(f"[CHECK][watch_phase1_dashboard] initial render current_ep={ep} last_val={last_val}")

    while True:
        if (current_train_episode(run_dir) or 0) >= args.target_episodes:
            render()
            print("[CHECK][watch_phase1_dashboard] target reached, final render, exit")
            return 0
        time.sleep(args.interval)
        cur_val = latest_validation_episode(run_dir)
        if cur_val is not None and cur_val != last_val:
            ep = render()
            last_val = cur_val
            print(f"[CHECK][watch_phase1_dashboard] validation train_ep={cur_val} rendered current_ep={ep}")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\hani0\anaconda3\envs\simenv\python.exe" -m unittest tests.test_phase1_dashboard -v`
Expected: PASS (전체).

- [ ] **Step 5: Commit**

```bash
git add scripts/watch_phase1_dashboard.py tests/test_phase1_dashboard.py
git commit -m "$(printf 'feat(21th): validation 폴링 대시보드 워처\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 4: 회귀검증 + 실 런에 워처 기동

**Files:**
- (변경 없음 — 검증·배포 단계)

**Interfaces:**
- Consumes: 완성된 `scripts/build_phase1_dashboard.py`, `scripts/watch_phase1_dashboard.py`.

- [ ] **Step 1: 컴파일 체크 (두 신규 스크립트 포함)**

Run:
```bash
& "C:\Users\hani0\anaconda3\envs\simenv\python.exe" -m py_compile scripts/build_phase1_dashboard.py scripts/watch_phase1_dashboard.py
```
Expected: 출력 없음(성공).

- [ ] **Step 2: 대시보드 관련 전체 unit test**

Run: `& "C:\Users\hani0\anaconda3\envs\simenv\python.exe" -m unittest tests.test_phase1_dashboard -v`
Expected: 전건 PASS.

- [ ] **Step 3: 실 런 실데이터로 생성기 1회 수동 실행 (연기 검증)**

Run:
```bash
& "C:\Users\hani0\anaconda3\envs\simenv\python.exe" scripts/build_phase1_dashboard.py --run-dir output/phase1_mixed_resource_pool_v2 --target-episodes 20000 --output output/dashboard/phase1_v2/dashboard.html
```
Expected: `[CHECK][build_phase1_dashboard] wrote=...dashboard.html bytes=<큰 값> current_ep=<현재 episode> best_rate_latest=<값 또는 None>`. `output/dashboard/phase1_v2/dashboard.html`과 `dashboard_data.json` 생성 확인.

- [ ] **Step 4: 워처를 detached 백그라운드로 기동 (PowerShell)**

Run:
```powershell
$py = "C:\Users\hani0\anaconda3\envs\simenv\python.exe"
$repo = "c:\Users\hani0\OneDrive\lab\code\2026_FAB_SHI-8th"
$out = "$repo\output\dashboard\phase1_v2"
New-Item -ItemType Directory -Force $out | Out-Null
$argv = @("scripts/watch_phase1_dashboard.py","--run-dir","output/phase1_mixed_resource_pool_v2","--target-episodes","20000","--output","output/dashboard/phase1_v2/dashboard.html","--interval","60")
$p = Start-Process -FilePath $py -ArgumentList $argv -WorkingDirectory $repo -RedirectStandardOutput "$out\watch_stdout.log" -RedirectStandardError "$out\watch_stderr.log" -PassThru -WindowStyle Hidden
$p.Id | Out-File "$out\watch.pid" -Encoding ascii
Write-Output "watcher PID: $($p.Id)"
```
Expected: PID 출력. 잠시 후 `watch_stdout.log`에 `initial render current_ep=...` 로그, `dashboard.html` mtime 갱신 확인. stderr 로그 비어 있음.

- [ ] **Step 5: 체크리스트 반영 커밋**

`현재과제_진행체크리스트.md`에 대시보드 항목 한 줄 추가(예: "Phase 1 런 validation 대시보드 자동 갱신 워처 추가") 후:
```bash
git add 현재과제_진행체크리스트.md
git commit -m "$(printf 'docs(21th): Phase 1 대시보드 자동 갱신 체크리스트 반영\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Self-Review 결과

- **Spec coverage:** 단일 Phase 1 런(Task 1–2), 워처/폴링(Task 3), 로컬 self-contained HTML(Task 2), 차트 4종(loss/best-rate/adoption/sec-ep, Task 2 renderAll), 타일 4종(Task 2 renderTiles), 승자 막대·validation 표(Task 2), hard/temperature 제외(Task 2 DOM/renderAll에 없음), strict 실패(Task 1 build 없음 → Task 2/3 main의 `raise`), 테스트(Task 1–3), 60초 폴링·출력 경로(Global Constraints + Task 3/4). 모두 매핑됨.
- **Placeholder scan:** TBD/TODO 없음. 재사용 지시는 파일:line 명시(스테이블 원본 참조, 다른 task 참조 아님).
- **Type consistency:** `write_dashboard`·`build_payload`·`render_html` 시그니처가 Task 2·3에서 일치. payload 키(`run`, `best_rate_*`, `adoption`, `best_rate_latest`, `source_counts`, `speed_intervals`, `validation_rows`)를 renderAll이 그대로 사용. `latest_validation_episode`/`current_train_episode` 이름 일치.
