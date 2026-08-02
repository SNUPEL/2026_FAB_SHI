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
    _to_float,
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


def phase1_validation_table(validation_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """holdout 상세 표: 부모 view(NP_NC/FN_FL)만 pool해 tile/chart의 overall과 일치시킨다.

    train_episode 별 rate=100*mean(agent_is_best), rank=mean(agent_rank); 최근 8개만.
    """
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in validation_rows:
        if row.get("validation_view") not in PARENT_VIEWS:
            continue
        episode = _to_int(row.get("train_episode"))
        if episode is not None:
            grouped.setdefault(episode, []).append(row)
    out: list[dict[str, Any]] = []
    for episode in sorted(grouped)[-8:]:
        block = grouped[episode]
        ranks = [_to_float(r.get("agent_rank")) for r in block]
        ranks = [value for value in ranks if value is not None]
        flags = [1 if str(r.get("agent_is_best", "")).strip().lower() in {"true", "1"} else 0 for r in block]
        rate = 100.0 * sum(flags) / len(flags) if flags else None
        out.append({
            "episode": episode,
            "rate": rate,
            "rank": sum(ranks) / len(ranks) if ranks else None,
        })
    return out


def build_payload(run_dir: Path, target: int, window: int) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        print(f"[ERROR][build_phase1_dashboard] cause=missing_metrics path={metrics_path}")
        raise RuntimeError(f"missing metrics.csv: {metrics_path}")

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
    run["validation_rows"] = phase1_validation_table(validation_rows)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "window": window,
        "run": run,
    }


_TEMPLATE = r"""<title>Phase 1 학습 진행 모니터</title>
<style>
:root{
  color-scheme:light;
  --surface-0:#e9edf2; --surface-1:#ffffff; --surface-2:#f4f6f9;
  --ink:#12161c; --ink-2:#515b68; --ink-3:#8b93a0;
  --border:rgba(18,22,28,.11); --grid:rgba(18,22,28,.075); --axis:rgba(18,22,28,.28);
  --p1:#2a78d6; --p1-soft:rgba(42,120,214,.16);
  --p2:#eb6834; --p2-soft:rgba(235,104,52,.16);
  --good:#0a8a3f; --warn:#b06a00;
  --shadow:0 1px 2px rgba(18,22,28,.06),0 8px 24px rgba(18,22,28,.05);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --surface-0:#0e1115; --surface-1:#171b21; --surface-2:#1e232b;
    --ink:#eef2f7; --ink-2:#a7b0bd; --ink-3:#6b7480;
    --border:rgba(255,255,255,.10); --grid:rgba(255,255,255,.07); --axis:rgba(255,255,255,.24);
    --p1:#3987e5; --p1-soft:rgba(57,135,229,.20);
    --p2:#e06a3a; --p2-soft:rgba(224,106,58,.20);
    --good:#41b56d; --warn:#d99a2b;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --surface-0:#0e1115; --surface-1:#171b21; --surface-2:#1e232b;
  --ink:#eef2f7; --ink-2:#a7b0bd; --ink-3:#6b7480;
  --border:rgba(255,255,255,.10); --grid:rgba(255,255,255,.07); --axis:rgba(255,255,255,.24);
  --p1:#3987e5; --p1-soft:rgba(57,135,229,.20);
  --p2:#e06a3a; --p2-soft:rgba(224,106,58,.20);
  --good:#41b56d; --warn:#d99a2b;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--surface-0); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans KR",sans-serif;
  -webkit-font-smoothing:antialiased;
  padding:clamp(16px,3.5vw,40px);
}
#app{max-width:1080px;margin:0 auto;display:flex;flex-direction:column;gap:clamp(14px,2vw,22px)}
.hdr{display:flex;justify-content:space-between;align-items:flex-start;gap:16px}
.eyebrow{font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--p1);font-weight:650;margin:0 0 6px}
h1{font-size:clamp(21px,3vw,30px);line-height:1.15;margin:0;font-weight:700;letter-spacing:-.01em;text-wrap:balance}
.sub{margin:9px 0 0;color:var(--ink-2);font-size:14px;line-height:1.55;max-width:66ch}
.sub strong{color:var(--ink);font-weight:650}
.theme-btn{flex:none;width:38px;height:38px;border-radius:10px;border:1px solid var(--border);
  background:var(--surface-1);color:var(--ink);font-size:17px;cursor:pointer;box-shadow:var(--shadow)}
.theme-btn:focus-visible{outline:2px solid var(--p1);outline-offset:2px}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:13px;padding:15px 16px;
  box-shadow:var(--shadow);position:relative;overflow:hidden}
.tile::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--ink-3);opacity:.5}
.tile.p1::before{background:var(--p1);opacity:1}
.tile.p2::before{background:var(--p2);opacity:1}
.tile.good::before{background:var(--good);opacity:1}
.t-label{margin:0;font-size:12px;color:var(--ink-2);font-weight:550}
.t-val{margin:6px 0 3px;font-size:26px;font-weight:720;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;font-family:ui-monospace,"SF Mono",Menlo,monospace}
.t-ctx{margin:0;font-size:11.5px;color:var(--ink-3)}
.prog{height:5px;border-radius:3px;background:var(--surface-2);margin:8px 0 7px;overflow:hidden}
.prog span{display:block;height:100%;border-radius:3px;background:var(--ink-3)}
.tile.p1 .prog span{background:var(--p1)}
.tile.p2 .prog span{background:var(--p2)}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-left:5px;vertical-align:middle}
.dot.on{background:var(--good);animation:pulse 2s ease-in-out infinite}
.dot.off{background:var(--ink-3)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}

.card{background:var(--surface-1);border:1px solid var(--border);border-radius:15px;
  padding:clamp(15px,2vw,22px);box-shadow:var(--shadow)}
.card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap;margin-bottom:14px}
h2{font-size:16px;margin:0;font-weight:670;letter-spacing:-.01em}
h2 .unit{font-weight:500;color:var(--ink-3);font-size:13px}
.card-sub{margin:6px 0 0;font-size:12.5px;color:var(--ink-2);line-height:1.5;max-width:70ch}
.legend{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.lg{display:inline-flex;align-items:center;gap:7px;font-size:12.5px;color:var(--ink-2);font-weight:550}
.lg .sw{width:22px;height:3px;border-radius:2px}
.chart-wrap{width:100%;overflow-x:auto}
svg.chart{width:100%;height:auto;display:block}
.ax{fill:var(--ink-3);font-size:11px;font-variant-numeric:tabular-nums}
.gridline{stroke:var(--grid);stroke-width:1}
.axisline{stroke:var(--axis);stroke-width:1}

.bars{display:flex;flex-direction:column;gap:9px}
.bar-row{display:grid;grid-template-columns:170px 1fr auto;align-items:center;gap:12px;font-size:12.5px}
.bar-row .nm{color:var(--ink-2);font-family:ui-monospace,Menlo,monospace;font-size:12px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar-row .track{height:16px;border-radius:4px;background:var(--surface-2);overflow:hidden}
.bar-row .track span{display:block;height:100%;border-radius:4px}
.bar-row .ct{font-variant-numeric:tabular-nums;font-weight:640;font-family:ui-monospace,Menlo,monospace;
  min-width:44px;text-align:right}

.table-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:right;padding:9px 12px;border-bottom:1px solid var(--border);
  font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--ink-2);font-weight:600;font-size:11.5px;letter-spacing:.03em;text-transform:uppercase}
tbody tr td:first-child{font-weight:600}
.pill{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;vertical-align:middle}

.ftr{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;font-size:11.5px;color:var(--ink-3)}
.ftr code{font-family:ui-monospace,Menlo,monospace;color:var(--ink-2)}
.tooltip{position:fixed;z-index:50;pointer-events:none;opacity:0;transition:opacity .1s;
  background:var(--surface-1);border:1px solid var(--border);border-radius:9px;padding:9px 11px;
  box-shadow:var(--shadow);font-size:12px;min-width:120px}
.tt-x{font-size:11px;color:var(--ink-3);margin-bottom:5px;font-variant-numeric:tabular-nums}
.tt-row{display:flex;align-items:center;gap:8px;justify-content:space-between}
.tt-row .k{display:inline-flex;align-items:center;gap:6px;color:var(--ink-2)}
.tt-row .k .sw{width:10px;height:10px;border-radius:2px}
.tt-row .v{font-weight:650;font-variant-numeric:tabular-nums;font-family:ui-monospace,Menlo,monospace}
</style>

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

<script>
const DATA = __DATA__;

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (value, digits = 1) =>
  (value === null || value === undefined) ? "–" : Number(value).toFixed(digits);
const fmtInt = (value) =>
  (value === null || value === undefined) ? "–" : Number(value).toLocaleString();

function tile(cls, label, value, ctx, bar, live) {
  const dot = live === undefined ? "" :
    `<span class="dot ${live ? "on" : "off"}" title="${live ? "실행 중" : "멈춤"}"></span>`;
  const prog = bar === undefined ? "" :
    `<div class="prog"><span style="width:${Math.max(0, Math.min(100, bar))}%"></span></div>`;
  return `<div class="tile ${cls}"><p class="t-label">${label}${dot}</p>
    <p class="t-val">${value}</p>${prog}<p class="t-ctx">${ctx}</p></div>`;
}

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

function legend(target, items) {
  document.getElementById(target).innerHTML = items
    .map((it) => `<span class="lg"><span class="sw" style="background:${it.color}"></span>${it.label}</span>`)
    .join("");
}

function lineChart(svgId, series, opts) {
  const svg = document.getElementById(svgId);
  const W = 900, H = opts.height || 260;
  const M = { t: 14, r: 16, b: 34, l: 52 };
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  const live = series.filter((s) => s.points && s.points.length);
  if (!live.length) {
    svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" class="ax">데이터 없음</text>`;
    return;
  }
  const xs = live.flatMap((s) => s.points.map((p) => p[0]));
  const ys = live.flatMap((s) => s.points.map((p) => p[1]));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);

  // 눈금 값을 정한다. tickStep 이 주어지면 그 간격으로 고정 눈금을 쓴다.
  // yMax 는 요청값과 실제 데이터 최대치 중 큰 쪽을 쓴다 — 고정 상한이 실제 값을 화면 밖으로
  // 밀어내면 모니터링 지표를 놓치기 때문이다(hard violation 이 2를 넘는 경우 등).
  let y0, y1, ticks = [];
  if (opts.tickStep) {
    y0 = opts.yMin ?? 0;
    const dataMax = Math.max(...ys);
    const wanted = opts.yMax ?? 0;
    y1 = Math.max(wanted, Math.ceil(dataMax / opts.tickStep) * opts.tickStep);
    if (y1 <= y0) y1 = y0 + opts.tickStep;
    for (let v = y0; v <= y1 + 1e-9; v += opts.tickStep) ticks.push(v);
  } else {
    y0 = opts.zeroBase ? 0 : Math.min(...ys);
    y1 = Math.max(...ys);
    if (y1 === y0) { y1 = y0 + 1; }
    const pad = (y1 - y0) * 0.08; y1 += pad; if (!opts.zeroBase) y0 -= pad;
    for (let i = 0; i <= 4; i++) ticks.push(y0 + (y1 - y0) * i / 4);
  }
  const sx = (v) => M.l + (W - M.l - M.r) * (x1 === x0 ? 0.5 : (v - x0) / (x1 - x0));
  const sy = (v) => H - M.b - (H - M.t - M.b) * ((v - y0) / (y1 - y0));

  let out = "";
  for (const v of ticks) {
    const y = sy(v);
    out += `<line class="gridline" x1="${M.l}" y1="${y}" x2="${W - M.r}" y2="${y}"/>`;
    out += `<text class="ax" x="${M.l - 8}" y="${y + 4}" text-anchor="end">${
      Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : v.toFixed(opts.yDigits ?? 1)}</text>`;
  }
  out += `<line class="axisline" x1="${M.l}" y1="${H - M.b}" x2="${W - M.r}" y2="${H - M.b}"/>`;
  for (let i = 0; i <= 4; i++) {
    const v = x0 + (x1 - x0) * i / 4;
    out += `<text class="ax" x="${sx(v)}" y="${H - M.b + 18}" text-anchor="middle">${Math.round(v).toLocaleString()}</text>`;
  }
  for (const s of live) {
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
    out += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2}"
      stroke-opacity="${s.opacity ?? 1}" stroke-linejoin="round" stroke-linecap="round"/>`;
    if (s.dots) {
      for (const p of s.points) {
        out += `<circle cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="2.6" fill="${s.color}"/>`;
      }
    }
  }
  out += `<line class="crosshair" y1="${M.t}" y2="${H - M.b}"/><g class="hover-dots"></g>`;
  svg.innerHTML = out;

  // ── 마우스 상호작용 ───────────────────────────────────────────────
  // 커서에 가장 가까운 episode 를 찾아 계열별 값을 함께 보여준다.
  // 계열마다 x 격자가 다르므로(다운샘플 간격·진행도 차이) 계열별로 따로 최근접을 찾는다.
  const named = live.filter((s) => s.label);
  const cross = svg.querySelector(".crosshair");
  const dotGroup = svg.querySelector(".hover-dots");
  const tip = document.getElementById("tooltip");
  const digits = opts.tipDigits ?? opts.yDigits ?? 1;

  const nearest = (points, target) => {
    let best = points[0], bestGap = Math.abs(points[0][0] - target);
    for (const p of points) {
      const gap = Math.abs(p[0] - target);
      if (gap < bestGap) { best = p; bestGap = gap; }
    }
    return best;
  };

  // onX 로 대입하면 재렌더(테마 전환)마다 핸들러가 중복되지 않는다.
  svg.onmousemove = (event) => {
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const pt = svg.createSVGPoint();
    pt.x = event.clientX; pt.y = event.clientY;
    const local = pt.matrixTransform(ctm.inverse());
    if (local.x < M.l || local.x > W - M.r) { svg.onmouseleave(); return; }
    const target = x0 + (local.x - M.l) / (W - M.l - M.r) * (x1 - x0);

    const hits = named.map((s) => ({ s, p: nearest(s.points, target) }));
    if (!hits.length) return;
    const anchor = hits.reduce((a, b) =>
      Math.abs(a.p[0] - target) <= Math.abs(b.p[0] - target) ? a : b);

    cross.setAttribute("x1", sx(anchor.p[0]));
    cross.setAttribute("x2", sx(anchor.p[0]));
    cross.style.opacity = 1;
    dotGroup.innerHTML = hits.map(({ s, p }) =>
      `<circle class="hover-dot" cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}"
        r="4" fill="${s.color}"/>`).join("");

    tip.innerHTML = `<div class="tt-x">episode ${Math.round(anchor.p[0]).toLocaleString()}</div>` +
      hits.map(({ s, p }) =>
        `<div class="tt-row"><span class="k"><span class="sw" style="background:${s.color}"></span>${s.label}</span>
          <span class="v">${p[1].toFixed(digits)}${opts.tipSuffix || ""}</span></div>`).join("");
    tip.style.opacity = 1;
    const box = tip.getBoundingClientRect();
    const left = Math.min(event.clientX + 14, window.innerWidth - box.width - 8);
    const top = Math.max(8, event.clientY - box.height - 12);
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  };
  svg.onmouseleave = () => {
    cross.style.opacity = 0;
    dotGroup.innerHTML = "";
    tip.style.opacity = 0;
  };
}

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

function renderTable() {
  const body = document.querySelector("#val-table tbody");
  const rows = (DATA.run.validation_rows || []).map((r) =>
    `<tr><td>${fmtInt(r.episode)}</td>
      <td>${r.rate === null ? "–" : fmt(r.rate) + "%"}</td>
      <td>${fmt(r.rank, 2)}</td></tr>`);
  body.innerHTML = rows.join("") ||
    `<tr><td colspan="3" style="text-align:left;color:var(--ink-3)">validation은 100 episode마다 생성된다 — 아직 없음</td></tr>`;
}

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

document.getElementById("themeToggle").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = getComputedStyle(root).colorScheme.includes("dark");
  root.setAttribute("data-theme", dark ? "light" : "dark");
  renderAll();
});

renderAll();
</script>
"""


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
