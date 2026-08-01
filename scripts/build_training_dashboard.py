"""Phase 1 / Phase 2 동시 학습의 진행 현황 대시보드를 만든다.

두 run 디렉터리의 metrics.csv / validation_summary.csv 와 checkpoint 파일 시각을 읽어
자체 완결형 HTML 한 장을 쓴다. 외부 요청이 없으므로 파일만 열면 그대로 보인다.

사용:
    python scripts/build_training_dashboard.py \
        --phase1-dir output/phase1_run_260728 \
        --phase2-dir output/phase2_run_260728 \
        --output output/dashboard/training_dashboard.html

metrics.csv 에는 시각 컬럼이 없으므로 sec/ep 와 ETA 는 checkpoint 파일의 mtime 간격에서 구한다.
checkpoint 가 2개 미만이면 해당 값은 None 으로 두고 화면에서 '-' 로 표시한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 차트에 실을 최대 점 개수. 20000 에피소드에서도 파일이 비대해지지 않게 균등 추출한다.
MAX_CHART_POINTS = 1200
# 이동평균 창.
DEFAULT_WINDOW = 25

CHECKPOINT_EPISODE_RE = re.compile(r"_ep(\d+)\.pt$")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """CSV 를 읽는다. 학습이 쓰는 중이라 마지막 줄이 잘려 있을 수 있으므로 그런 행은 버린다."""

    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if "\x00" in text:
        text = text.replace("\x00", "")
    lines = text.splitlines()
    if len(lines) < 2:
        return []
    reader = csv.DictReader(lines)
    rows: list[dict[str, str]] = []
    for row in reader:
        # 잘린 마지막 행: 컬럼 수가 모자라면 None 값이 생긴다.
        if any(value is None for value in row.values()):
            continue
        rows.append(dict(row))
    return rows


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def _to_int(value: str | None) -> int | None:
    number = _to_float(value)
    return None if number is None else int(number)


def _downsample(points: list[list[float]], limit: int = MAX_CHART_POINTS) -> list[list[float]]:
    """균등 간격으로 솎되 마지막 점은 반드시 남긴다(최신값이 화면에서 잘리면 안 된다)."""

    if len(points) <= limit:
        return points
    step = len(points) / limit
    picked = [points[int(index * step)] for index in range(limit)]
    if picked[-1] is not points[-1]:
        picked[-1] = points[-1]
    return picked


def _moving_average(points: list[list[float]], window: int) -> list[list[float]]:
    if window <= 1 or not points:
        return points
    out: list[list[float]] = []
    total = 0.0
    queue: list[float] = []
    for episode, value in points:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        out.append([episode, total / len(queue)])
    return out


def _subproblem_sources(best_source: str) -> list[str]:
    """'22:lpt_batch|23:agent_sample_4' 형태를 후보 이름 목록으로 푼다."""

    if not best_source:
        return []
    sources: list[str] = []
    for chunk in best_source.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        _, _, source = chunk.partition(":")
        source = (source or chunk).strip()
        if source:
            sources.append(source)
    return sources


def _agent_rate(rows: list[dict[str, str]], field: str = "best_source") -> float | None:
    """subproblem 단위로 agent 후보가 이긴 비율(%)."""

    total = 0
    agent = 0
    for row in rows:
        for source in _subproblem_sources(row.get(field, "")):
            total += 1
            if source.startswith("agent_"):
                agent += 1
    return None if total == 0 else 100.0 * agent / total


def _source_counts(rows: list[dict[str, str]], field: str = "best_source") -> list[list[Any]]:
    """subproblem 승자를 후보 종류별로 센다.

    rollout_samples 가 64면 agent 후보가 agent_sample_1..64 로 쪼개져 개별 이름의 빈도가
    전부 한 자릿수가 된다. 그 상태로 상위 N개만 보여주면 전체의 20%도 담기지 않으므로
    (실측: Phase 1 은 400건 중 78건), agent_sample_* 는 하나로 묶어 센다.
    """

    counts: dict[str, int] = {}
    sample_variants: set[str] = set()
    for row in rows:
        for source in _subproblem_sources(row.get(field, "")):
            if source.startswith("agent_sample_"):
                sample_variants.add(source)
                key = "agent_sample_*"
            else:
                key = source
            counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    # 라벨의 종수는 '생성된 샘플 수'가 아니라 '이 구간에서 한 번이라도 이긴 이름의 수'다.
    # (Phase 1 은 sample_index=1 을 greedy 로 재사용해 agent_sample_1 이 없으므로 최대 63종,
    #  Phase 2 는 greedy 를 따로 두어 최대 64종. 게다가 한 번도 못 이긴 이름은 여기 안 잡힌다.)
    return [
        [f"{name} (이긴 변형 {len(sample_variants)}종)" if name == "agent_sample_*" else name, count]
        for name, count in ordered
    ]


def _run_start_time(run_dir: Path) -> float | None:
    """학습 시작 시각. `<run_dir>.pid` 의 mtime 을 쓴다.

    이 파일은 기동 시 한 번 쓰고 이후 건드리지 않으므로 mtime 이 곧 기동 시각이다.
    `.log` 는 쓰지 않는다 — 학습이 계속 append 하므로 리눅스에서 ctime/mtime 이 매 쓰기마다
    갱신되어 '지금'에 수렴한다(실측: 로그 ctime 0초 전 vs 실제 기동 1624초 전).
    launch 스크립트를 거치지 않아 .pid 가 없으면 None 을 돌려주고 보정을 건너뛴다.
    """

    pid_path = run_dir.parent / f"{run_dir.name}.pid"
    if not pid_path.exists():
        return None
    return float(pid_path.stat().st_mtime)


def _checkpoint_timing(run_dir: Path) -> dict[str, Any]:
    """checkpoint mtime 간격에서 구간 속도를 만든다. metrics.csv 에 시각 컬럼이 없어서 필요하다."""

    checkpoint_dir = run_dir / "checkpoints"
    stamps: list[tuple[int, float]] = []
    if checkpoint_dir.is_dir():
        for path in checkpoint_dir.glob("*.pt"):
            match = CHECKPOINT_EPISODE_RE.search(path.name)
            if not match:
                continue
            stamps.append((int(match.group(1)), path.stat().st_mtime))
    stamps.sort()
    # 체크포인트가 1개뿐이면 구간이 안 나온다. 실행 시작 시각을 episode 0 으로 삼아 첫 구간을 만든다.
    # 시작 시각은 stdout 리다이렉트 로그의 생성 시각이며, 없으면 이 보정을 건너뛴다.
    start_at = _run_start_time(run_dir)
    if start_at is not None and stamps and start_at < stamps[0][1]:
        stamps.insert(0, (0, start_at))
    intervals: list[list[float]] = []
    for (prev_ep, prev_at), (ep, at) in zip(stamps, stamps[1:]):
        span_ep = ep - prev_ep
        if span_ep > 0 and at > prev_at:
            intervals.append([ep, (at - prev_at) / span_ep])
    recent = intervals[-5:]
    sec_per_ep = sum(value for _, value in recent) / len(recent) if recent else None
    return {
        "intervals": intervals,
        "sec_per_ep": sec_per_ep,
        "last_checkpoint_at": stamps[-1][1] if stamps else None,
        "checkpoint_count": len(stamps),
    }


def _is_running(run_dir: Path, now: float) -> bool:
    """metrics.csv 가 최근에 갱신됐는지로 판단한다. 에피소드가 느려도 15분이면 충분한 여유다."""

    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        return False
    return (now - metrics.stat().st_mtime) < 15 * 60


def _validation_rate_series(rows: list[dict[str, str]], agent_field: str) -> list[list[float]]:
    """train_episode 별 agent 최상위율(%) 시계열."""

    grouped: dict[int, list[int]] = {}
    for row in rows:
        episode = _to_int(row.get("train_episode"))
        if episode is None:
            continue
        sources = _subproblem_sources(row.get(agent_field, "")) or [row.get(agent_field, "")]
        for source in sources:
            if not source:
                continue
            grouped.setdefault(episode, []).append(1 if source.startswith("agent_") else 0)
    series: list[list[float]] = []
    for episode in sorted(grouped):
        flags = grouped[episode]
        series.append([float(episode), 100.0 * sum(flags) / len(flags)])
    return series


def _phase1_validation_rate(rows: list[dict[str, str]]) -> list[list[float]]:
    """Phase 1 validation 은 agent_is_best 플래그를 그대로 쓴다."""

    grouped: dict[int, list[int]] = {}
    for row in rows:
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


def collect_phase(run_dir: Path, target_episodes: int, window: int, now: float) -> dict[str, Any]:
    metrics = _read_csv_rows(run_dir / "metrics.csv")
    validation = _read_csv_rows(run_dir / "validation_summary.csv")

    loss_points: list[list[float]] = []
    hard_points: list[list[float]] = []
    for row in metrics:
        episode = _to_int(row.get("episode"))
        loss = _to_float(row.get("loss"))
        if episode is not None and loss is not None:
            loss_points.append([float(episode), loss])
        hard = _to_float(row.get("hard_violation_count"))
        if episode is not None and hard is not None:
            hard_points.append([float(episode), hard])

    timing = _checkpoint_timing(run_dir)
    current_ep = _to_int(metrics[-1].get("episode")) if metrics else 0
    current_ep = current_ep or 0
    sec_per_ep = timing["sec_per_ep"]
    remaining = max(0, target_episodes - current_ep)
    eta_h = (sec_per_ep * remaining / 3600.0) if sec_per_ep else None

    recent = metrics[-200:]
    return {
        "dir": str(run_dir),
        "current_ep": current_ep,
        "target": target_episodes,
        "progress_pct": (100.0 * current_ep / target_episodes) if target_episodes else 0.0,
        "running": _is_running(run_dir, now),
        "sec_per_ep": sec_per_ep,
        "eta_h": eta_h,
        "loss_first": loss_points[0][1] if loss_points else None,
        "loss_latest": loss_points[-1][1] if loss_points else None,
        "loss": _downsample(loss_points),
        "loss_avg": _downsample(_moving_average(loss_points, window)),
        "hard": _downsample(hard_points),
        "hard_latest": hard_points[-1][1] if hard_points else None,
        "agent_rate_recent": _agent_rate(recent),
        "agent_rate_all": _agent_rate(metrics),
        # agent_sample_* 를 묶었으므로 항목 수가 적다. 잘라내지 않고 전부 싣는다 —
        # 막대 합이 subproblem 총건수와 맞아야 분포로 읽을 수 있다.
        "source_counts": _source_counts(recent),
        "source_total": sum(count for _, count in _source_counts(recent)),
        "source_episodes": len(recent),
        "speed_intervals": timing["intervals"],
        # agent 가 teacher 를 이겼는지는 '전체 후보 중 승자'(best_source)로만 판정할 수 있다.
        # agent_best_source 는 'agent 후보 중 최선'이라 정의상 항상 agent_* 이므로 쓰면 안 된다.
        "validation_rate": (
            _phase1_validation_rate(validation)
            if validation and "agent_is_best" in validation[0]
            else _validation_rate_series(validation, "best_source")
        ),
        "validation_rows": _validation_table(validation),
    }


def _validation_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """train_episode 별 holdout 요약. 최근 8개만 싣는다."""

    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        episode = _to_int(row.get("train_episode"))
        if episode is not None:
            grouped.setdefault(episode, []).append(row)
    out: list[dict[str, Any]] = []
    for episode in sorted(grouped)[-8:]:
        block = grouped[episode]
        # rank 컬럼 이름이 phase 별로 다르다: Phase 1 은 agent_rank, Phase 2 는 agent_best_rank.
        ranks = [
            _to_float(r.get("agent_best_rank") if r.get("agent_best_rank") not in (None, "") else r.get("agent_rank"))
            for r in block
        ]
        ranks = [value for value in ranks if value is not None]
        makespans = [_to_float(r.get("makespan")) for r in block]
        makespans = [value for value in makespans if value is not None]
        if "agent_is_best" in block[0]:
            flags = [1 if str(r.get("agent_is_best", "")).lower() in {"true", "1"} else 0 for r in block]
            rate = 100.0 * sum(flags) / len(flags) if flags else None
        else:
            # 위와 같은 이유로 best_source 를 본다.
            rate = _agent_rate(block, "best_source")
        out.append({
            "episode": episode,
            "rate": rate,
            "rank": sum(ranks) / len(ranks) if ranks else None,
            "makespan": sum(makespans) / len(makespans) if makespans else None,
        })
    return out


def build_payload(phase1_dir: Path, phase2_dir: Path, target: int, window: int) -> dict[str, Any]:
    now = datetime.now(timezone.utc).timestamp()
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "window": window,
        "phase1": collect_phase(phase1_dir, target, window, now),
        "phase2": collect_phase(phase2_dir, target, window, now),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", required=True)
    parser.add_argument("--phase2-dir", required=True)
    parser.add_argument("--output", default="output/dashboard/training_dashboard.html")
    parser.add_argument("--target-episodes", type=int, default=20000)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    args = parser.parse_args()

    phase1_dir = Path(args.phase1_dir)
    phase2_dir = Path(args.phase2_dir)
    for run_dir in (phase1_dir, phase2_dir):
        if not run_dir.is_dir():
            print(f"[ERROR][build_training_dashboard] cause=missing_run_dir path={run_dir}")
            raise RuntimeError(f"missing run directory: {run_dir}")

    payload = build_payload(phase1_dir, phase2_dir, args.target_episodes, args.window)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(payload), encoding="utf-8")
    (output.parent / "dashboard_data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    p1 = payload["phase1"]
    p2 = payload["phase2"]
    print(
        f"[CHECK][build_training_dashboard] wrote={output} bytes={output.stat().st_size} "
        f"phase1_ep={p1['current_ep']} phase2_ep={p2['current_ep']}"
    )
    return 0


def render_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return _TEMPLATE.replace("__DATA__", data_json)


_TEMPLATE = r"""<title>Phase 1 · Phase 2 학습 진행 모니터</title>
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
      <p class="eyebrow" id="eyebrow">MIXED 다계열 · 동시 병렬 학습</p>
      <h1>Phase 1 · Phase 2 학습 진행 모니터</h1>
      <p class="sub">self-labeling 학습의 현재까지 스냅샷 — 각 episode는 서로 다른 무작위 문제라 raw 점수는 노이즈가 크므로,
        학습 신호는 <strong>CE loss 추세</strong>와 <strong>agent가 teacher 휴리스틱을 이기는 비율</strong>로 읽는다.</p>
    </div>
    <button id="themeToggle" class="theme-btn" aria-label="테마 전환">◐</button>
  </header>

  <section class="tiles" aria-label="요약 지표"></section>

  <section class="card">
    <div class="card-head">
      <div>
        <h2>학습 loss <span class="unit">(teacher 모방 CE)</span></h2>
        <p class="card-sub">episode별 raw(옅은 선)와 이동평균(굵은 선). 문제 크기에 따라 흔들리지만 추세가 학습을 나타낸다.</p>
      </div>
      <div class="legend" id="loss-legend"></div>
    </div>
    <div class="chart-wrap"><svg id="chart-loss" class="chart"></svg></div>
  </section>

  <section class="card">
    <div class="card-head">
      <div>
        <h2>에피소드당 소요 시간 <span class="unit">(checkpoint 100 에피소드 구간별, 초)</span></h2>
        <p class="card-sub">평평해야 정상이다. 계속 우상향하면 누적 I/O가 다시 쌓이고 있다는 뜻 —
          이전 학습에서는 Phase 1이 20초에서 99초까지 단조 증가했다.
          <strong>첫 점은 낮게 나온다</strong>: checkpoint가 validation보다 먼저 저장되므로 첫 구간에만
          validation 비용이 빠져 있다. 두 번째 점부터 서로 비교할 것.</p>
      </div>
      <div class="legend" id="speed-legend"></div>
    </div>
    <div class="chart-wrap"><svg id="chart-speed" class="chart"></svg></div>
  </section>

  <section class="card">
    <div class="card-head">
      <div>
        <h2>holdout agent 최상위율 <span class="unit">(agent 후보가 teacher를 이긴 비율, %)</span></h2>
        <p class="card-sub">고정된 검증 문제에서 100 episode마다 측정 — 학습의 실제 진척을 보여주는 지표.</p>
      </div>
      <div class="legend" id="val-legend"></div>
    </div>
    <div class="chart-wrap"><svg id="chart-val" class="chart"></svg></div>
  </section>

  <section class="card" id="hv-card">
    <div class="card-head">
      <div>
        <h2>Phase 2 실행 가능성 <span class="unit">(episode별 hard violation 수)</span></h2>
        <p class="card-sub">스케줄이 hard 제약을 위반하지 않는지 — 0에 붙어 있어야 정상.</p>
      </div>
    </div>
    <div class="chart-wrap"><svg id="chart-hv" class="chart"></svg></div>
  </section>

  <section class="card">
    <div class="card-head">
      <div>
        <h2>승리 후보 분포 <span class="unit">(최근 200 episode, subproblem별 best source)</span></h2>
        <p class="card-sub">teacher 휴리스틱과 agent 후보 중 어느 쪽이 이겼는지 — agent 비중이 늘수록 학습이 teacher를 추월.
          agent 샘플은 rollout 수만큼 이름이 갈리므로(agent_sample_1..N) 하나로 묶어 센다.
          막대 합은 subproblem 총 건수와 같다 — Phase 1은 episode당 자원군 2개, Phase 2는 Bay 5개.</p>
      </div>
    </div>
    <div class="bars" id="src-bars"></div>
  </section>

  <section class="card">
    <div class="card-head"><h2>holdout validation 상세</h2></div>
    <div class="table-wrap">
      <table id="val-table">
        <thead><tr><th>Phase</th><th>train ep</th><th>agent 최상위율</th><th>agent 평균 rank</th><th>평균 makespan</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

  <footer class="ftr">
    <span>출처: <code id="src-p1"></code> · <code id="src-p2"></code></span>
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
  const p1 = DATA.phase1, p2 = DATA.phase2;
  const etaText = (p) => p.eta_h === null || p.eta_h === undefined
    ? `${fmt(p.progress_pct)}% · 속도 측정 대기`
    : `${fmt(p.progress_pct)}% · ${fmt(p.sec_per_ep)} s/ep · 남은 ${fmt(p.eta_h)}h`;
  const html = [
    tile("p1", "Phase 1 진행", `${fmtInt(p1.current_ep)} / ${fmtInt(p1.target)}`,
         etaText(p1), p1.progress_pct, p1.running),
    tile("p2", "Phase 2 진행", `${fmtInt(p2.current_ep)} / ${fmtInt(p2.target)}`,
         etaText(p2), p2.progress_pct, p2.running),
    tile("p1", "Phase 1 agent 최상위율",
         p1.agent_rate_recent === null ? "–" : `${fmt(p1.agent_rate_recent)}%`,
         `최근 200 episode 기준`),
    tile("p2", "Phase 2 agent 최상위율",
         p2.agent_rate_recent === null ? "–" : `${fmt(p2.agent_rate_recent)}%`,
         `최근 200 episode 기준`),
    tile("good", "Phase 2 제약 위반",
         p2.hard_latest === null ? "–" : fmtInt(p2.hard_latest),
         "최신 episode hard violation"),
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
  const rows = [];
  for (const [tag, phase, color] of [["P1", DATA.phase1, css("--p1")], ["P2", DATA.phase2, css("--p2")]]) {
    const counts = phase.source_counts || [];
    if (!counts.length) continue;
    const total = phase.source_total || counts.reduce((s, c) => s + c[1], 0);
    const max = counts.reduce((m, c) => Math.max(m, c[1]), 0) || 1;
    const agentWins = counts.filter((c) => c[0].startsWith("agent_")).reduce((s, c) => s + c[1], 0);
    rows.push(`<p class="t-label" style="margin:${rows.length ? "14px" : "0"} 0 2px">
      ${tag} — ${total.toLocaleString()}건 중 agent 승리 ${agentWins.toLocaleString()}건
      (${(100 * agentWins / total).toFixed(1)}%)</p>`);
    for (const [name, count] of counts) {
      const isAgent = name.startsWith("agent_");
      rows.push(`<div class="bar-row"><span class="nm">${name}</span>
        <span class="track"><span style="width:${100 * count / max}%;background:${isAgent ? css("--good") : color}"></span></span>
        <span class="ct">${count}</span></div>`);
    }
  }
  document.getElementById("src-bars").innerHTML = rows.join("") ||
    `<p class="card-sub">아직 집계할 데이터가 없다.</p>`;
}

function renderTable() {
  const body = document.querySelector("#val-table tbody");
  const rows = [];
  for (const [tag, phase, color] of [["Phase 1", DATA.phase1, css("--p1")], ["Phase 2", DATA.phase2, css("--p2")]]) {
    for (const r of (phase.validation_rows || [])) {
      rows.push(`<tr><td><span class="pill" style="background:${color}"></span>${tag}</td>
        <td>${fmtInt(r.episode)}</td>
        <td>${r.rate === null ? "–" : fmt(r.rate) + "%"}</td>
        <td>${fmt(r.rank, 2)}</td>
        <td>${r.makespan === null ? "–" : fmtInt(Math.round(r.makespan))}</td></tr>`);
    }
  }
  body.innerHTML = rows.join("") ||
    `<tr><td colspan="5" style="text-align:left;color:var(--ink-3)">validation은 100 episode마다 생성된다 — 아직 없음</td></tr>`;
}

function renderAll() {
  const p1c = css("--p1"), p2c = css("--p2");
  renderTiles();
  legend("loss-legend", [{ color: p1c, label: "Phase 1" }, { color: p2c, label: "Phase 2" }]);
  // CE loss 는 음수가 될 수 없으므로 하한을 0 으로 고정한다.
  // raw 는 옅게 깔고 툴팁에는 이동평균만 싣는다 — 네 계열을 다 띄우면 읽기 어렵다.
  lineChart("chart-loss", [
    { points: DATA.phase1.loss, color: p1c, width: 1, opacity: .22 },
    { points: DATA.phase2.loss, color: p2c, width: 1, opacity: .22 },
    { points: DATA.phase1.loss_avg, color: p1c, width: 2.4, label: `Phase 1 (평균 ${DATA.window}ep)` },
    { points: DATA.phase2.loss_avg, color: p2c, width: 2.4, label: `Phase 2 (평균 ${DATA.window}ep)` },
  ], { zeroBase: true, yDigits: 1, tipDigits: 3 });

  legend("speed-legend", [{ color: p1c, label: "Phase 1" }, { color: p2c, label: "Phase 2" }]);
  lineChart("chart-speed", [
    { points: DATA.phase1.speed_intervals, color: p1c, width: 2.4, dots: true, label: "Phase 1" },
    { points: DATA.phase2.speed_intervals, color: p2c, width: 2.4, dots: true, label: "Phase 2" },
  ], { zeroBase: true, yDigits: 0, height: 220, tipDigits: 1, tipSuffix: " s/ep" });

  legend("val-legend", [{ color: p1c, label: "Phase 1" }, { color: p2c, label: "Phase 2" }]);
  // 비율이므로 0~100 고정, 20단위 눈금.
  lineChart("chart-val", [
    { points: DATA.phase1.validation_rate, color: p1c, width: 2.4, dots: true, label: "Phase 1" },
    { points: DATA.phase2.validation_rate, color: p2c, width: 2.4, dots: true, label: "Phase 2" },
  ], { yMin: 0, yMax: 100, tickStep: 20, yDigits: 0, height: 220, tipDigits: 1, tipSuffix: "%" });

  // 정상값이 0이므로 0~2 고정, 1단위 눈금. 위반이 2를 넘으면 상한이 따라 올라간다.
  lineChart("chart-hv", [
    { points: DATA.phase2.hard, color: p2c, width: 1.6, label: "Phase 2 위반" },
  ], { yMin: 0, yMax: 2, tickStep: 1, yDigits: 0, height: 200, tipDigits: 0, tipSuffix: "건" });

  renderBars();
  renderTable();
  document.getElementById("src-p1").textContent = DATA.phase1.dir;
  document.getElementById("src-p2").textContent = DATA.phase2.dir;
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


if __name__ == "__main__":
    sys.exit(main())
