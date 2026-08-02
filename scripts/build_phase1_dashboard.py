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
