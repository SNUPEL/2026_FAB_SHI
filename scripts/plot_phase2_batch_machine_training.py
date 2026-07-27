"""Plot Phase 2 merged batch-machine self-labeling training metrics.

`scripts/plot_phase1_pair_training.py`와 동일한 형식/스타일로 Phase 2 학습
진행 곡선을 만든다. Phase 2 metrics.csv의 `best_source`는 Bay 단위
`22:<source>|23:<source>|...` 형태라 subproblem별 후보 이름으로 분해해 집계한다.
Phase 2 기본 학습은 candidate_summary를 쓰지 않으므로(agent best-of-K 별도 점수
없음) best score 곡선은 metrics.csv의 teacher-best 점수(episode 승리 후보)를 쓴다.

Usage:
    python scripts/plot_phase2_batch_machine_training.py output/phase2_upstream_wo
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List


# Phase 2 raw 사전식 score 순서와 동일한 라벨
PHASE2_SCORE_LABELS = [
    "hard violation",
    "makespan",
    "cut length gap",
    "W/O count gap",
    "bevel quantity gap",
    "occupancy gap",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Phase 2 batch-machine self-labeling metrics.")
    parser.add_argument("output_dir", help="Directory containing metrics.csv")
    parser.add_argument("--window", type=int, default=25, help="Moving-average window")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = _read_metrics(output_dir / "metrics.csv")
    _write_plots(output_dir, rows, window=args.window)


def _read_metrics(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"[ERROR][plot_phase2_batch_machine_training._read_metrics] cause=missing_metrics path={path}")
        raise RuntimeError(f"missing metrics file: {path}")
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    if not rows:
        print(f"[ERROR][plot_phase2_batch_machine_training._read_metrics] cause=empty_metrics path={path}")
        raise RuntimeError(f"empty metrics file: {path}")
    required = {"episode", "loss", "best_source", "score_json"}
    missing = required - set(rows[0])
    if missing:
        print(
            "[ERROR][plot_phase2_batch_machine_training._read_metrics] "
            f"cause=missing_columns columns={sorted(missing)} path={path}"
        )
        raise RuntimeError(f"metrics file missing columns: {sorted(missing)}")
    return rows


def _write_plots(output_dir: Path, rows: list[dict[str, str]], window: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[ERROR][plot_phase2_batch_machine_training._write_plots] cause=matplotlib_import_failed error={exc}")
        raise RuntimeError("matplotlib is required to plot Phase 2 training metrics") from exc

    episodes = [int(row["episode"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    sources = [row["best_source"] for row in rows]
    scores = [json.loads(row["score_json"]) for row in rows]
    labels = PHASE2_SCORE_LABELS

    _plot_loss(output_dir / "loss_curve.png", plt, episodes, losses, normalized=False, window=window)
    _plot_loss(output_dir / "loss_curve_normalized.png", plt, episodes, losses, normalized=True, window=window)
    _plot_best_sources(output_dir / "best_source_counts.png", plt, sources)
    _plot_agent_rate(output_dir / "agent_best_rate_curve.png", plt, episodes, sources)
    _plot_score_components(output_dir / "best_score_curve_normalized.png", plt, episodes, scores, labels, mode="minmax", window=window)
    _plot_score_components(output_dir / "best_score_curve_relative.png", plt, episodes, scores, labels, mode="relative", window=window)

    status = _quick_status(rows, losses, sources)
    status.update(
        {
            "loss_curve_png": str(output_dir / "loss_curve.png"),
            "loss_curve_normalized_png": str(output_dir / "loss_curve_normalized.png"),
            "best_source_counts_png": str(output_dir / "best_source_counts.png"),
            "agent_best_rate_curve_png": str(output_dir / "agent_best_rate_curve.png"),
            "best_score_curve_normalized_png": str(output_dir / "best_score_curve_normalized.png"),
            "best_score_curve_relative_png": str(output_dir / "best_score_curve_relative.png"),
        }
    )
    (output_dir / "training_quick_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[CHECK][plot_phase2_batch_machine_training] passed=true")
    print(f"- metrics_rows: {len(rows)}")
    print(f"- latest_episode: {rows[-1]['episode']}")
    print(f"- output_dir: {output_dir}")


def _subproblem_sources(source: str) -> list[str]:
    """parent best_source를 Bay별 후보 이름으로 나눈다.

    Phase 2는 '22:<source>|23:<source>|...' 형태로 Bay 단위 승리 후보를 쓴다.
    """

    return [
        part.split(":", 1)[1] if ":" in part else part
        for part in source.split("|")
        if part
    ]


def _plot_loss(path: Path, plt, episodes: list[int], losses: list[float], normalized: bool, window: int) -> None:
    values = _minmax(losses) if normalized else losses
    plt.figure(figsize=(10, 4))
    plt.plot(episodes, values, alpha=0.45, label="loss")
    plt.plot(episodes, _moving_average(values, window), linewidth=2, label=f"loss MA{window}")
    plt.xlabel("Episode")
    plt.ylabel("Min-max normalized loss" if normalized else "CE/NLL loss")
    plt.title("Phase 2 batch-machine self-labeling loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_best_sources(path: Path, plt, sources: list[str]) -> None:
    subproblem_sources = [candidate for source in sources for candidate in _subproblem_sources(source)]
    top = Counter(subproblem_sources).most_common(12)
    plt.figure(figsize=(11, 4))
    plt.bar([item[0] for item in top], [item[1] for item in top])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Best count")
    plt.title("Best source counts (per Bay subproblem)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_agent_rate(path: Path, plt, episodes: list[int], sources: list[str]) -> None:
    agent_count = 0
    subproblem_count = 0
    rates = []
    for source in sources:
        for candidate in _subproblem_sources(source):
            subproblem_count += 1
            agent_count += int(candidate.startswith("agent_"))
        rates.append(agent_count / subproblem_count if subproblem_count else 0.0)
    plt.figure(figsize=(10, 4))
    plt.plot(episodes, rates, linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative agent-best rate")
    plt.ylim(0, 1)
    plt.title("Agent candidate selected as best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_score_components(
    path: Path,
    plt,
    episodes: list[int],
    scores: list[list[float]],
    labels: list[str],
    mode: str,
    window: int,
) -> None:
    plt.figure(figsize=(12, 6))
    for index, label in enumerate(labels):
        values = [float(score[index]) for score in scores if len(score) > index]
        if not values:
            continue
        if mode == "minmax":
            plotted = _moving_average(_minmax(values), window)
            ylabel = "Min-max normalized score, lower is better"
            title = "Phase 2 batch-machine normalized teacher-best score trend"
        elif mode == "relative":
            ma = _moving_average(values, window)
            base = ma[0] if ma and ma[0] != 0 else 1.0
            plotted = [value / base for value in ma]
            ylabel = "Relative score vs initial MA, lower is better"
            title = "Phase 2 batch-machine relative teacher-best score trend"
        else:
            print(f"[ERROR][plot_phase2_batch_machine_training._plot_score_components] cause=unknown_mode mode={mode}")
            raise RuntimeError(f"unknown score plot mode: {mode}")
        plt.plot(episodes[: len(plotted)], plotted, linewidth=1.8, label=label)
    if mode == "relative":
        plt.axhline(1.0, color="black", linewidth=1, alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _quick_status(rows: list[dict[str, str]], losses: list[float], sources: list[str]) -> dict:
    subproblem_sources = [candidate for source in sources for candidate in _subproblem_sources(source)]
    agent_best_count = sum(1 for candidate in subproblem_sources if candidate.startswith("agent_"))
    recent = sources[-50:] if len(sources) >= 50 else sources
    recent_subproblem_sources = [candidate for source in recent for candidate in _subproblem_sources(source)]
    recent_agent_count = sum(1 for candidate in recent_subproblem_sources if candidate.startswith("agent_"))
    if not subproblem_sources or not recent_subproblem_sources:
        print("[ERROR][plot_phase2_batch_machine_training._quick_status] cause=empty_best_source")
        raise RuntimeError("metrics best_source column has no candidate names")
    return {
        "episodes": len(rows),
        "latest_episode": int(rows[-1]["episode"]),
        "loss_first": losses[0],
        "loss_latest": losses[-1],
        "loss_min": min(losses),
        "agent_best_count": agent_best_count,
        "agent_best_rate": agent_best_count / len(subproblem_sources),
        "last50_agent_best_rate": recent_agent_count / len(recent_subproblem_sources),
        "top_sources": Counter(subproblem_sources).most_common(10),
    }


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 0:
        print(f"[ERROR][plot_phase2_batch_machine_training._moving_average] cause=non_positive_window window={window}")
        raise RuntimeError("window must be positive")
    return [
        sum(values[max(0, index - window + 1) : index + 1])
        / len(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def _minmax(values: Iterable[float]) -> list[float]:
    data = list(values)
    if not data:
        print("[ERROR][plot_phase2_batch_machine_training._minmax] cause=empty_values")
        raise RuntimeError("cannot normalize empty values")
    low = min(data)
    high = max(data)
    if high == low:
        return [0.0 for _ in data]
    return [(value - low) / (high - low) for value in data]


if __name__ == "__main__":
    main()
