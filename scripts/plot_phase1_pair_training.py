"""Plot Phase 1 pair self-labeling training metrics.

Usage:
    python scripts/plot_phase1_pair_training.py output/phase1_pair_self_labeling_ce_1000x12_80
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping


SCORE_LABELS_BY_MODE = {
    "steel_first": [
        "steel gap",
        "cut gap",
        "bevel gap",
    ],
}
WO_FIRST_SCORE_LABELS_BY_OBJECTIVE_SCOPE = {
    "shared_and_series": [
        "shared W/O gap",
        "series W/O gap",
        "shared cut gap",
        "series cut gap",
        "shared bevel gap",
        "series bevel gap",
    ],
    "series_only": [
        "series W/O gap",
        "series cut gap",
        "series bevel gap",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Phase 1 pair self-labeling metrics.")
    parser.add_argument("output_dir", help="Directory containing metrics.csv")
    parser.add_argument("--window", type=int, default=25, help="Moving-average window")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = _read_metrics(output_dir / "metrics.csv")
    candidate_rows = _read_candidate_summary(output_dir / "candidate_summary.csv")
    rows, objective_scope = _latest_objective_scope_rows(rows)
    if objective_scope:
        candidate_rows = [
            row for row in candidate_rows
            if _row_objective_scope(row) == objective_scope
        ]
    proposed_rows = _best_agent_rows_by_episode(candidate_rows)
    _write_plots(output_dir, rows, proposed_rows, window=args.window)


def _read_metrics(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"[ERROR][plot_phase1_pair_training._read_metrics] cause=missing_metrics path={path}")
        raise RuntimeError(f"missing metrics file: {path}")
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    if not rows:
        print(f"[ERROR][plot_phase1_pair_training._read_metrics] cause=empty_metrics path={path}")
        raise RuntimeError(f"empty metrics file: {path}")
    required = {"episode", "loss", "best_source", "score_json"}
    missing = required - set(rows[0])
    if missing:
        print(f"[ERROR][plot_phase1_pair_training._read_metrics] cause=missing_columns columns={sorted(missing)} path={path}")
        raise RuntimeError(f"metrics file missing columns: {sorted(missing)}")
    return rows


def _read_candidate_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"[ERROR][plot_phase1_pair_training._read_candidate_summary] cause=missing_candidate_summary path={path}")
        raise RuntimeError(f"missing candidate summary file: {path}")
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    if not rows:
        print(f"[ERROR][plot_phase1_pair_training._read_candidate_summary] cause=empty_candidate_summary path={path}")
        raise RuntimeError(f"empty candidate summary file: {path}")
    required = {"episode", "source", "score_json", "learning_score_json"}
    missing = required - set(rows[0])
    if missing:
        print(
            "[ERROR][plot_phase1_pair_training._read_candidate_summary] "
            f"cause=missing_columns columns={sorted(missing)} path={path}"
        )
        raise RuntimeError(f"candidate summary file missing columns: {sorted(missing)}")
    return rows


def _write_plots(
    output_dir: Path,
    rows: list[dict[str, str]],
    proposed_rows: Mapping[int, dict[str, str]],
    window: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[ERROR][plot_phase1_pair_training._write_plots] cause=matplotlib_import_failed error={exc}")
        raise RuntimeError("matplotlib is required to plot Phase 1 training metrics") from exc

    episodes = [int(row["episode"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    sources = [row["best_source"] for row in rows]
    proposed_sources = [_proposed_row(proposed_rows, episode)["source"] for episode in episodes]
    scores = [json.loads(_proposed_row(proposed_rows, episode)["score_json"]) for episode in episodes]
    score_mode = rows[0].get("score_mode", "steel_first")
    objective_scope = _objective_scope(rows, score_mode)
    labels = _score_labels(score_mode, objective_scope)

    _plot_loss(output_dir / "loss_curve.png", plt, episodes, losses, normalized=False, window=window)
    _plot_loss(output_dir / "loss_curve_normalized.png", plt, episodes, losses, normalized=True, window=window)
    _plot_best_sources(output_dir / "best_source_counts.png", plt, sources)
    _plot_best_sources(output_dir / "proposed_source_counts.png", plt, proposed_sources)
    _plot_agent_rate(output_dir / "agent_best_rate_curve.png", plt, episodes, sources)
    _plot_score_components(output_dir / "best_score_curve_normalized.png", plt, episodes, scores, labels, mode="minmax", window=window)
    _plot_score_components(output_dir / "best_score_curve_relative.png", plt, episodes, scores, labels, mode="relative", window=window)

    status = _quick_status(rows, losses, sources, proposed_sources)
    status.update(
        {
            "loss_curve_png": str(output_dir / "loss_curve.png"),
            "loss_curve_normalized_png": str(output_dir / "loss_curve_normalized.png"),
            "best_source_counts_png": str(output_dir / "best_source_counts.png"),
            "proposed_source_counts_png": str(output_dir / "proposed_source_counts.png"),
            "agent_best_rate_curve_png": str(output_dir / "agent_best_rate_curve.png"),
            "best_score_curve_normalized_png": str(output_dir / "best_score_curve_normalized.png"),
            "best_score_curve_relative_png": str(output_dir / "best_score_curve_relative.png"),
        }
    )
    (output_dir / "training_quick_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[CHECK][plot_phase1_pair_training] passed=true")
    print(f"- metrics_rows: {len(rows)}")
    print(f"- latest_episode: {rows[-1]['episode']}")
    print(f"- output_dir: {output_dir}")


def _best_agent_rows_by_episode(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    best_rows: dict[int, dict[str, str]] = {}
    for row in rows:
        source = row["source"]
        if source != "agent_greedy" and not source.startswith("agent_sample_"):
            continue
        episode = int(row["episode"])
        previous = best_rows.get(episode)
        if previous is None or _score_key(row["learning_score_json"]) < _score_key(previous["learning_score_json"]):
            best_rows[episode] = row
    if not best_rows:
        print("[ERROR][plot_phase1_pair_training._best_agent_rows_by_episode] cause=no_agent_candidates")
        raise RuntimeError("candidate summary has no agent_greedy/agent_sample rows")
    return best_rows


def _proposed_row(rows_by_episode: Mapping[int, dict[str, str]], episode: int) -> dict[str, str]:
    row = rows_by_episode.get(episode)
    if row is None:
        print(f"[ERROR][plot_phase1_pair_training._proposed_row] cause=missing_proposed_episode episode={episode}")
        raise RuntimeError(f"missing Proposed candidate row for episode {episode}")
    return row


def _score_key(score_json: str) -> tuple[float, ...]:
    values = json.loads(score_json)
    if not isinstance(values, list):
        print(f"[ERROR][plot_phase1_pair_training._score_key] cause=score_not_list value={score_json}")
        raise RuntimeError("score_json must be a JSON list")
    return tuple(float(value) for value in values)


def _plot_loss(path: Path, plt, episodes: list[int], losses: list[float], normalized: bool, window: int) -> None:
    values = _minmax(losses) if normalized else losses
    plt.figure(figsize=(10, 4))
    plt.plot(episodes, values, alpha=0.45, label="loss")
    plt.plot(episodes, _moving_average(values, window), linewidth=2, label=f"loss MA{window}")
    plt.xlabel("Episode")
    plt.ylabel("Min-max normalized loss" if normalized else "CE/NLL loss")
    plt.title("Phase 1 pair self-labeling loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_best_sources(path: Path, plt, sources: list[str]) -> None:
    top = Counter(sources).most_common(12)
    plt.figure(figsize=(11, 4))
    plt.bar([item[0] for item in top], [item[1] for item in top])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Best count")
    plt.title("Best source counts")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_agent_rate(path: Path, plt, episodes: list[int], sources: list[str]) -> None:
    total = 0
    rates = []
    for index, source in enumerate(sources, start=1):
        total += int(source.startswith("agent_"))
        rates.append(total / index)
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


def _objective_scope(rows: list[dict[str, str]], score_mode: str) -> str:
    """CSV 전체에서 하나의 Phase 1 objective scope를 엄격히 확인한다."""

    if score_mode != "wo_first":
        return ""
    scopes = {
        str(row.get("objective_scope") or "shared_and_series")
        for row in rows
    }
    if len(scopes) != 1:
        print(
            "[ERROR][plot_phase1_pair_training._objective_scope] "
            f"cause=mixed_objective_scopes values={sorted(scopes)}"
        )
        raise RuntimeError("metrics.csv contains multiple Phase 1 objective scopes")
    objective_scope = next(iter(scopes))
    if objective_scope not in WO_FIRST_SCORE_LABELS_BY_OBJECTIVE_SCOPE:
        print(
            "[ERROR][plot_phase1_pair_training._objective_scope] "
            f"cause=unknown_objective_scope value={objective_scope}"
        )
        raise RuntimeError(f"unknown Phase 1 objective scope: {objective_scope}")
    return objective_scope


def _row_objective_scope(row: Mapping[str, str]) -> str:
    """Interpret old rows without an explicit scope as shared_and_series."""

    return str(row.get("objective_scope") or "shared_and_series")


def _latest_objective_scope_rows(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], str]:
    """Keep the latest objective-scope segment when one output contains a transition."""

    score_mode = str(rows[-1].get("score_mode") or "steel_first")
    if score_mode != "wo_first":
        return rows, ""
    latest_scope = _row_objective_scope(rows[-1])
    scopes = {_row_objective_scope(row) for row in rows}
    if len(scopes) > 1:
        print(
            "[CHECK][plot_phase1_pair_training._latest_objective_scope_rows] "
            f"objective_scope_transition=true latest={latest_scope} "
            f"available={sorted(scopes)}"
        )
    return [row for row in rows if _row_objective_scope(row) == latest_scope], latest_scope


def _score_labels(
    score_mode: str,
    objective_scope: str = "shared_and_series",
) -> list[str]:
    if score_mode == "wo_first":
        labels = WO_FIRST_SCORE_LABELS_BY_OBJECTIVE_SCOPE.get(objective_scope)
        if labels is None:
            print(
                "[ERROR][plot_phase1_pair_training._score_labels] "
                f"cause=unknown_objective_scope value={objective_scope}"
            )
            raise RuntimeError(f"unknown Phase 1 objective scope: {objective_scope}")
        return labels
    labels = SCORE_LABELS_BY_MODE.get(score_mode)
    if labels is None:
        print(f"[ERROR][plot_phase1_pair_training._score_labels] cause=unknown_score_mode score_mode={score_mode}")
        raise RuntimeError(f"unknown score_mode: {score_mode}")
    return labels


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
            title = "Phase 1 pair self-labeling normalized Proposed best-of-K score trend"
        elif mode == "relative":
            ma = _moving_average(values, window)
            base = ma[0] if ma and ma[0] != 0 else 1.0
            plotted = [value / base for value in ma]
            ylabel = "Relative score vs initial MA, lower is better"
            title = "Phase 1 pair self-labeling relative Proposed best-of-K score trend"
        else:
            print(f"[ERROR][plot_phase1_pair_training._plot_score_components] cause=unknown_mode mode={mode}")
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


def _quick_status(
    rows: list[dict[str, str]],
    losses: list[float],
    sources: list[str],
    proposed_sources: list[str],
) -> dict:
    source_counts = Counter(sources)
    agent_best_count = sum(count for source, count in source_counts.items() if source.startswith("agent_"))
    recent = sources[-50:] if len(sources) >= 50 else sources
    recent_agent_count = sum(1 for source in recent if source.startswith("agent_"))
    return {
        "episodes": len(rows),
        "latest_episode": int(rows[-1]["episode"]),
        "loss_first": losses[0],
        "loss_latest": losses[-1],
        "loss_min": min(losses),
        "agent_best_count": agent_best_count,
        "agent_best_rate": agent_best_count / len(rows),
        "last50_agent_best_rate": recent_agent_count / len(recent),
        "top_sources": source_counts.most_common(10),
        "proposed_source_counts": Counter(proposed_sources).most_common(10),
    }


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 0:
        print(f"[ERROR][plot_phase1_pair_training._moving_average] cause=non_positive_window window={window}")
        raise RuntimeError("window must be positive")
    return [
        sum(values[max(0, index - window + 1) : index + 1])
        / len(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def _minmax(values: Iterable[float]) -> list[float]:
    data = list(values)
    if not data:
        print("[ERROR][plot_phase1_pair_training._minmax] cause=empty_values")
        raise RuntimeError("cannot normalize empty values")
    low = min(data)
    high = max(data)
    if high == low:
        return [0.0 for _ in data]
    return [(value - low) / (high - low) for value in data]


if __name__ == "__main__":
    main()
