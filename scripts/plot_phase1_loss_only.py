"""Plot Phase 1 loss curves using only metrics.csv.

Usage:
    python scripts/plot_phase1_loss_only.py output/phase1_pair_v9_capacity_actual8_ppb_lcp6 --window 100
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot Phase 1 loss curves from metrics.csv only.")
    parser.add_argument("output_dir", help="Directory containing metrics.csv")
    parser.add_argument("--window", type=int, default=25, help="Moving-average window")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    rows = _read_loss_rows(output_dir / "metrics.csv")
    _write_loss_plots(output_dir, rows, window=args.window)


def _read_loss_rows(path: Path) -> list[dict[str, float | int]]:
    if not path.exists():
        print(f"[ERROR][plot_phase1_loss_only._read_loss_rows] cause=missing_metrics path={path}")
        raise RuntimeError(f"missing metrics file: {path}")
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        print(f"[ERROR][plot_phase1_loss_only._read_loss_rows] cause=empty_metrics path={path}")
        raise RuntimeError(f"empty metrics file: {path}")
    required = {"episode", "loss"}
    missing = required - set(rows[0])
    if missing:
        print(f"[ERROR][plot_phase1_loss_only._read_loss_rows] cause=missing_columns columns={sorted(missing)} path={path}")
        raise RuntimeError(f"metrics file missing columns: {sorted(missing)}")

    parsed: list[dict[str, float | int]] = []
    for line_number, row in enumerate(rows, start=2):
        try:
            episode = int(row["episode"])
            loss = float(row["loss"])
        except (TypeError, ValueError) as exc:
            print(
                "[ERROR][plot_phase1_loss_only._read_loss_rows] "
                f"cause=invalid_numeric_value line={line_number} episode={row.get('episode')} loss={row.get('loss')}"
            )
            raise RuntimeError(f"invalid numeric value in metrics file at line {line_number}") from exc
        parsed.append({"episode": episode, "loss": loss})
    return parsed


def _write_loss_plots(output_dir: Path, rows: list[dict[str, float | int]], window: int) -> None:
    if window <= 0:
        print(f"[ERROR][plot_phase1_loss_only._write_loss_plots] cause=invalid_window value={window}")
        raise ValueError("window must be positive")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[ERROR][plot_phase1_loss_only._write_loss_plots] cause=matplotlib_import_failed error={exc}")
        raise RuntimeError("matplotlib is required to plot Phase 1 loss curves") from exc

    episodes = [int(row["episode"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    raw_path = output_dir / "loss_curve.png"
    normalized_path = output_dir / "loss_curve_normalized.png"
    _plot_loss(raw_path, plt, episodes, losses, normalized=False, window=window)
    _plot_loss(normalized_path, plt, episodes, losses, normalized=True, window=window)
    status = {
        "metrics_rows": len(rows),
        "latest_episode": episodes[-1],
        "window": window,
        "loss_curve_png": str(raw_path),
        "loss_curve_normalized_png": str(normalized_path),
    }
    (output_dir / "loss_only_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[CHECK][plot_phase1_loss_only] passed=true")
    print(f"- metrics_rows: {len(rows)}")
    print(f"- latest_episode: {episodes[-1]}")
    print(f"- output_dir: {output_dir}")


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


def _moving_average(values: Sequence[float], window: int) -> list[float]:
    result: list[float] = []
    running_sum = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(float(value))
        running_sum += float(value)
        if len(queue) > window:
            running_sum -= queue.pop(0)
        result.append(running_sum / len(queue))
    return result


def _minmax(values: Sequence[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if high == low:
        return [0.0 for _ in values]
    return [(float(value) - low) / (high - low) for value in values]


if __name__ == "__main__":
    main()
