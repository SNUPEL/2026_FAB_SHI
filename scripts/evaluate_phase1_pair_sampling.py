"""Evaluate Phase 1 pair policy with best-of-K stochastic sampling.

Usage:
    python scripts/evaluate_phase1_pair_sampling.py \
      --checkpoint output/phase1_pair_v2_1000x12_80/phase1_pair_pointer_best.pt \
      --wo-xlsx input/절단03~04_NP물량_마스킹_WO_수정_260618.xlsx \
      --samples 256 \
      --heuristic-algorithms business6 \
      --output-dir output/phase1_actual_6heuristic_proposed_s256
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from Train.algorithm.phase1_pair_self_labeling import (
    PHASE1_PAIR_ENV_FEATURE_NAMES,
    PHASE1_PAIR_FEATURE_NAMES,
    Phase1PairCandidate,
    _score_bay_loads,
    run_phase1_pair_policy_rollout,
)
from Train.algorithm.phase1_self_labeling import PHASE1_SELF_LABEL_HEURISTIC_BANK
from Train.algorithm.phase1_self_labeling import run_phase1_heuristic_candidate
from Train.network.phase1_pointer import Phase1PairPointerPolicy
from Utils.phase1_episode_dataset import build_phase1_actual_workday_jobs


BUSINESS6_HEURISTICS = [
    "steel_first_balanced",
    "cut_first_balanced",
    "bevel_first_balanced",
    "steel_first_long_cut_preferred",
    "cut_first_long_cut_preferred",
    "bevel_first_long_cut_preferred",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 pair policy with proposed sampling K.")
    parser.add_argument("--checkpoint", required=True, help="Phase1PairPointerPolicy checkpoint path")
    parser.add_argument("--wo-xlsx", required=True, help="Actual W/O Excel/CSV path")
    parser.add_argument("--bay-ids", default="22,23,24", help="Comma-separated Bay ids")
    parser.add_argument("--workdays", default="20260331,20260407,20260408,20260413,20260414,20260415,20260424,20260429")
    parser.add_argument("--gyel", default="NP")
    parser.add_argument("--samples", type=int, default=64, help="Number of stochastic policy rollouts for Proposed")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--score-mode", default="steel_first", choices=["steel_first"])
    parser.add_argument("--heuristic-algorithms", default="all8", help="Comma-separated heuristic names or all8")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    bay_ids = [item.strip() for item in args.bay_ids.split(",") if item.strip()]
    workdays = [item.strip() for item in args.workdays.split(",") if item.strip()]
    heuristic_algorithms = _resolve_heuristics(args.heuristic_algorithms)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = f"sampling{args.samples}"

    model = _load_model(Path(args.checkpoint))
    problems = build_phase1_actual_workday_jobs(
        source_path=args.wo_xlsx,
        workdays=workdays,
        bay_ids=bay_ids,
        gyel=args.gyel,
    )

    summary_rows = []
    candidate_rows = []
    sample_rows = []
    proposed_scores = []
    proposed_best_count = 0
    for problem_index, payload in enumerate(problems, start=1):
        jobs = payload["jobs"]
        metadata = payload["metadata"]
        problem_id = str(metadata.get("problem_id") or metadata.get("workday") or f"ACT{problem_index:05d}")
        block_count = int(metadata.get("block_count") or len(jobs))

        proposed, raw_samples = _best_sampling_candidate(
            jobs=jobs,
            bay_ids=bay_ids,
            model=model,
            samples=args.samples,
            temperature=args.temperature,
            seed=args.seed + problem_index * 100_000,
            score_mode=args.score_mode,
        )
        actual = _actual_history_candidate(jobs=jobs, bay_ids=bay_ids)
        proposed_source = proposed.source
        method_candidates = [actual, proposed]
        for algorithm in heuristic_algorithms:
            method_candidates.append(_heuristic_candidate_for_eval(jobs, bay_ids, algorithm))
        ranked = sorted(
            [(_score_bay_loads(candidate.bay_loads, args.score_mode), candidate) for candidate in method_candidates],
            key=lambda item: item[0],
        )
        rank_by_source = {candidate.source: rank for rank, (_, candidate) in enumerate(ranked, start=1)}
        proposed_score = _score_bay_loads(proposed.bay_loads, args.score_mode)
        actual_score = _score_bay_loads(actual.bay_loads, args.score_mode)
        best_score, best_candidate = ranked[0]
        proposed_rank = rank_by_source[proposed_source]
        proposed_is_best = int(proposed_rank == 1)
        proposed_best_count += proposed_is_best
        proposed_scores.append(proposed_score)

        summary_rows.append(
            {
                "validation_episode": problem_index,
                "problem_id": problem_id,
                "block_count": block_count,
                "actual_score_json": json.dumps(list(actual_score), ensure_ascii=False),
                "actual_rank": rank_by_source["actual_history"],
                "proposed_source": proposed_source,
                "proposed_score_json": json.dumps(list(proposed_score), ensure_ascii=False),
                "best_source": best_candidate.source,
                "best_score_json": json.dumps(list(best_score), ensure_ascii=False),
                "proposed_rank": proposed_rank,
                "proposed_is_best": proposed_is_best,
                "candidate_count": len(method_candidates),
                "sample_count": args.samples,
            }
        )
        for candidate in method_candidates:
            score = _score_bay_loads(candidate.bay_loads, args.score_mode)
            candidate_rows.append(
                _score_row(problem_index, problem_id, block_count, candidate.source, score, rank_by_source[candidate.source])
            )
        for sample_index, sample in enumerate(raw_samples, start=1):
            score = _score_bay_loads(sample.bay_loads, args.score_mode)
            sample_rows.append(
                _score_row(problem_index, problem_id, block_count, sample.source, score, sample_index)
            )

    summary_csv = output_dir / f"{output_prefix}_summary.csv"
    candidate_csv = output_dir / f"{output_prefix}_candidates.csv"
    raw_sample_csv = output_dir / f"{output_prefix}_raw_samples.csv"
    report_json = output_dir / f"{output_prefix}_report.json"
    _write_csv(summary_csv, summary_rows)
    _write_csv(candidate_csv, candidate_rows)
    _write_csv(raw_sample_csv, sample_rows)
    plot_paths = _write_metric_plots(output_dir=output_dir, rows=candidate_rows, score_mode=args.score_mode)
    report = {
        "checkpoint": str(args.checkpoint),
        "wo_xlsx": str(args.wo_xlsx),
        "workdays": workdays,
        "bay_ids": bay_ids,
        "score_mode": args.score_mode,
        "samples": args.samples,
        "problem_count": len(problems),
        "proposed_best_rate": proposed_best_count / len(problems) if problems else 0.0,
        "proposed_mean_score": _mean_score(proposed_scores),
        "summary_csv": str(summary_csv),
        "candidate_csv": str(candidate_csv),
        "raw_sample_csv": str(raw_sample_csv),
        **plot_paths,
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[CHECK][evaluate_phase1_pair_sampling] passed=true")
    print(f"- problem_count: {len(problems)}")
    print(f"- proposed_best_rate: {report['proposed_best_rate']:.6f}")
    print(f"- output_dir: {output_dir}")


def _load_model(checkpoint_path: Path) -> Phase1PairPointerPolicy:
    if not checkpoint_path.exists():
        print(f"[ERROR][evaluate_phase1_pair_sampling._load_model] cause=missing_checkpoint path={checkpoint_path}")
        raise RuntimeError(f"missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("pair_feature_names") != PHASE1_PAIR_FEATURE_NAMES:
        print(f"[ERROR][evaluate_phase1_pair_sampling._load_model] cause=pair_feature_mismatch path={checkpoint_path}")
        raise RuntimeError("checkpoint pair features do not match current code")
    if checkpoint.get("env_feature_names") != PHASE1_PAIR_ENV_FEATURE_NAMES:
        print(f"[ERROR][evaluate_phase1_pair_sampling._load_model] cause=env_feature_mismatch path={checkpoint_path}")
        raise RuntimeError("checkpoint env features do not match current code")
    hidden_dim = int(checkpoint.get("hidden_dim", 0))
    if hidden_dim <= 0:
        print(f"[ERROR][evaluate_phase1_pair_sampling._load_model] cause=invalid_hidden_dim hidden_dim={hidden_dim}")
        raise RuntimeError("checkpoint hidden_dim is invalid")
    model = Phase1PairPointerPolicy(
        pair_feature_dim=len(PHASE1_PAIR_FEATURE_NAMES),
        env_feature_dim=len(PHASE1_PAIR_ENV_FEATURE_NAMES),
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _best_sampling_candidate(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    model: Phase1PairPointerPolicy,
    samples: int,
    temperature: float,
    seed: int,
    score_mode: str,
) -> tuple[Phase1PairCandidate, list[Phase1PairCandidate]]:
    if samples <= 0:
        print(f"[ERROR][evaluate_phase1_pair_sampling._best_sampling_candidate] cause=non_positive_samples samples={samples}")
        raise RuntimeError("samples must be positive")
    raw_samples = [
        run_phase1_pair_policy_rollout(
            jobs=jobs,
            bay_ids=bay_ids,
            model=model,
            temperature=temperature,
            seed=seed + sample_index,
            source=f"proposed_sample_{sample_index + 1}",
            selection="sample",
        )
        for sample_index in range(samples)
    ]
    best = min(raw_samples, key=lambda candidate: _score_bay_loads(candidate.bay_loads, score_mode))
    proposed = Phase1PairCandidate(
        source=f"proposed_sampling{samples}",
        transitions=best.transitions,
        assignments=best.assignments,
        bay_loads=best.bay_loads,
    )
    return proposed, raw_samples


def _heuristic_candidate_for_eval(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    algorithm: str,
) -> Phase1PairCandidate:
    """Run comparison heuristics without Proposed's long-cut hard mask."""

    heuristic = run_phase1_heuristic_candidate(
        jobs=jobs,
        bay_ids=bay_ids,
        algorithm=algorithm,
        long_cut_hard_mask=False,
    )
    return Phase1PairCandidate(
        source=algorithm,
        transitions=[],
        assignments=dict(heuristic.assignments),
        bay_loads=dict(heuristic.bay_loads),
    )


def _actual_history_candidate(jobs: Mapping[str, object], bay_ids: Sequence[str]) -> Phase1PairCandidate:
    bay_set = {str(bay_id) for bay_id in bay_ids}
    bay_loads = {
        str(bay_id): {
            "steel_quantity_sum": 0,
            "cut_length_sum": 0.0,
            "bevel_quantity_sum": 0,
            "long_cut_bay24_count": 0,
            "wo_count": 0,
            "block_count": 0,
        }
        for bay_id in bay_ids
    }
    block_totals: dict[str, dict[str, float | int | set[str]]] = {}
    for job in jobs.values():
        block_id = str(getattr(job, "block_set_id"))
        source_bay = str(getattr(job, "source_cut_bay", "") or "")
        if source_bay not in bay_set:
            print(
                "[ERROR][evaluate_phase1_pair_sampling._actual_history_candidate] "
                f"cause=actual_bay_outside_scope block_id={block_id} source_bay={source_bay}"
            )
            raise RuntimeError(f"actual source bay outside scope: {source_bay}")
        steel = int(getattr(job, "steel_quantity"))
        cut_length = float(getattr(job, "cut_length"))
        bevel = int(getattr(job, "bevel_quantity"))
        bay_loads[source_bay]["steel_quantity_sum"] += steel
        bay_loads[source_bay]["cut_length_sum"] += cut_length
        bay_loads[source_bay]["bevel_quantity_sum"] += bevel
        bay_loads[source_bay]["wo_count"] += 1
        block = block_totals.setdefault(
            block_id,
            {"cut_length_sum": 0.0, "bays": set(), "source_bays": set()},
        )
        block["cut_length_sum"] = float(block["cut_length_sum"]) + cut_length
        block["bays"].add(source_bay)
        block["source_bays"].add(source_bay)

    assignments: dict[str, str] = {}
    for block_id, block in block_totals.items():
        touched_bays = sorted(str(bay) for bay in block["bays"])
        assignments[block_id] = touched_bays[0] if len(touched_bays) == 1 else "SPLIT:" + "|".join(touched_bays)
        for bay in touched_bays:
            bay_loads[bay]["block_count"] += 1
        if float(block["cut_length_sum"]) > 1000 and "24" in touched_bays:
            bay_loads["24"]["long_cut_bay24_count"] += 1

    return Phase1PairCandidate(
        source="actual_history",
        transitions=[],
        assignments=assignments,
        bay_loads=bay_loads,
    )


def _resolve_heuristics(text: str) -> list[str]:
    if text.strip().lower() in {"all", "all8"}:
        return list(PHASE1_SELF_LABEL_HEURISTIC_BANK)
    if text.strip().lower() in {"business6", "all6"}:
        return list(BUSINESS6_HEURISTICS)
    return [item.strip() for item in text.split(",") if item.strip()]


def _score_row(validation_episode: int, problem_id: str, block_count: int, source: str, score: tuple, rank: int) -> dict:
    row = {
        "validation_episode": validation_episode,
        "problem_id": problem_id,
        "block_count": block_count,
        "source": source,
        "rank": rank,
        "score_json": json.dumps(list(score), ensure_ascii=False),
    }
    for index, value in enumerate(score):
        row[f"score_{index}"] = value
    return row


def _mean_score(scores: Sequence[tuple]) -> list[float]:
    if not scores:
        return []
    width = len(scores[0])
    return [
        round(sum(float(score[index]) for score in scores) / len(scores), 12)
        for index in range(width)
    ]


def _write_metric_plots(output_dir: Path, rows: Sequence[Mapping], score_mode: str) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[ERROR][evaluate_phase1_pair_sampling._write_metric_plots] cause=matplotlib_import_failed error={exc}")
        raise RuntimeError("matplotlib is required to write phase1 sampling metric plots") from exc
    metric_specs = _metric_specs(score_mode)
    plot_paths = {}
    for key, score_field, title, ylabel in metric_specs:
        path = output_dir / f"{key}.png"
        _plot_metric(plt, path, rows, score_field, title, ylabel)
        plot_paths[f"{key}_png"] = str(path)
    combined = output_dir / "metric_all_objectives.png"
    _plot_metric_grid(plt, combined, rows, metric_specs)
    plot_paths["metric_all_objectives_png"] = str(combined)
    return plot_paths


def _metric_specs(score_mode: str) -> list[tuple[str, str, str, str]]:
    if score_mode == "steel_first":
        return [
            ("metric_steel_gap", "score_0", "Steel quantity workload gap", "Steel gap"),
            ("metric_cut_gap", "score_1", "Cut length workload gap", "Cut gap"),
            ("metric_bevel_gap", "score_2", "Bevel quantity workload gap", "Bevel gap"),
        ]
    print(f"[ERROR][evaluate_phase1_pair_sampling._metric_specs] cause=unknown_score_mode score_mode={score_mode}")
    raise RuntimeError(f"unknown_score_mode: {score_mode}")


def _plot_metric(plt, path: Path, rows: Sequence[Mapping], score_field: str, title: str, ylabel: str) -> None:
    groups = _plot_groups(rows)
    plt.figure(figsize=(11, 4.8))
    for source_index, source in enumerate(groups["sources"]):
        x_values = []
        y_values = []
        for problem_index, problem_id in enumerate(groups["problems"], start=1):
            row = groups["by_key"].get((problem_id, source))
            if row is None:
                continue
            x_values.append(problem_index + _offset(source_index, len(groups["sources"])))
            y_values.append(float(row[score_field]))
        plt.scatter(x_values, y_values, marker=_marker_for(source), s=_size_for(source), label=_label_for(source), alpha=0.9)
    plt.xticks(range(1, len(groups["problems"]) + 1), [_short_problem(problem) for problem in groups["problems"]], rotation=25)
    plt.xlabel("Actual workday problem")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_metric_grid(plt, path: Path, rows: Sequence[Mapping], specs: Sequence[tuple[str, str, str, str]]) -> None:
    groups = _plot_groups(rows)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for axis, (_, score_field, title, ylabel) in zip(axes, specs):
        for source_index, source in enumerate(groups["sources"]):
            x_values = []
            y_values = []
            for problem_index, problem_id in enumerate(groups["problems"], start=1):
                row = groups["by_key"].get((problem_id, source))
                if row is None:
                    continue
                x_values.append(problem_index + _offset(source_index, len(groups["sources"])))
                y_values.append(float(row[score_field]))
            axis.scatter(x_values, y_values, marker=_marker_for(source), s=_size_for(source), label=_label_for(source), alpha=0.9)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.set_xticks(range(1, len(groups["problems"]) + 1))
        axis.set_xticklabels([_short_problem(problem) for problem in groups["problems"]], rotation=25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=5, loc="upper center")
    fig.supxlabel("Actual workday problem")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_groups(rows: Sequence[Mapping]) -> dict:
    problems = []
    sources = []
    by_key = {}
    for row in rows:
        problem_id = str(row["problem_id"])
        source = str(row["source"])
        if problem_id not in problems:
            problems.append(problem_id)
        if source not in sources:
            sources.append(source)
        by_key[(problem_id, source)] = row
    return {"problems": problems, "sources": _ordered_sources(sources), "by_key": by_key}


def _ordered_sources(sources: Sequence[str]) -> list[str]:
    preferred = ["actual_history"] + sorted(source for source in sources if source.startswith("proposed_sampling"))
    return [source for source in preferred if source in sources] + sorted(source for source in sources if source not in preferred)


def _offset(index: int, count: int) -> float:
    if count <= 1:
        return 0.0
    return (index - (count - 1) / 2) * min(0.08, 0.7 / count)


def _marker_for(source: str) -> str:
    if source == "actual_history":
        return "s"
    if source.startswith("proposed_sampling"):
        return "D"
    if "bevel" in source:
        return "o"
    if "cut" in source:
        return "^"
    return "x"


def _size_for(source: str) -> int:
    return 70 if source == "actual_history" or source.startswith("proposed_sampling") else 38


def _label_for(source: str) -> str:
    names = {
        "actual_history": "Actual",
        "steel_first_balanced": "MSF+MOP",
        "cut_first_balanced": "LCF+MOP",
        "bevel_first_balanced": "MBF+MOP",
        "long_cut_first_balanced": "Long",
        "steel_first_long_cut_preferred": "MSF+LCP",
        "cut_first_long_cut_preferred": "LCF+LCP",
        "bevel_first_long_cut_preferred": "MBF+LCP",
        "long_cut_first_long_cut_preferred": "Long+Long",
    }
    if source.startswith("proposed_sampling"):
        return f"Proposed {source.replace('proposed_', '')}"
    return names.get(source, source)


def _short_problem(problem_id: str) -> str:
    return problem_id.replace("ACTUAL_2026", "")


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        print(f"[ERROR][evaluate_phase1_pair_sampling._write_csv] cause=no_rows path={path}")
        raise RuntimeError(f"no rows to write: {path}")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
