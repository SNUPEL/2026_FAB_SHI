"""Evaluate Phase 1 pair policy or heuristic baselines on actual workday problems.

Usage:
    python scripts/evaluate_phase1_pair_sampling.py \
      --checkpoint output/phase1_pair_v2_1000x12_80/phase1_pair_pointer_best.pt \
      --candidate-xlsx "착수일 후보 블록.xlsx" \
      --samples 256 \
      --heuristic-algorithms ppb_lcp6 \
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

from Phase1.pair_self_labeling import (
    PHASE1_PAIR_ENV_FEATURE_NAMES,
    PHASE1_PAIR_FEATURE_NAMES,
    Phase1PairCandidate,
    _score_bay_loads,
    run_phase1_pair_policy_rollout,
)
from Phase1.self_labeling import PHASE1_SELF_LABEL_HEURISTIC_BANK
from Phase1.self_labeling import run_phase1_heuristic_candidate
from Phase1.pointer_policy import Phase1PairPointerPolicy
from Utils.phase1.phase1_episode_dataset import (
    build_phase1_actual_workday_jobs,
    build_phase1_candidate_workbook_jobs,
)


PPB_LCP6_HEURISTICS = [
    "steel_first_balanced",
    "cut_first_balanced",
    "bevel_first_balanced",
    "steel_first_long_cut_preferred",
    "cut_first_long_cut_preferred",
    "bevel_first_long_cut_preferred",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 pair policy with proposed sampling K.")
    parser.add_argument("--checkpoint", default="", help="Phase1PairPointerPolicy checkpoint path")
    parser.add_argument("--wo-xlsx", default="", help="Actual W/O Excel/CSV path")
    parser.add_argument("--candidate-xlsx", default="착수일 후보 블록.xlsx", help="Candidate block workbook path with {workday}_BLK sheets")
    parser.add_argument("--bay-ids", default="22,23,24", help="Comma-separated Bay ids")
    parser.add_argument("--workdays", default="20260331,20260407,20260408,20260413,20260414,20260415,20260424,20260429")
    parser.add_argument("--gyel", default="NP")
    parser.add_argument("--samples", type=int, default=64, help="Number of stochastic policy rollouts for Proposed")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--score-mode", default="steel_first", choices=["steel_first"])
    parser.add_argument("--heuristic-algorithms", default="ppb_lcp6", help="Comma-separated heuristic names, ppb_lcp6, or all8")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--no-proposed-long-cut-hard-mask",
        action="store_true",
        help="Allow Proposed sampling to assign CUT_LTH>1000 blocks to Bay 24 during evaluation.",
    )
    parser.add_argument(
        "--skip-proposed",
        action="store_true",
        help="Evaluate only Actual and heuristic algorithms without loading a checkpoint or running policy sampling.",
    )
    parser.add_argument(
        "--require-complete-block-workday",
        action="store_true",
        help="Use only blocks whose every W/O starts and ends inside the same 08:00 workday.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    bay_ids = [item.strip() for item in args.bay_ids.split(",") if item.strip()]
    workdays = [item.strip() for item in args.workdays.split(",") if item.strip()]
    heuristic_algorithms = _resolve_heuristics(args.heuristic_algorithms)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = "heuristics_only" if args.skip_proposed else f"sampling{args.samples}"

    model = None
    if args.skip_proposed:
        print("[CHECK][evaluate_phase1_pair_sampling] mode=heuristics_only proposed_sampling=false")
    else:
        if not args.checkpoint:
            print("[ERROR][evaluate_phase1_pair_sampling] cause=missing_checkpoint_when_proposed_enabled")
            raise RuntimeError("--checkpoint is required unless --skip-proposed is set")
        model = _load_model(Path(args.checkpoint))
    if bool(args.wo_xlsx) == bool(args.candidate_xlsx):
        print("[ERROR][evaluate_phase1_pair_sampling] cause=choose_exactly_one_input --wo-xlsx_or_--candidate-xlsx")
        raise RuntimeError("choose exactly one of --wo-xlsx or --candidate-xlsx")
    if args.candidate_xlsx:
        problems = build_phase1_candidate_workbook_jobs(
            candidate_path=Path(args.candidate_xlsx),
            workdays=workdays,
            bay_ids=bay_ids,
        )
    else:
        problems = build_phase1_actual_workday_jobs(
            source_path=args.wo_xlsx,
            workdays=workdays,
            bay_ids=bay_ids,
            gyel=args.gyel,
            require_complete_block_workday=args.require_complete_block_workday,
        )

    summary_rows = []
    candidate_rows = []
    sample_rows = []
    input_block_rows = []
    bay_load_rows = []
    selection_rows = []
    bay_block_rows = []
    proposed_scores = []
    proposed_best_count = 0
    for problem_index, payload in enumerate(problems, start=1):
        jobs = payload["jobs"]
        metadata = payload["metadata"]
        problem_id = str(metadata.get("problem_id") or metadata.get("workday") or f"ACT{problem_index:05d}")
        block_count = int(metadata.get("block_count") or len(jobs))
        evaluation_input_type = _evaluation_input_type(metadata)
        print(
            "[CHECK][evaluate_phase1_pair_sampling.problem_start] "
            f"problem={problem_index}/{len(problems)} problem_id={problem_id} "
            f"block_count={block_count} input_type={evaluation_input_type} "
            f"samples={0 if args.skip_proposed else args.samples}",
            flush=True,
        )

        actual = _actual_history_candidate(jobs=jobs, bay_ids=bay_ids)
        method_candidates = [actual]
        raw_samples = []
        proposed_source = ""
        proposed_score = ()
        proposed_rank = ""
        proposed_is_best = ""
        if not args.skip_proposed:
            proposed, raw_samples = _best_sampling_candidate(
                jobs=jobs,
                bay_ids=bay_ids,
                model=model,
                samples=args.samples,
                temperature=args.temperature,
                seed=args.seed + problem_index * 100_000,
                score_mode=args.score_mode,
                problem_id=problem_id,
                long_cut_hard_mask=not args.no_proposed_long_cut_hard_mask,
            )
            proposed_source = proposed.source
            method_candidates.append(proposed)
        for algorithm in heuristic_algorithms:
            method_candidates.append(_heuristic_candidate_for_eval(jobs, bay_ids, algorithm))
        ranked = sorted(
            [(_score_bay_loads(candidate.bay_loads, args.score_mode), candidate) for candidate in method_candidates],
            key=lambda item: item[0],
        )
        rank_by_source = {candidate.source: rank for rank, (_, candidate) in enumerate(ranked, start=1)}
        block_details = _block_details(jobs)
        input_block_rows.extend(
            _input_block_row(problem_index=problem_index, metadata=metadata, block=block)
            for block in block_details.values()
        )
        actual_score = _score_bay_loads(actual.bay_loads, args.score_mode)
        best_score, best_candidate = ranked[0]
        if not args.skip_proposed:
            proposed_score = _score_bay_loads(method_candidates[1].bay_loads, args.score_mode)
            proposed_rank = rank_by_source[proposed_source]
            proposed_is_best = int(proposed_rank == 1)
            proposed_best_count += proposed_is_best
            proposed_scores.append(proposed_score)

        summary_rows.append(
            {
                "validation_episode": problem_index,
                "problem_id": problem_id,
                "evaluation_input_type": evaluation_input_type,
                "block_count": block_count,
                "actual_score_json": json.dumps(list(actual_score), ensure_ascii=False),
                "actual_long_cut_bay24_count": _long_cut_bay24_count(actual),
                "actual_rank": rank_by_source["actual_history"],
                "proposed_source": proposed_source,
                "proposed_score_json": json.dumps(list(proposed_score), ensure_ascii=False) if proposed_score else "",
                "proposed_long_cut_bay24_count": _long_cut_bay24_count(method_candidates[1]) if proposed_score else "",
                "best_source": best_candidate.source,
                "best_score_json": json.dumps(list(best_score), ensure_ascii=False),
                "best_long_cut_bay24_count": _long_cut_bay24_count(best_candidate),
                "proposed_rank": proposed_rank,
                "proposed_is_best": proposed_is_best,
                "candidate_count": len(method_candidates),
                "sample_count": 0 if args.skip_proposed else args.samples,
            }
        )
        for candidate in method_candidates:
            score = _score_bay_loads(candidate.bay_loads, args.score_mode)
            candidate_rows.append(
                _score_row(
                    problem_index,
                    problem_id,
                    evaluation_input_type,
                    block_count,
                    candidate.source,
                    score,
                    rank_by_source[candidate.source],
                )
            )
            bay_load_rows.extend(
                _bay_load_rows(
                    problem_index=problem_index,
                    metadata=metadata,
                    candidate=candidate,
                    score=score,
                    rank=rank_by_source[candidate.source],
                    bay_ids=bay_ids,
                )
            )
            detail = _assignment_detail_rows(
                problem_index=problem_index,
                metadata=metadata,
                candidate=candidate,
                score=score,
                rank=rank_by_source[candidate.source],
                block_details=block_details,
                bay_ids=bay_ids,
            )
            selection_rows.extend(detail["selection_rows"])
            bay_block_rows.extend(detail["bay_block_rows"])
        for sample_index, sample in enumerate(raw_samples, start=1):
            score = _score_bay_loads(sample.bay_loads, args.score_mode)
            sample_rows.append(
                _score_row(problem_index, problem_id, evaluation_input_type, block_count, sample.source, score, sample_index)
            )
        print(
            "[CHECK][evaluate_phase1_pair_sampling.problem_done] "
            f"problem={problem_index}/{len(problems)} problem_id={problem_id} "
            f"best_source={best_candidate.source} best_score={tuple(best_score)} "
            f"proposed_rank={proposed_rank} proposed_score={tuple(proposed_score)}",
            flush=True,
        )

    summary_csv = output_dir / f"{output_prefix}_summary.csv"
    candidate_csv = output_dir / f"{output_prefix}_candidates.csv"
    report_json = output_dir / f"{output_prefix}_report.json"
    _write_csv(summary_csv, summary_rows)
    _write_csv(candidate_csv, candidate_rows)
    raw_sample_csv = ""
    if sample_rows:
        raw_sample_path = output_dir / f"{output_prefix}_raw_samples.csv"
        _write_csv(raw_sample_path, sample_rows)
        raw_sample_csv = str(raw_sample_path)
    input_blocks_csv = output_dir / "phase1_input_blocks.csv"
    bay_loads_csv = output_dir / "phase1_ppb_lcp6_bay_loads_by_method.csv"
    selection_csv = output_dir / "phase1_ppb_lcp6_selection_order_by_method.csv"
    bay_block_csv = output_dir / "phase1_ppb_lcp6_bay_block_assignments_by_method.csv"
    _write_csv(input_blocks_csv, input_block_rows)
    _write_csv(bay_loads_csv, bay_load_rows)
    _write_csv(selection_csv, selection_rows)
    _write_csv(bay_block_csv, bay_block_rows)
    plot_paths = _write_metric_plots(output_dir=output_dir, rows=candidate_rows, score_mode=args.score_mode)
    report = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else "",
        "wo_xlsx": str(args.wo_xlsx),
        "candidate_xlsx": str(args.candidate_xlsx),
        "evaluation_input_type": "candidate_workbook" if args.candidate_xlsx else "wo_actual_workday",
        "workdays": workdays,
        "bay_ids": bay_ids,
        "score_mode": args.score_mode,
        "require_complete_block_workday": args.require_complete_block_workday,
        "samples": 0 if args.skip_proposed else args.samples,
        "skip_proposed": args.skip_proposed,
        "proposed_long_cut_hard_mask": not args.no_proposed_long_cut_hard_mask,
        "problem_count": len(problems),
        "proposed_best_rate": None if args.skip_proposed else (proposed_best_count / len(problems) if problems else 0.0),
        "proposed_mean_score": _mean_score(proposed_scores),
        "summary_csv": str(summary_csv),
        "candidate_csv": str(candidate_csv),
        "raw_sample_csv": raw_sample_csv,
        "input_blocks_csv": str(input_blocks_csv),
        "bay_loads_csv": str(bay_loads_csv),
        "selection_order_csv": str(selection_csv),
        "bay_block_assignments_csv": str(bay_block_csv),
        **plot_paths,
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[CHECK][evaluate_phase1_pair_sampling] passed=true")
    print(f"- problem_count: {len(problems)}")
    if args.skip_proposed:
        print("- proposed_best_rate: skipped")
    else:
        print(f"- proposed_best_rate: {report['proposed_best_rate']:.6f}")
    print(f"- output_dir: {output_dir}")


def _block_details(jobs: Mapping[str, object]) -> dict[str, dict]:
    blocks: dict[str, dict] = {}
    for job in jobs.values():
        block_id = str(getattr(job, "block_set_id"))
        extra = getattr(job, "extra", {}) or {}
        block = blocks.setdefault(
            block_id,
            {
                "block_set_id": block_id,
                "project_no": str(extra.get("source_project_no") or block_id.split("::")[0]),
                "block_no": str(extra.get("source_block_no") or block_id.split("::")[-1]),
                "actual_source_bays": set(),
                "job_ids": [],
                "wo_count": 0,
                "steel_quantity_sum": 0,
                "cut_length_sum": 0.0,
                "bevel_quantity_sum": 0,
                "plate_length_max": 0.0,
                "thickness_max": 0.0,
            },
        )
        block["actual_source_bays"].add(str(getattr(job, "source_cut_bay", "") or ""))
        block["job_ids"].append(str(getattr(job, "job_id")))
        block["wo_count"] += 1
        block["steel_quantity_sum"] += int(getattr(job, "steel_quantity"))
        block["cut_length_sum"] += float(getattr(job, "cut_length"))
        block["bevel_quantity_sum"] += int(getattr(job, "bevel_quantity"))
        block["plate_length_max"] = max(float(block["plate_length_max"]), float(getattr(job, "plate_length")))
        block["thickness_max"] = max(float(block["thickness_max"]), float(getattr(job, "thickness")))
    for block in blocks.values():
        block["actual_source_bays"] = "|".join(sorted(bay for bay in block["actual_source_bays"] if bay))
        block["job_ids"] = "|".join(block["job_ids"])
        block["cut_length_sum"] = round(float(block["cut_length_sum"]), 6)
        block["plate_length_max"] = round(float(block["plate_length_max"]), 6)
        block["thickness_max"] = round(float(block["thickness_max"]), 6)
        block["long_cut_over_1000"] = int(float(block["cut_length_sum"]) > 1000)
    return blocks


def _input_block_row(problem_index: int, metadata: Mapping, block: Mapping) -> dict:
    return {
        "validation_episode": problem_index,
        "problem_id": str(metadata.get("problem_id", "")),
        "evaluation_input_type": _evaluation_input_type(metadata),
        "workday": str(metadata.get("workday", "")),
        **dict(block),
    }


def _bay_load_rows(
    problem_index: int,
    metadata: Mapping,
    candidate: Phase1PairCandidate,
    score: tuple,
    rank: int,
    bay_ids: Sequence[str],
) -> list[dict]:
    rows = []
    for bay_id in bay_ids:
        load = candidate.bay_loads[str(bay_id)]
        rows.append(
            {
                "validation_episode": problem_index,
                "problem_id": str(metadata.get("problem_id", "")),
                "evaluation_input_type": _evaluation_input_type(metadata),
                "workday": str(metadata.get("workday", "")),
                "source": candidate.source,
                "rank": rank,
                "bay_id": str(bay_id),
                "bay_block_count": int(load.get("block_count", 0)),
                "bay_wo_count": int(load.get("wo_count", 0)),
                "bay_steel_quantity_sum": int(load.get("steel_quantity_sum", 0)),
                "bay_cut_length_sum": round(float(load.get("cut_length_sum", 0.0)), 6),
                "bay_bevel_quantity_sum": int(load.get("bevel_quantity_sum", 0)),
                "bay_long_cut_bay24_count": int(load.get("long_cut_bay24_count", 0)),
                "steel_gap": score[0],
                "cut_gap": score[1],
                "bevel_gap": score[2],
                "long_cut_bay24_count": _long_cut_bay24_count(candidate),
            }
        )
    return rows


def _assignment_detail_rows(
    problem_index: int,
    metadata: Mapping,
    candidate: Phase1PairCandidate,
    score: tuple,
    rank: int,
    block_details: Mapping[str, Mapping],
    bay_ids: Sequence[str],
) -> dict[str, list[dict]]:
    selection_rows = []
    bay_block_rows = []
    bay_order: dict[tuple[str, str], int] = {}
    for selection_order, (block_id, assigned_bay) in enumerate(candidate.assignments.items(), start=1):
        if block_id not in block_details:
            print(
                "[ERROR][evaluate_phase1_pair_sampling._assignment_detail_rows] "
                f"cause=assignment_block_missing source={candidate.source} block_id={block_id}"
            )
            raise RuntimeError(f"assignment_block_missing: {block_id}")
        block = dict(block_details[block_id])
        common = {
            "validation_episode": problem_index,
            "problem_id": str(metadata.get("problem_id", "")),
            "evaluation_input_type": _evaluation_input_type(metadata),
            "workday": str(metadata.get("workday", "")),
            "source": candidate.source,
            "rank": rank,
            "selection_order": selection_order,
            "selection_order_type": "actual_block_source_order" if candidate.source == "actual_history" else "heuristic_decision_order",
            **block,
            "assigned_bay": str(assigned_bay),
            "steel_gap": score[0],
            "cut_gap": score[1],
            "bevel_gap": score[2],
            "long_cut_bay24_count": _long_cut_bay24_count(candidate),
        }
        selection_rows.append(common)
        for bay_id in _assigned_bays(str(assigned_bay)):
            if bay_id not in {str(item) for item in bay_ids}:
                print(
                    "[ERROR][evaluate_phase1_pair_sampling._assignment_detail_rows] "
                    f"cause=bay_outside_scope source={candidate.source} block_id={block_id} bay_id={bay_id}"
                )
                raise RuntimeError(f"bay_outside_scope: {bay_id}")
            key = (str(candidate.source), bay_id)
            bay_order[key] = bay_order.get(key, 0) + 1
            bay_block_rows.append(
                {
                    **common,
                    "bay_id": bay_id,
                    "bay_selection_order": bay_order[key],
                    "is_actual_split_block": int(str(assigned_bay).startswith("SPLIT:")),
                }
            )
    return {"selection_rows": selection_rows, "bay_block_rows": bay_block_rows}


def _assigned_bays(assigned_bay: str) -> list[str]:
    if assigned_bay.startswith("SPLIT:"):
        return [bay for bay in assigned_bay[len("SPLIT:") :].split("|") if bay]
    return [assigned_bay]


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
    problem_id: str = "",
    long_cut_hard_mask: bool = True,
) -> tuple[Phase1PairCandidate, list[Phase1PairCandidate]]:
    if samples <= 0:
        print(f"[ERROR][evaluate_phase1_pair_sampling._best_sampling_candidate] cause=non_positive_samples samples={samples}")
        raise RuntimeError("samples must be positive")
    raw_samples = []
    for sample_index in range(samples):
        sample_number = sample_index + 1
        if sample_number == 1 or sample_number % 64 == 0 or sample_number == samples:
            print(
                "[CHECK][evaluate_phase1_pair_sampling.sample_progress] "
                f"problem_id={problem_id} sample={sample_number}/{samples}",
                flush=True,
            )
        raw_samples.append(
            run_phase1_pair_policy_rollout(
                jobs=jobs,
                bay_ids=bay_ids,
                model=model,
                temperature=temperature,
                seed=seed + sample_index,
                source=f"proposed_sample_{sample_number}",
                selection="sample",
                long_cut_hard_mask=long_cut_hard_mask,
            )
        )
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
    if text.strip().lower() in {"business6", "all6", "ppb_lcp6"}:
        return list(PPB_LCP6_HEURISTICS)
    return [item.strip() for item in text.split(",") if item.strip()]


def _score_row(
    validation_episode: int,
    problem_id: str,
    evaluation_input_type: str,
    block_count: int,
    source: str,
    score: tuple,
    rank: int,
) -> dict:
    row = {
        "validation_episode": validation_episode,
        "problem_id": problem_id,
        "evaluation_input_type": evaluation_input_type,
        "block_count": block_count,
        "source": source,
        "rank": rank,
        "score_json": json.dumps(list(score), ensure_ascii=False),
    }
    for index, value in enumerate(score):
        row[f"score_{index}"] = value
    if len(score) >= 3:
        row["steel_gap"] = score[0]
        row["cut_gap"] = score[1]
        row["bevel_gap"] = score[2]
    return row


def _evaluation_input_type(metadata: Mapping) -> str:
    input_type = str(metadata.get("evaluation_input_type") or "")
    if not input_type:
        print(
            "[ERROR][evaluate_phase1_pair_sampling._evaluation_input_type] "
            f"cause=missing_evaluation_input_type metadata_keys={sorted(metadata.keys())}"
        )
        raise RuntimeError("evaluation_input_type is required in Phase 1 evaluation metadata")
    return input_type


def _long_cut_bay24_count(candidate: Phase1PairCandidate) -> int:
    """Return audit count of long-cut blocks assigned to Bay 24."""

    return int(
        sum(
            int(loads.get("long_cut_bay24_count", 0))
            for loads in candidate.bay_loads.values()
        )
    )


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
    column_count = 2 if len(specs) > 3 else len(specs)
    row_count = (len(specs) + column_count - 1) // column_count
    fig, axes = plt.subplots(row_count, column_count, figsize=(9 * column_count, 4.8 * row_count), sharex=True)
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    else:
        axes = list(axes)
    if axes and hasattr(axes[0], "flat"):
        axes = list(axes[0].flat)
    elif axes and isinstance(axes[0], (list, tuple)):
        axes = [axis for row in axes for axis in row]
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
    for axis in axes[len(specs):]:
        axis.axis("off")
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
        "steel_first_balanced": "MSF+PPB",
        "cut_first_balanced": "LCF+PPB",
        "bevel_first_balanced": "MBF+PPB",
        "long_cut_first_balanced": "Long+PPB",
        "steel_first_long_cut_preferred": "MSF+LCP",
        "cut_first_long_cut_preferred": "LCF+LCP",
        "bevel_first_long_cut_preferred": "MBF+LCP",
        "long_cut_first_long_cut_preferred": "Long+LCP",
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
