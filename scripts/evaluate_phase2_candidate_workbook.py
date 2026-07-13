"""Evaluate merged Phase 2 on `착수일 후보 블록.xlsx` actual-8days problems.

Example:
    python scripts/evaluate_phase2_candidate_workbook.py \
      --config config_np_100.yaml \
      --wo-xlsx input/절단03~04_NP물량_마스킹_WO_수정_260618.xlsx \
      --candidate-xlsx "착수일 후보 블록.xlsx" \
      --phase1-heuristic bevel_first_balanced \
      --heuristic-algorithms min_makespan,best_fit_lth,balanced_tact_load,spt_batch,lpt_batch \
      --output-dir output/phase2_candidate_workbook_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Phase1.self_labeling import run_phase1_heuristic_candidate
from Phase2.merged import (
    PHASE2_BATCH_MACHINE_DEFAULT_HEURISTIC_BANK,
    PHASE2_BATCH_MACHINE_NORMALIZED_SCORE_FIELD_NAMES,
    PHASE2_BATCH_MACHINE_SCORE_FIELD_NAMES,
    build_phase2_batch_machine_candidate_bank,
)
from Phase2.run_spec import build_phase2_run_spec, require_matching_phase2_run_spec
from Environment.constraints.profiles import (
    PhaseConstraintProfile,
    load_phase_constraint_profile,
)
from Utils.config import load_config
from Utils.data.phase2_candidate_workbook import load_phase2_candidate_workbook_problems
from Utils.learning.phase_agent_checkpoints import (
    load_phase2_checkpoint_run_spec,
    load_phase2_set_pointer_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate merged Phase 2 on candidate-workbook actual 8days.")
    parser.add_argument("--config", default="config_np_100.yaml", help="Factory/config YAML path")
    parser.add_argument("--wo-xlsx", default="input/절단03~04_NP물량_마스킹_WO_수정_260618.xlsx")
    parser.add_argument("--candidate-xlsx", default="착수일 후보 블록.xlsx")
    parser.add_argument("--workdays", default="20260331,20260407,20260408,20260413,20260414,20260415,20260424,20260429")
    parser.add_argument("--bay-ids", default="22,23,24")
    parser.add_argument("--gyel", default="NP")
    parser.add_argument("--phase1-heuristic", default="bevel_first_balanced")
    parser.add_argument("--heuristic-algorithms", default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", default="", help="Optional merged Phase2 checkpoint")
    parser.add_argument("--samples", type=int, default=argparse.SUPPRESS, help="Must match checkpoint validation samples")
    parser.add_argument("--max-wo-count", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-length-sum", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--action-pool-limit",
        type=_parse_action_pool_limit,
        default=argparse.SUPPRESS,
        help="Positive integer or None; must match checkpoint when provided",
    )
    parser.add_argument("--score-mode", choices=("raw", "normalized"), default=argparse.SUPPRESS)
    parser.add_argument("--no-phase1-long-cut-hard-mask", action="store_true")
    parser.add_argument("--output-dir", default="output/phase2_candidate_workbook_eval")
    args = parser.parse_args()

    config = load_config(args.config)
    phase2_constraint_profile = load_phase_constraint_profile(config, "phase2")
    bay_ids = _split_csv(args.bay_ids)
    workdays = _split_csv(args.workdays)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problems = load_phase2_candidate_workbook_problems(
        wo_path=args.wo_xlsx,
        candidate_path=args.candidate_xlsx,
        workdays=workdays,
        bay_ids=bay_ids,
        gyel=args.gyel,
        factory_config=config.get("factory"),
    )
    phase1_bay_capacity_weights = _phase1_capacity_weights_from_problems(problems, bay_ids)
    checkpoint_spec = None
    model = None
    if args.checkpoint:
        checkpoint_spec = load_phase2_checkpoint_run_spec(
            args.checkpoint,
            context="phase2_candidate_workbook_eval",
        )
        model = load_phase2_set_pointer_checkpoint(
            args.checkpoint,
            context="phase2_candidate_workbook_eval",
        )
    run_spec = _resolve_evaluation_run_spec(
        args=args,
        checkpoint_spec=checkpoint_spec,
        constraint_profile=phase2_constraint_profile,
        phase1_bay_capacity_weights=phase1_bay_capacity_weights,
    )
    heuristic_algorithms = tuple(run_spec["heuristic_algorithms"])
    samples = int(run_spec["validation_rollout_samples"])
    max_wo_count = int(run_spec["max_wo_count"])
    max_length_sum = float(run_spec["max_length_sum"])
    action_pool_limit = run_spec["action_pool_limit"]
    score_mode = str(run_spec["score_mode"])
    score_fields = list(run_spec["score_fields"])
    long_cut_hard_mask = bool(run_spec["phase1_long_cut_hard_mask"])
    if samples and model is None:
        print("[ERROR][evaluate_phase2_candidate_workbook.main] cause=samples_without_checkpoint")
        raise RuntimeError("--samples requires --checkpoint")

    print("[phase2-candidate-workbook-eval]")
    print(f"- config: {args.config}")
    print(f"- wo_xlsx: {args.wo_xlsx}")
    print(f"- candidate_xlsx: {args.candidate_xlsx}")
    print(f"- workdays: {','.join(workdays)}")
    print(f"- phase1_heuristic: {args.phase1_heuristic}")
    print(f"- phase1_bay_capacity_weights: {phase1_bay_capacity_weights}")
    print(f"- phase1_long_cut_hard_mask: {long_cut_hard_mask}")
    print(f"- phase2_heuristics: {','.join(heuristic_algorithms)}")
    print(f"- checkpoint: {args.checkpoint or 'none'} samples={samples}")
    print(f"- score_mode: {score_mode}")
    print(f"- action_pool_limit: {action_pool_limit}")

    summary_rows: list[dict] = []
    phase1_rows: list[dict] = []
    assignment_rows: list[dict] = []
    batch_rows: list[dict] = []
    timeline_rows: list[dict] = []
    actual_load_rows: list[dict] = []
    actual_load_excluded_rows: list[dict] = []
    constraint_audit_rows: list[dict] = []

    for problem_index, problem in enumerate(problems, start=1):
        workday = str(problem["workday"])
        phase1_candidate = run_phase1_heuristic_candidate(
            jobs=problem["phase1_jobs"],
            bay_ids=bay_ids,
            algorithm=args.phase1_heuristic,
            long_cut_hard_mask=long_cut_hard_mask,
            bay_capacity_weights=phase1_bay_capacity_weights,
        )
        phase1_rows.extend(_phase1_assignment_rows(workday, args.phase1_heuristic, phase1_candidate.assignments))
        jobs = _jobs_by_id(problem["scenario"])
        machines = _machines_by_id(problem["scenario"])
        load_rows, excluded_rows = _actual_source_machine_load_rows(workday, jobs, machines)
        actual_load_rows.extend(load_rows)
        actual_load_excluded_rows.extend(excluded_rows)
        candidates = _candidate_bank(
            jobs=jobs,
            machines=machines,
            phase1_assignments=phase1_candidate.assignments,
            heuristic_algorithms=heuristic_algorithms,
            model=model,
            samples=samples,
            max_wo_count=max_wo_count,
            max_length_sum=max_length_sum,
            action_pool_limit=action_pool_limit,
            score_mode=score_mode,
            constraint_profile=phase2_constraint_profile,
            seed=10_000 * problem_index,
        )
        ranked = sorted(candidates, key=lambda candidate: (*candidate.score_tuple, candidate.source))
        for rank, candidate in enumerate(ranked, start=1):
            summary_rows.append(_summary_row(problem, candidate, rank, score_fields))
            assignment_rows.extend(_assignment_rows(workday, candidate, jobs))
            batch_rows.extend(_prefixed_rows(workday, candidate.source, candidate.batches))
            timeline_rows.extend(_prefixed_rows(workday, candidate.source, candidate.timeline))
            constraint_audit_rows.extend(
                _prefixed_rows(workday, candidate.source, candidate.constraint_audit_rows)
            )
        print(
            "[CHECK][evaluate_phase2_candidate_workbook.main] "
            f"workday={workday} blocks={problem['candidate_block_count']} wo={problem['wo_count']} "
            f"candidate_count={len(candidates)} best={ranked[0].source} score={ranked[0].score_tuple}"
        )

    paths = {
        "summary_csv": output_dir / "phase2_candidate_workbook_summary.csv",
        "phase1_assignments_csv": output_dir / "phase1_assignments.csv",
        "assignment_detail_csv": output_dir / "phase2_assignment_detail.csv",
        "batch_detail_csv": output_dir / "phase2_batch_detail.csv",
        "timeline_detail_csv": output_dir / "phase2_timeline_detail.csv",
        "actual_source_machine_load_csv": output_dir / "actual_source_machine_load_summary.csv",
        "actual_source_machine_load_excluded_csv": output_dir / "actual_source_machine_load_excluded.csv",
        "constraint_audit_csv": output_dir / "phase2_constraint_audit.csv",
        "manifest_json": output_dir / "manifest.json",
    }
    _write_rows(paths["summary_csv"], summary_rows)
    _write_rows(paths["phase1_assignments_csv"], phase1_rows)
    _write_rows(paths["assignment_detail_csv"], assignment_rows)
    _write_rows(paths["batch_detail_csv"], batch_rows)
    _write_rows(paths["timeline_detail_csv"], timeline_rows)
    _write_rows(paths["actual_source_machine_load_csv"], actual_load_rows)
    _write_rows(paths["actual_source_machine_load_excluded_csv"], actual_load_excluded_rows, allow_empty=True)
    _write_rows(paths["constraint_audit_csv"], constraint_audit_rows)
    manifest = {
        "config": args.config,
        "wo_xlsx": args.wo_xlsx,
        "candidate_xlsx": args.candidate_xlsx,
        "workdays": workdays,
        "bay_ids": bay_ids,
        "gyel": args.gyel,
        "phase1_heuristic": args.phase1_heuristic,
        "phase1_bay_capacity_weights": phase1_bay_capacity_weights,
        "phase2_heuristic_algorithms": heuristic_algorithms,
        "checkpoint": args.checkpoint,
        "samples": samples,
        "action_pool_limit": action_pool_limit,
        "score_fields": score_fields,
        "effective_run_spec": run_spec,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest_json"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[VALIDATION][evaluate_phase2_candidate_workbook.main] "
        f"passed=true problems={len(problems)} summary={paths['summary_csv']}"
    )


def _resolve_evaluation_run_spec(
    *,
    args: argparse.Namespace,
    checkpoint_spec: Mapping[str, Any] | None,
    constraint_profile: PhaseConstraintProfile,
    phase1_bay_capacity_weights: Mapping[str, int | float],
) -> dict[str, Any]:
    """CLI와 checkpoint를 하나의 actual 평가 RunSpec으로 확정한다."""

    checkpoint = dict(checkpoint_spec) if checkpoint_spec is not None else None
    heuristic_value = getattr(args, "heuristic_algorithms", None)
    if heuristic_value is None:
        heuristic_algorithms = (
            tuple(checkpoint["heuristic_algorithms"])
            if checkpoint is not None
            else PHASE2_BATCH_MACHINE_DEFAULT_HEURISTIC_BANK
        )
    else:
        heuristic_algorithms = tuple(_split_csv(heuristic_value))
    samples = int(
        getattr(
            args,
            "samples",
            checkpoint["validation_rollout_samples"] if checkpoint is not None else 0,
        )
    )
    max_wo_count = int(
        getattr(
            args,
            "max_wo_count",
            checkpoint["max_wo_count"] if checkpoint is not None else 3,
        )
    )
    max_length_sum = float(
        getattr(
            args,
            "max_length_sum",
            checkpoint["max_length_sum"] if checkpoint is not None else 55_000.0,
        )
    )
    action_pool_limit = getattr(
        args,
        "action_pool_limit",
        checkpoint["action_pool_limit"] if checkpoint is not None else None,
    )
    score_mode = str(
        getattr(
            args,
            "score_mode",
            checkpoint["score_mode"] if checkpoint is not None else "raw",
        )
    )
    checkpoint_long_cut_mask = (
        bool(checkpoint["phase1_long_cut_hard_mask"])
        if checkpoint is not None
        else True
    )
    long_cut_hard_mask = (
        False
        if bool(getattr(args, "no_phase1_long_cut_hard_mask", False))
        else checkpoint_long_cut_mask
    )
    score_fields = (
        PHASE2_BATCH_MACHINE_SCORE_FIELD_NAMES
        if score_mode == "raw"
        else PHASE2_BATCH_MACHINE_NORMALIZED_SCORE_FIELD_NAMES
    )
    requested = build_phase2_run_spec(
        score_mode=score_mode,
        score_fields=score_fields,
        action_pool_limit=action_pool_limit,
        max_wo_count=max_wo_count,
        max_length_sum=max_length_sum,
        phase1_bay_capacity_weights=phase1_bay_capacity_weights,
        phase1_long_cut_hard_mask=long_cut_hard_mask,
        constraint_profile=constraint_profile,
        heuristic_algorithms=heuristic_algorithms,
        train_rollout_samples=(
            int(checkpoint["train_rollout_samples"])
            if checkpoint is not None
            else 0
        ),
        validation_rollout_samples=samples,
    )
    if checkpoint is not None:
        require_matching_phase2_run_spec(
            checkpoint,
            requested,
            context="actual_8days",
        )
    return requested


def _phase1_capacity_weights_from_problems(
    problems: Sequence[Mapping[str, Any]],
    bay_ids: Sequence[str],
) -> dict[str, float]:
    """actual 문제의 enabled machine 수가 모든 날짜에서 같은지 검증한다."""

    if not problems:
        print("[ERROR][evaluate_phase2_candidate_workbook._phase1_capacity_weights_from_problems] cause=no_problems")
        raise RuntimeError("actual evaluation requires at least one problem")
    requested_bays = tuple(str(bay_id) for bay_id in bay_ids)
    expected: dict[str, float] | None = None
    for problem in problems:
        workday = str(problem.get("workday") or "")
        machines = _machines_by_id(problem["scenario"])
        counts = {bay_id: 0.0 for bay_id in requested_bays}
        for machine_id, machine in machines.items():
            enabled = _job_field(machine, "enabled")
            if not isinstance(enabled, bool):
                print(
                    "[ERROR][evaluate_phase2_candidate_workbook._phase1_capacity_weights_from_problems] "
                    f"cause=non_boolean_machine_enabled workday={workday} machine_id={machine_id} value={enabled}"
                )
                raise RuntimeError("actual evaluation machine enabled flag must be boolean")
            bay_id = str(_job_field(machine, "bay_id"))
            if enabled and bay_id in counts:
                counts[bay_id] += 1.0
        missing = sorted(bay_id for bay_id, count in counts.items() if count <= 0)
        if missing:
            print(
                "[ERROR][evaluate_phase2_candidate_workbook._phase1_capacity_weights_from_problems] "
                f"cause=bay_without_enabled_machine workday={workday} bay_ids={missing}"
            )
            raise RuntimeError(f"actual evaluation Bays have no enabled machines: {missing}")
        if expected is None:
            expected = counts
        elif counts != expected:
            print(
                "[ERROR][evaluate_phase2_candidate_workbook._phase1_capacity_weights_from_problems] "
                f"cause=machine_capacity_changed workday={workday} expected={expected} actual={counts}"
            )
            raise RuntimeError("actual evaluation machine capacity differs between workdays")
    assert expected is not None
    return expected


def _parse_action_pool_limit(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"none", "all", "full"}:
        return None
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("action pool limit must be a positive integer or None") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("action pool limit must be positive")
    return parsed


def _candidate_bank(
    jobs: Mapping[str, object],
    machines: Mapping[str, object],
    phase1_assignments: Mapping[str, str],
    heuristic_algorithms: Sequence[str],
    model: object | None,
    samples: int,
    max_wo_count: int,
    max_length_sum: float,
    action_pool_limit: int | None,
    score_mode: str,
    constraint_profile: PhaseConstraintProfile,
    seed: int,
) -> list:
    return build_phase2_batch_machine_candidate_bank(
        jobs=jobs,
        machines=machines,
        phase1_assignments=phase1_assignments,
        model=model,
        heuristic_algorithms=heuristic_algorithms,
        rollout_samples=max(0, int(samples)),
        max_wo_count=max_wo_count,
        max_length_sum=max_length_sum,
        action_pool_limit=action_pool_limit,
        score_mode=score_mode,
        constraint_profile=constraint_profile,
        seed=seed,
    )


def _summary_row(
    problem: Mapping[str, Any],
    candidate: object,
    rank: int,
    score_fields: Sequence[str],
) -> dict:
    row = {
        "workday": problem["workday"],
        "problem_id": problem["problem_id"],
        "method": candidate.source,
        "rank": rank,
        "candidate_block_count": problem["candidate_block_count"],
        "wo_count": problem["wo_count"],
        "batch_count": len(candidate.batches),
        "transition_count": len(candidate.transitions),
        "score_json": json.dumps(list(candidate.score_tuple), ensure_ascii=False),
    }
    for index, field in enumerate(score_fields):
        row[field] = candidate.score_tuple[index]
    return row


def _phase1_assignment_rows(workday: str, phase1_heuristic: str, assignments: Mapping[str, str]) -> list[dict]:
    return [
        {
            "workday": workday,
            "phase1_heuristic": phase1_heuristic,
            "block_set_id": block_set_id,
            "assigned_bay": assigned_bay,
        }
        for block_set_id, assigned_bay in sorted(assignments.items())
    ]


def _assignment_rows(workday: str, candidate: object, jobs: Mapping[str, object]) -> list[dict]:
    rows = []
    for job_id, machine_id in sorted(candidate.machine_assignments.items()):
        job = jobs[job_id]
        rows.append(
            {
                "workday": workday,
                "method": candidate.source,
                "job_id": job_id,
                "work_order_no": _job_field(job, "source_wk_ord_no"),
                "block_set_id": _job_field(job, "block_set_id"),
                "machine_id": machine_id,
                "bay_id": candidate.machine_bay_ids[str(machine_id)],
            }
        )
    return rows


def _actual_source_machine_load_rows(workday: str, jobs: Mapping[str, object], machines: Mapping[str, object]) -> tuple[list[dict], list[dict]]:
    machine_bay = {str(machine_id): str(_job_field(machine, "bay_id")) for machine_id, machine in machines.items()}
    loads: Dict[str, Dict[str, Any]] = {}
    excluded: list[dict] = []
    for job in jobs.values():
        machine_id = str(_job_optional_field(job, "source_machine_id")).strip()
        if not machine_id:
            excluded.append(_actual_load_excluded_row(workday, job, "missing_source_machine_id"))
            continue
        if machine_id not in machine_bay:
            excluded.append(_actual_load_excluded_row(workday, job, f"source_machine_not_in_factory:{machine_id}"))
            continue
        load = loads.setdefault(
            machine_id,
            {
                "workday": workday,
                "machine_id": machine_id,
                "bay_id": machine_bay[machine_id],
                "wo_count": 0,
                "cut_length_sum": 0.0,
                "bevel_quantity_sum": 0,
                "tact_time_sum": 0.0,
            },
        )
        load["wo_count"] += 1
        load["cut_length_sum"] += float(_job_field(job, "cut_length"))
        load["bevel_quantity_sum"] += int(_job_field(job, "bevel_quantity"))
        base_stage = _job_field(job, "base_stage_minutes")
        load["tact_time_sum"] += float(sum(base_stage.values()) if isinstance(base_stage, Mapping) else 0.0)
    if excluded:
        print(
            "[CHECK][evaluate_phase2_candidate_workbook._actual_source_machine_load_rows] "
            f"workday={workday} excluded_actual_source_machine_rows={len(excluded)}"
        )
    return [dict(row) for row in sorted(loads.values(), key=lambda row: (str(row["bay_id"]), str(row["machine_id"])))], excluded


def _actual_load_excluded_row(workday: str, job: object, reason: str) -> dict:
    return {
        "workday": workday,
        "reason": reason,
        "job_id": _job_optional_field(job, "job_id"),
        "work_order_no": _job_optional_field(job, "source_wk_ord_no"),
        "block_set_id": _job_optional_field(job, "block_set_id"),
        "source_machine_id": _job_optional_field(job, "source_machine_id"),
    }


def _prefixed_rows(workday: str, method: str, rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    return [{"workday": workday, "method": method, **dict(row)} for row in rows]


def _jobs_by_id(scenario: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(job["job_id"]): job for job in scenario["jobs"]}


def _machines_by_id(scenario: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(machine["machine_id"]): machine for machine in scenario["machines"]}


def _job_field(row: object, key: str) -> Any:
    if isinstance(row, Mapping):
        value = row.get(key)
    else:
        value = getattr(row, key, None)
    if value in (None, ""):
        print(f"[ERROR][evaluate_phase2_candidate_workbook._job_field] cause=missing_field key={key}")
        raise RuntimeError(f"missing_field: {key}")
    return value


def _job_optional_field(row: object, key: str) -> Any:
    if isinstance(row, Mapping):
        value = row.get(key)
    else:
        value = getattr(row, key, None)
    return "" if value is None else value


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]], allow_empty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if allow_empty:
            path.write_text("", encoding="utf-8-sig")
            return
        print(f"[ERROR][evaluate_phase2_candidate_workbook._write_rows] cause=no_rows path={path}")
        raise RuntimeError(f"no_rows: {path}")
    fieldnames = list(dict.fromkeys(field for row in rows for field in row.keys()))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _split_csv(value: str) -> list[str]:
    result = [item.strip() for item in str(value).split(",") if item.strip()]
    if not result:
        print(f"[ERROR][evaluate_phase2_candidate_workbook._split_csv] cause=empty_csv value={value}")
        raise RuntimeError("empty comma-separated value")
    return result


if __name__ == "__main__":
    main()
