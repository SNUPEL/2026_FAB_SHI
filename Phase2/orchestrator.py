"""Phase 1 plan을 merged Phase 2 batch-machine 실행·리포트에 연결한다.

Phase 2는 Phase 1의 Block->Bay 결과를 받은 뒤 Bay 내부에서 환경이 다음
가용 설비를 확정하고, policy는 해당 설비의 batch에 넣을 W/O만 선택한다.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from Environment.constraints.profiles import PhaseConstraintProfile
from Phase2.evaluation import evaluate_phase2_schedule
from Phase2.set_pointer_policy import Phase2SetPointerPolicy
from Phase2.merged import build_phase2_batch_machine_candidate_bank, run_phase2_batch_machine_candidate
from Utils.learning.phase1_phase2_communication import (
    apply_phase1_messages_to_scenario,
    build_phase1_plan_messages,
    build_phase2_feedback_messages,
)
from Utils.learning.phase_graph_mdp import build_phase2_wo_machine_graph


def run_phase2_graph_workflow(
    scenario: Mapping[str, Any],
    phase1_plan: Mapping[str, Any],
    assignment_mode: str = "allowed_bay_ids",
) -> Dict[str, Any]:
    """Phase 1 plan을 적용하고 Phase 2 W/O-Machine graph state를 만든다."""

    phase1_messages = build_phase1_plan_messages(scenario=scenario, plan=phase1_plan)
    phase2_feedback = build_phase2_feedback_messages(scenario=scenario, plan=phase1_plan)
    applied = apply_phase1_messages_to_scenario(
        scenario=scenario,
        phase1_messages=phase1_messages,
        assignment_mode=assignment_mode,
    )
    phase2_scenario = applied["scenario"]
    jobs = _jobs_by_id(phase2_scenario)
    machines = _machines_by_id(phase2_scenario)
    phase1_assignments = {
        str(message["block_set_id"]): str(message["assigned_bay"])
        for message in phase1_messages
    }
    graph = build_phase2_wo_machine_graph(
        jobs=jobs,
        machines=machines,
        phase1_assignments=phase1_assignments,
    )
    return {
        "phase": "phase2",
        "phase1_messages": phase1_messages,
        "phase2_feedback": phase2_feedback,
        "phase1_message_count": len(phase1_messages),
        "phase2_feedback_count": len(phase2_feedback),
        "applied_scenario": phase2_scenario,
        "applied_summary": applied["summary"],
        "graph": graph,
    }


def run_phase2_full_graph_workflow(
    scenario: Mapping[str, Any],
    phase1_plan: Mapping[str, Any],
    assignment_mode: str = "allowed_bay_ids",
    model: Phase2SetPointerPolicy | None = None,
    max_wo_count: int = 3,
    max_length_sum: float = 55000.0,
    batch_machine_heuristic: str | None = "min_makespan",
    score_mode: str = "raw",
    action_pool_limit: int | None = None,
    constraint_profile: PhaseConstraintProfile | None = None,
    effective_run_spec: Mapping[str, Any] | None = None,
    rollout_samples: int = 0,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run merged Phase 2 batch-machine scheduling and exportable timeline."""

    if batch_machine_heuristic and model is not None:
        print(
            "[ERROR][Phase2.orchestrator.run_phase2_full_graph_workflow] "
            "cause=batch_machine_heuristic_and_model_both_set"
        )
        raise RuntimeError("Choose either merged Phase 2 heuristic or checkpoint model, not both")

    base = run_phase2_graph_workflow(
        scenario=scenario,
        phase1_plan=phase1_plan,
        assignment_mode=assignment_mode,
    )
    jobs = _jobs_by_id(base["applied_scenario"])
    machines = _machines_by_id(base["applied_scenario"])
    phase1_assignments = {
        str(message["block_set_id"]): str(message["assigned_bay"])
        for message in base["phase1_messages"]
    }
    if model is None:
        batch_candidates = [
            run_phase2_batch_machine_candidate(
                jobs=jobs,
                machines=machines,
                phase1_assignments=phase1_assignments,
                source=str(batch_machine_heuristic),
                model=None,
                max_wo_count=max_wo_count,
                max_length_sum=max_length_sum,
                action_pool_limit=action_pool_limit,
                score_mode=score_mode,
                constraint_profile=constraint_profile,
                seed=seed,
            )
        ]
    else:
        if rollout_samples < 0:
            print(
                "[ERROR][Phase2.orchestrator.run_phase2_full_graph_workflow] "
                f"cause=negative_rollout_samples value={rollout_samples}"
            )
            raise RuntimeError("full-flow rollout_samples must not be negative")
        batch_candidates = build_phase2_batch_machine_candidate_bank(
            jobs=jobs,
            machines=machines,
            phase1_assignments=phase1_assignments,
            model=model,
            heuristic_algorithms=(),
            rollout_samples=rollout_samples,
            max_wo_count=max_wo_count,
            max_length_sum=max_length_sum,
            action_pool_limit=action_pool_limit,
            score_mode=score_mode,
            constraint_profile=constraint_profile,
            seed=seed,
        )
    batch_candidate = min(batch_candidates, key=lambda candidate: (*candidate.score_tuple, candidate.source))
    assignment = _assignment_from_merged_candidate(batch_candidate, machines)
    batches = _batches_from_merged_candidate(batch_candidate, max_wo_count, max_length_sum)
    sequence = _timeline_from_merged_candidate(batch_candidate)
    constraint_audit = {
        "rows": [dict(row) for row in batch_candidate.constraint_audit_rows],
        "hard_violation_count": sum(
            1 for row in batch_candidate.constraint_audit_rows if not bool(row["passed"])
        ),
    }
    evaluation = evaluate_phase2_schedule(
        machine_loads=assignment["machine_loads"],
        machine_bay_ids={machine_id: str(_machine_field(machine, "bay_id")) for machine_id, machine in machines.items()},
        batches=batches["batches"],
        timeline=sequence["timeline"],
        max_wo_count=max_wo_count,
        max_length_sum=max_length_sum,
        score_mode=score_mode,
        hard_violation_count=int(constraint_audit["hard_violation_count"]),
    )
    if tuple(evaluation["score_tuple"]) != tuple(batch_candidate.score_tuple):
        print(
            "[ERROR][Phase2.orchestrator.run_phase2_full_graph_workflow] "
            f"cause=score_contract_mismatch candidate={batch_candidate.score_tuple} "
            f"evaluation={evaluation['score_tuple']}"
        )
        raise RuntimeError("full-flow Phase 2 score differs from the training candidate score")
    return {
        **base,
        "assignment": assignment,
        "batches": batches,
        "sequence": sequence,
        "constraint_audit": constraint_audit,
        "evaluation": evaluation,
        "batch_machine_candidate_score": tuple(batch_candidate.score_tuple),
        "batch_machine_candidate_count": len(batch_candidates),
        "effective_run_spec": dict(effective_run_spec) if effective_run_spec is not None else None,
    }


def _assignment_from_merged_candidate(candidate: Any, machines: Mapping[str, Any]) -> Dict[str, Any]:
    selection_rows = []
    for step, transition in enumerate(candidate.transitions):
        machine_id = str(transition.selected_machine_id)
        if machine_id not in machines:
            print(
                "[ERROR][Phase2.orchestrator._assignment_from_merged_candidate] "
                f"cause=missing_machine machine_id={machine_id}"
            )
            raise RuntimeError(f"missing machine for merged Phase 2 candidate: {machine_id}")
        for job_id in transition.selected_job_ids:
            selection_rows.append(
                {
                    "step": step,
                    "job_id": job_id,
                    "machine_id": machine_id,
                    "bay_id": str(_machine_field(machines[machine_id], "bay_id")),
                    "edge_id": "",
                    "selection_mode": candidate.source,
                    "logit": "",
                }
            )
    return {
        "machine_assignments": dict(candidate.machine_assignments),
        "machine_loads": {key: dict(value) for key, value in candidate.machine_loads.items()},
        "selection_rows": selection_rows,
        "summary": {
            "assigned_job_count": len(candidate.machine_assignments),
            "machine_count": len(machines),
            "selection_count": len(selection_rows),
            "assignment_source": "merged_batch_machine",
            "batch_machine_source": candidate.source,
        },
    }


def _batches_from_merged_candidate(candidate: Any, max_wo_count: int, max_length_sum: float) -> Dict[str, Any]:
    assigned_job_count = sum(int(batch["wo_count"]) for batch in candidate.batches)
    return {
        "batches": list(candidate.batches),
        "summary": {
            "assigned_job_count": assigned_job_count,
            "batch_count": len(candidate.batches),
            "max_wo_count": max_wo_count,
            "max_length_sum": max_length_sum,
            "batch_source": candidate.source,
        },
    }


def _timeline_from_merged_candidate(candidate: Any) -> Dict[str, Any]:
    finish_times = [float(row["finish_time"]) for row in candidate.timeline]
    if not finish_times:
        print("[ERROR][Phase2.orchestrator._timeline_from_merged_candidate] cause=no_timeline")
        raise RuntimeError("merged Phase 2 candidate produced no timeline")
    return {
        "timeline": list(candidate.timeline),
        "event_log": [dict(row) for row in candidate.event_log],
        "summary": {
            "batch_count": len(candidate.timeline),
            "timeline_count": len(candidate.timeline),
            "makespan": round(max(finish_times), 6),
            "sequence_source": candidate.source,
        },
    }


def _machine_field(machine: Any, key: str) -> Any:
    if isinstance(machine, Mapping):
        if key not in machine:
            print(f"[ERROR][Phase2.orchestrator._machine_field] cause=missing_key key={key} machine={machine}")
            raise RuntimeError(f"machine row missing {key}")
        return machine[key]
    if not hasattr(machine, key):
        print(f"[ERROR][Phase2.orchestrator._machine_field] cause=missing_attr key={key} machine={machine}")
        raise RuntimeError(f"machine object missing {key}")
    return getattr(machine, key)


def write_phase2_workflow_outputs(result: Mapping[str, Any], output_dir: str | Path) -> Dict[str, Any]:
    """Write merged Phase 2 workflow outputs for debugging and review."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scenario = result.get("applied_scenario")
    assignment = result.get("assignment")
    batches = result.get("batches")
    sequence = result.get("sequence")
    evaluation = result.get("evaluation")
    constraint_audit = result.get("constraint_audit")
    effective_run_spec = result.get("effective_run_spec")
    candidate_count = result.get("batch_machine_candidate_count")
    candidate_score = result.get("batch_machine_candidate_score")
    if not isinstance(scenario, Mapping):
        print("[ERROR][Phase2.orchestrator.write_phase2_workflow_outputs] cause=missing_applied_scenario")
        raise RuntimeError("missing applied_scenario")
    if not isinstance(assignment, Mapping):
        print("[ERROR][Phase2.orchestrator.write_phase2_workflow_outputs] cause=missing_assignment")
        raise RuntimeError("missing assignment")
    if not isinstance(batches, Mapping):
        print("[ERROR][Phase2.orchestrator.write_phase2_workflow_outputs] cause=missing_batches")
        raise RuntimeError("missing batches")
    if not isinstance(sequence, Mapping):
        print("[ERROR][Phase2.orchestrator.write_phase2_workflow_outputs] cause=missing_sequence")
        raise RuntimeError("missing sequence")
    if not isinstance(evaluation, Mapping):
        print("[ERROR][Phase2.orchestrator.write_phase2_workflow_outputs] cause=missing_evaluation")
        raise RuntimeError("missing evaluation")
    if not isinstance(constraint_audit, Mapping):
        print("[ERROR][Phase2.orchestrator.write_phase2_workflow_outputs] cause=missing_constraint_audit")
        raise RuntimeError("missing common Phase 2 constraint audit")
    if effective_run_spec is not None and not isinstance(effective_run_spec, Mapping):
        print(
            "[ERROR][Phase2.orchestrator.write_phase2_workflow_outputs] "
            f"cause=invalid_effective_run_spec type={type(effective_run_spec).__name__}"
        )
        raise RuntimeError("effective Phase 2 RunSpec must be a mapping")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count <= 0:
        print(
            "[ERROR][Phase2.orchestrator.write_phase2_workflow_outputs] "
            f"cause=invalid_candidate_count value={candidate_count}"
        )
        raise RuntimeError("merged Phase 2 workflow requires a positive candidate count")
    if (
        not isinstance(candidate_score, (list, tuple))
        or not candidate_score
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in candidate_score
        )
    ):
        print(
            "[ERROR][Phase2.orchestrator.write_phase2_workflow_outputs] "
            f"cause=invalid_candidate_score value={candidate_score}"
        )
        raise RuntimeError("merged Phase 2 workflow requires a finite candidate score")

    jobs = _jobs_by_id(scenario)
    machines = _machines_by_id(scenario)
    assignment_rows = _assignment_rows(
        machine_assignments=assignment.get("machine_assignments", {}),
        jobs=jobs,
        selection_rows=assignment.get("selection_rows", []),
    )
    machine_load_rows = _machine_load_rows(assignment.get("machine_loads", {}))
    batch_rows = _batch_rows(batches.get("batches", []))
    timeline_rows = _timeline_rows(sequence.get("timeline", []))
    summary_rows = _evaluation_summary_rows(evaluation)
    bay_metric_rows = _per_bay_metric_rows(evaluation.get("per_bay"))
    constraint_audit_rows = constraint_audit.get("rows")
    if not isinstance(constraint_audit_rows, list) or not constraint_audit_rows:
        print("[ERROR][Phase2.orchestrator.write_phase2_workflow_outputs] cause=missing_constraint_audit_rows")
        raise RuntimeError("common Phase 2 constraint audit rows are required")
    event_rows = sequence.get("event_log")
    if not isinstance(event_rows, list) or not event_rows:
        print(
            "[ERROR][Phase2.orchestrator.write_phase2_workflow_outputs] "
            "cause=missing_transition_event_log"
        )
        raise RuntimeError("merged Phase 2 workflow requires transition-generated event log")
    _validate_transition_event_rows(event_rows, jobs, sequence.get("timeline", []))

    assignment_csv = output_path / "wo_machine_assignment.csv"
    machine_load_csv = output_path / "machine_load_summary.csv"
    batch_csv = output_path / "batch_list.csv"
    timeline_csv = output_path / "machine_timeline.csv"
    makespan_csv = output_path / "makespan_summary.csv"
    bay_metric_csv = output_path / "bay_metric_summary.csv"
    constraint_audit_csv = output_path / "phase2_constraint_audit.csv"
    event_log_json = output_path / "phase2_event_log.json"
    report_json = output_path / "phase2_workflow_report.json"

    _write_csv(assignment_csv, assignment_rows)
    _write_csv(machine_load_csv, machine_load_rows)
    _write_csv(batch_csv, batch_rows)
    _write_csv(timeline_csv, timeline_rows)
    _write_csv(makespan_csv, summary_rows)
    _write_csv(bay_metric_csv, bay_metric_rows)
    _write_csv(constraint_audit_csv, constraint_audit_rows)
    event_log_json.write_text(json.dumps(event_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "assignment_csv": str(assignment_csv),
        "machine_load_csv": str(machine_load_csv),
        "batch_csv": str(batch_csv),
        "timeline_csv": str(timeline_csv),
        "makespan_csv": str(makespan_csv),
        "bay_metric_csv": str(bay_metric_csv),
        "constraint_audit_csv": str(constraint_audit_csv),
        "event_log_json": str(event_log_json),
        "assignment_summary": assignment.get("summary", {}),
        "batch_summary": batches.get("summary", {}),
        "sequence_summary": sequence.get("summary", {}),
        "evaluation": evaluation,
        "constraint_violation_count": int(constraint_audit["hard_violation_count"]),
        "event_count": len(event_rows),
        "batch_machine_candidate_count": candidate_count,
        "batch_machine_candidate_score": list(candidate_score),
        "effective_run_spec": dict(effective_run_spec) if effective_run_spec is not None else None,
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[VALIDATION][Phase2.orchestrator.write_phase2_workflow_outputs] "
        f"passed=true output_dir={output_path} assignments={len(assignment_rows)} "
        f"batches={len(batch_rows)} timeline={len(timeline_rows)} "
        f"constraint_violations={report['constraint_violation_count']} events={len(event_rows)}"
    )
    return {**report, "report_json": str(report_json)}


def _jobs_by_id(scenario: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    jobs = scenario.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        print("[ERROR][Phase2.orchestrator._jobs_by_id] cause=no_jobs")
        raise RuntimeError("Phase 2 workflow requires scenario jobs")
    result = {}
    for index, job in enumerate(jobs):
        if not isinstance(job, Mapping):
            print(
                "[ERROR][Phase2.orchestrator._jobs_by_id] "
                f"cause=invalid_job_row index={index} row_type={type(job).__name__}"
            )
            raise RuntimeError(f"invalid job row at index={index}")
        job_id = str(job.get("job_id") or "").strip()
        if not job_id:
            print(f"[ERROR][Phase2.orchestrator._jobs_by_id] cause=missing_job_id index={index}")
            raise RuntimeError(f"missing job_id at index={index}")
        result[job_id] = job
    return result


def _assignment_rows(
    machine_assignments: Mapping[str, str],
    jobs: Mapping[str, Mapping[str, Any]],
    selection_rows: Sequence[Mapping],
) -> list[dict]:
    if not machine_assignments:
        print("[ERROR][Phase2.orchestrator._assignment_rows] cause=no_machine_assignments")
        raise RuntimeError("machine_assignments are required")
    selection_by_job = {str(row["job_id"]): row for row in selection_rows}
    rows = []
    for job_id, machine_id in sorted(machine_assignments.items()):
        if job_id not in jobs:
            print(f"[ERROR][Phase2.orchestrator._assignment_rows] cause=missing_job job_id={job_id}")
            raise RuntimeError(f"missing job in assignment export: {job_id}")
        job = jobs[job_id]
        selected = selection_by_job.get(job_id, {})
        rows.append(
            {
                "job_id": job_id,
                "block_set_id": str(job.get("block_set_id", "")),
                "machine_id": str(machine_id),
                "bay_id": str(selected.get("bay_id", "")),
                "selection_step": selected.get("step", ""),
                "selection_mode": selected.get("selection_mode", ""),
                "steel_quantity": job.get("steel_quantity", ""),
                "cut_length": job.get("cut_length", ""),
                "bevel_quantity": job.get("bevel_quantity", ""),
                "plate_length": job.get("plate_length", ""),
                "processing_time": _job_processing_time(job),
            }
        )
    return rows


def _machine_load_rows(machine_loads: Mapping[str, Mapping[str, int | float]]) -> list[dict]:
    if not machine_loads:
        print("[ERROR][Phase2.orchestrator._machine_load_rows] cause=no_machine_loads")
        raise RuntimeError("machine_loads are required")
    return [
        {
            "machine_id": str(machine_id),
            "wo_count": loads["wo_count"],
            "processing_time_sum": loads["processing_time_sum"],
            "cut_length_sum": loads["cut_length_sum"],
            "bevel_quantity_sum": loads["bevel_quantity_sum"],
        }
        for machine_id, loads in sorted(machine_loads.items())
    ]


def _batch_rows(batches: Sequence[Mapping]) -> list[dict]:
    if not batches:
        print("[ERROR][Phase2.orchestrator._batch_rows] cause=no_batches")
        raise RuntimeError("batches are required")
    return [
        {
            "batch_id": str(batch["batch_id"]),
            "machine_id": str(batch["machine_id"]),
            "job_ids": "|".join(str(job_id) for job_id in batch["job_ids"]),
            "wo_count": batch["wo_count"],
            "length_sum": batch["length_sum"],
        }
        for batch in batches
    ]


def _timeline_rows(timeline: Sequence[Mapping]) -> list[dict]:
    if not timeline:
        print("[ERROR][Phase2.orchestrator._timeline_rows] cause=no_timeline")
        raise RuntimeError("timeline is required")
    return [
        {
            "batch_id": str(row["batch_id"]),
            "machine_id": str(row["machine_id"]),
            "job_ids": "|".join(str(job_id) for job_id in row["job_ids"]),
            "start_time": row["start_time"],
            "finish_time": row["finish_time"],
            "duration": row["duration"],
            "batch_duration": row["batch_duration"],
            "job_processing_time_sum": row["job_processing_time_sum"],
            "duration_rule": row["duration_rule"],
        }
        for row in timeline
    ]


def _evaluation_summary_rows(evaluation: Mapping[str, Any]) -> list[dict]:
    """중첩된 공통 metric 결과를 손실 없이 CSV long format으로 펼친다."""

    rows: list[dict] = []

    def append_value(path: tuple[str, ...], value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                append_value((*path, str(key)), nested)
            return
        if isinstance(value, (list, tuple)):
            serialized = json.dumps(list(value), ensure_ascii=False)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            serialized = value
        else:
            print(
                "[ERROR][Phase2.orchestrator._evaluation_summary_rows] "
                f"cause=unsupported_metric_value path={'.'.join(path)} type={type(value).__name__}"
            )
            raise RuntimeError(f"unsupported evaluation metric value: {'.'.join(path)}")
        group = path[0] if len(path) > 1 else "evaluation"
        metric = ".".join(path[1:]) if len(path) > 1 else path[0]
        rows.append({"metric_group": group, "metric": metric, "value": serialized})

    for key, value in evaluation.items():
        append_value((str(key),), value)
    if not rows:
        print("[ERROR][Phase2.orchestrator._evaluation_summary_rows] cause=no_evaluation_rows")
        raise RuntimeError("evaluation has no rows")
    return rows


def _per_bay_metric_rows(per_bay: Any) -> list[dict]:
    """공통 metric contract의 Bay별 설비 지표를 wide CSV row로 만든다."""

    if not isinstance(per_bay, Mapping) or not per_bay:
        print("[ERROR][Phase2.orchestrator._per_bay_metric_rows] cause=missing_per_bay_metrics")
        raise RuntimeError("Phase 2 evaluation requires per-Bay metrics")
    rows: list[dict] = []
    for bay_id, metrics in sorted(per_bay.items(), key=lambda item: str(item[0])):
        if not isinstance(metrics, Mapping) or not metrics:
            print(
                "[ERROR][Phase2.orchestrator._per_bay_metric_rows] "
                f"cause=invalid_bay_metrics bay_id={bay_id} type={type(metrics).__name__}"
            )
            raise RuntimeError(f"invalid Phase 2 per-Bay metrics: {bay_id}")
        rows.append({"bay_id": str(bay_id), **dict(metrics)})
    return rows


def _validate_transition_event_rows(
    rows: Sequence[Mapping[str, Any]],
    jobs: Mapping[str, Mapping[str, Any]],
    timeline: Sequence[Mapping],
) -> None:
    if not rows:
        print("[ERROR][Phase2.orchestrator._validate_transition_event_rows] cause=no_events")
        raise RuntimeError("Phase 2 event log has no events")
    expected_jobs = set(jobs)
    event_ids = [str(row.get("event_id") or "") for row in rows]
    if any(not event_id for event_id in event_ids) or len(set(event_ids)) != len(event_ids):
        print(
            "[ERROR][Phase2.orchestrator._validate_transition_event_rows] "
            "cause=missing_or_duplicate_event_id"
        )
        raise RuntimeError("Phase 2 transition event ids must be non-empty and unique")
    started = [str(row["job_id"]) for row in rows if row["event_type"] == "PROCESS_START"]
    ended = [str(row["job_id"]) for row in rows if row["event_type"] == "PROCESS_FINISH"]
    if set(started) != expected_jobs or set(ended) != expected_jobs:
        print(
            "[ERROR][Phase2.orchestrator._validate_transition_event_rows] "
            f"cause=job_identity_mismatch expected={len(expected_jobs)} "
            f"started={len(set(started))} ended={len(set(ended))}"
        )
        raise RuntimeError("Phase 2 event log job identity mismatch")
    duplicate_starts = sorted(job_id for job_id in expected_jobs if started.count(job_id) != 1)
    duplicate_ends = sorted(job_id for job_id in expected_jobs if ended.count(job_id) != 1)
    if duplicate_starts or duplicate_ends:
        print(
            "[ERROR][Phase2.orchestrator._validate_transition_event_rows] "
            f"cause=job_event_count_mismatch start_errors={duplicate_starts[:5]} end_errors={duplicate_ends[:5]}"
        )
        raise RuntimeError("Phase 2 event log has duplicate/missing job start or end events")
    timeline_batch_ids = {str(row["batch_id"]) for row in timeline}
    start_batch_ids = {
        str(row.get("payload", {}).get("batch_id") or "")
        for row in rows
        if row["event_type"] == "PROCESS_START"
    }
    finish_batch_ids = {
        str(row.get("payload", {}).get("batch_id") or "")
        for row in rows
        if row["event_type"] == "PROCESS_FINISH"
    }
    if start_batch_ids != timeline_batch_ids or finish_batch_ids != timeline_batch_ids:
        print(
            "[ERROR][Phase2.orchestrator._validate_transition_event_rows] "
            f"cause=batch_identity_mismatch timeline={sorted(timeline_batch_ids)} "
            f"started={sorted(start_batch_ids)} finished={sorted(finish_batch_ids)}"
        )
        raise RuntimeError("Phase 2 transition event batch identity mismatch")


def _job_processing_time(job: Mapping[str, Any]) -> float:
    """공통 DES와 동일하게 scenario의 stage 합만 처리시간으로 사용한다."""

    job_id = str(job.get("job_id") or "").strip()
    minutes = job.get("base_stage_minutes")
    if not job_id or not isinstance(minutes, Mapping) or not minutes:
        print(
            "[ERROR][Phase2.orchestrator._job_processing_time] "
            f"cause=missing_or_invalid_base_stage_minutes job_id={job_id or '<missing>'}"
        )
        raise RuntimeError(f"invalid base_stage_minutes for job: {job_id or '<missing>'}")
    try:
        total = sum(float(value) for value in minutes.values())
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][Phase2.orchestrator._job_processing_time] "
            f"cause=non_numeric_base_stage_minutes job_id={job_id} values={minutes}"
        )
        raise RuntimeError(f"non-numeric base_stage_minutes for job: {job_id}") from exc
    if not math.isfinite(total) or total <= 0:
        print(
            "[ERROR][Phase2.orchestrator._job_processing_time] "
            f"cause=non_positive_processing_time job_id={job_id} value={total}"
        )
        raise RuntimeError(f"non-positive processing time for job: {job_id}")
    return total


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        print(f"[ERROR][Phase2.orchestrator._write_csv] cause=no_rows path={path}")
        raise RuntimeError(f"no rows to write: {path}")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _machines_by_id(scenario: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    machines = scenario.get("machines")
    if not isinstance(machines, list) or not machines:
        print("[ERROR][Phase2.orchestrator._machines_by_id] cause=no_machines")
        raise RuntimeError("Phase 2 workflow requires scenario machines")
    result = {}
    for index, machine in enumerate(machines):
        if not isinstance(machine, Mapping):
            print(
                "[ERROR][Phase2.orchestrator._machines_by_id] "
                f"cause=invalid_machine_row index={index} row_type={type(machine).__name__}"
            )
            raise RuntimeError(f"invalid machine row at index={index}")
        machine_id = str(machine.get("machine_id") or "").strip()
        if not machine_id:
            print(f"[ERROR][Phase2.orchestrator._machines_by_id] cause=missing_machine_id index={index}")
            raise RuntimeError(f"missing machine_id at index={index}")
        result[machine_id] = machine
    return result
