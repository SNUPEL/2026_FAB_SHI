"""Limited communication contract between Phase 1 and Phase 2 agents.

Phase 1 decides `block -> cutting Bay`.
Phase 2 later decides `W/O batch -> machine` inside the Bay chosen by Phase 1.

This module does not implement the final Phase 2 scheduler. It implements the
auditable message boundary between the two agents so the two phases can be
trained, tested, and debugged without giving each agent the other's full state.
"""

from __future__ import annotations

import csv
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


COMMUNICATION_SCHEMA_VERSION = "phase1_phase2_comm_v1"
DEFAULT_WIDE_BTH_THRESHOLD = 4500.0
DEFAULT_BATCH_MAX_WO_COUNT = 3
DEFAULT_BATCH_MAX_LENGTH_SUM = 55000.0
WIDE_CAPABLE_BAYS = ("22", "23")


def build_phase1_plan_messages(
    scenario: Mapping[str, Any],
    plan: Mapping[str, Any],
    wide_bth_threshold: float = DEFAULT_WIDE_BTH_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Build the messages Phase 1 is allowed to send to Phase 2.

    Each message is one block-level assignment. It includes the block's own
    W/O identifiers and aggregate workload values, but it does not include
    Phase 1 policy logits, all candidate search states, or machine timelines.
    """

    assignment_rows = _assignment_rows_by_block(plan)
    jobs_by_block = _jobs_by_block(scenario)
    messages: List[Dict[str, Any]] = []
    for block_set_id in sorted(assignment_rows):
        assignment = assignment_rows[block_set_id]
        block_jobs = _require_block_jobs(jobs_by_block, block_set_id)
        metrics = _aggregate_block_jobs(
            block_set_id=block_set_id,
            block_jobs=block_jobs,
            wide_bth_threshold=wide_bth_threshold,
        )
        messages.append(
            {
                "schema_version": COMMUNICATION_SCHEMA_VERSION,
                "direction": "phase1_to_phase2",
                "message_type": "block_bay_assignment",
                "message_scope": "assigned_block_only",
                "phase1_algorithm": str(plan.get("algorithm") or ""),
                "block_set_id": block_set_id,
                "project_no": str(assignment.get("project_no") or metrics["project_no"]),
                "block_no": str(assignment.get("block_no") or metrics["block_no"]),
                "assigned_bay": _require_text(assignment.get("assigned_bay"), "assigned_bay", block_set_id),
                "candidate_bays": _candidate_bays(assignment),
                "family_values": metrics["family_values"],
                "workday_values": metrics["workday_values"],
                "job_ids": metrics["job_ids"],
                "wo_count": metrics["wo_count"],
                "steel_quantity_sum": metrics["steel_quantity_sum"],
                "cut_length_sum": metrics["cut_length_sum"],
                "bevel_quantity_sum": metrics["bevel_quantity_sum"],
                "plate_length_sum": metrics["plate_length_sum"],
                "plate_length_max": metrics["plate_length_max"],
                "plate_width_max": metrics["plate_width_max"],
                "width_data_present": metrics["width_data_present"],
                "wide_plate_flag": metrics["wide_plate_flag"],
                "long_cut_over_1000": metrics["long_cut_over_1000"],
            }
        )
    return messages


def build_phase2_feedback_messages(
    scenario: Mapping[str, Any],
    plan: Mapping[str, Any],
    wide_bth_threshold: float = DEFAULT_WIDE_BTH_THRESHOLD,
    batch_max_wo_count: int = DEFAULT_BATCH_MAX_WO_COUNT,
    batch_max_length_sum: float = DEFAULT_BATCH_MAX_LENGTH_SUM,
) -> List[Dict[str, Any]]:
    """Build block-level feedback that Phase 2 can send back to Phase 1.

    The feedback is intentionally compact: it reports whether Phase 1's Bay
    choice is feasible for Phase 2 and which Bay choices need repair. It does
    not send back raw machine timelines or every local Phase 2 candidate.
    """

    if batch_max_wo_count <= 0:
        print(
            "[ERROR][phase1_phase2_communication.build_phase2_feedback_messages] "
            f"cause=invalid_batch_max_wo_count value={batch_max_wo_count}"
        )
        raise ValueError("batch_max_wo_count must be positive")
    if batch_max_length_sum <= 0:
        print(
            "[ERROR][phase1_phase2_communication.build_phase2_feedback_messages] "
            f"cause=invalid_batch_max_length_sum value={batch_max_length_sum}"
        )
        raise ValueError("batch_max_length_sum must be positive")

    phase1_messages = build_phase1_plan_messages(
        scenario=scenario,
        plan=plan,
        wide_bth_threshold=wide_bth_threshold,
    )
    jobs_by_block = _jobs_by_block(scenario)
    machines_by_bay = _machines_by_bay(scenario)
    feedback: List[Dict[str, Any]] = []
    for message in phase1_messages:
        block_set_id = message["block_set_id"]
        assigned_bay = str(message["assigned_bay"])
        block_jobs = _require_block_jobs(jobs_by_block, block_set_id)
        plate_lengths = [
            _require_non_negative_float(job.get("plate_length"), "plate_length", _job_id(job))
            for job in block_jobs
        ]
        batch_info = _estimate_batching(
            plate_lengths=plate_lengths,
            batch_max_wo_count=batch_max_wo_count,
            batch_max_length_sum=batch_max_length_sum,
        )
        reasons: List[str] = []
        recommended_bays = [assigned_bay]
        missing_machine_bay_count = 0
        wide_bay_violation_count = 0
        length_violation_count = batch_info["single_wo_length_violation_count"]
        if assigned_bay not in machines_by_bay:
            missing_machine_bay_count = 1
            reasons.append("assigned_bay_has_no_machine")
            recommended_bays = sorted(_candidate_bays_with_machines(message["candidate_bays"], machines_by_bay))
        if message["wide_plate_flag"] == 1 and assigned_bay not in WIDE_CAPABLE_BAYS:
            wide_bay_violation_count = 1
            reasons.append("wide_plate_requires_bay22_23")
            recommended_bays = sorted(
                bay_id
                for bay_id in WIDE_CAPABLE_BAYS
                if bay_id in machines_by_bay and bay_id in set(message["candidate_bays"])
            )
        if length_violation_count:
            reasons.append("single_wo_exceeds_batch_length_sum")

        hard_violation_count = missing_machine_bay_count + wide_bay_violation_count + length_violation_count
        if missing_machine_bay_count or length_violation_count:
            status = "infeasible"
        elif hard_violation_count:
            status = "repair_required"
        else:
            status = "ok"

        feedback.append(
            {
                "schema_version": COMMUNICATION_SCHEMA_VERSION,
                "direction": "phase2_to_phase1",
                "message_type": "block_feasibility_feedback",
                "message_scope": "block_feedback_only",
                "block_set_id": block_set_id,
                "assigned_bay": assigned_bay,
                "status": status,
                "reason_codes": reasons,
                "recommended_bays": recommended_bays,
                "wo_count": message["wo_count"],
                "estimated_batch_count": batch_info["estimated_batch_count"],
                "estimated_machine_count": len(machines_by_bay.get(assigned_bay, [])),
                "estimated_machine_load_gap": _estimate_machine_load_gap(
                    batch_lengths=batch_info["batch_length_sums"],
                    machine_ids=machines_by_bay.get(assigned_bay, []),
                ),
                "hard_violation_count": hard_violation_count,
                "wide_bay_violation_count": wide_bay_violation_count,
                "missing_machine_bay_count": missing_machine_bay_count,
                "single_wo_length_violation_count": length_violation_count,
                "batch_wo_count_violation_count": 0,
                "repair_priority": _repair_priority(
                    missing_machine_bay_count=missing_machine_bay_count,
                    length_violation_count=length_violation_count,
                    wide_bay_violation_count=wide_bay_violation_count,
                ),
            }
        )
    return feedback


def write_phase1_phase2_communication_package(
    scenario: Mapping[str, Any],
    plan: Mapping[str, Any],
    output_dir: str | Path,
    wide_bth_threshold: float = DEFAULT_WIDE_BTH_THRESHOLD,
    batch_max_wo_count: int = DEFAULT_BATCH_MAX_WO_COUNT,
    batch_max_length_sum: float = DEFAULT_BATCH_MAX_LENGTH_SUM,
) -> Dict[str, Any]:
    """Write JSONL/CSV artifacts for the Phase 1 <-> Phase 2 boundary."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    phase1_messages = build_phase1_plan_messages(
        scenario=scenario,
        plan=plan,
        wide_bth_threshold=wide_bth_threshold,
    )
    phase2_feedback = build_phase2_feedback_messages(
        scenario=scenario,
        plan=plan,
        wide_bth_threshold=wide_bth_threshold,
        batch_max_wo_count=batch_max_wo_count,
        batch_max_length_sum=batch_max_length_sum,
    )
    summary = _build_summary(phase1_messages, phase2_feedback)

    phase1_jsonl = output_path / "phase1_to_phase2_messages.jsonl"
    phase2_jsonl = output_path / "phase2_to_phase1_feedback.jsonl"
    feedback_csv = output_path / "phase2_feedback_summary.csv"
    manifest_json = output_path / "communication_manifest.json"

    _write_jsonl(phase1_jsonl, phase1_messages)
    _write_jsonl(phase2_jsonl, phase2_feedback)
    _write_feedback_csv(feedback_csv, phase2_feedback)
    with manifest_json.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "schema_version": COMMUNICATION_SCHEMA_VERSION,
                "wide_bth_threshold": wide_bth_threshold,
                "batch_max_wo_count": batch_max_wo_count,
                "batch_max_length_sum": batch_max_length_sum,
                "summary": summary,
                "files": {
                    "phase1_to_phase2_jsonl": str(phase1_jsonl),
                    "phase2_to_phase1_jsonl": str(phase2_jsonl),
                    "feedback_csv": str(feedback_csv),
                },
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "phase1_to_phase2_jsonl": str(phase1_jsonl),
        "phase2_to_phase1_jsonl": str(phase2_jsonl),
        "feedback_csv": str(feedback_csv),
        "manifest_json": str(manifest_json),
        "summary": summary,
    }


def build_phase2_feedback_score(feedback: Sequence[Mapping[str, Any]]) -> tuple[int, int, int, int, int]:
    """Return a compact score tuple Phase 1 can prepend to its own objective.

    Lower is better. The tuple intentionally puts hard feasibility before load
    terms, so a Phase 1 candidate that Phase 2 cannot execute loses before
    normal Bay workload balancing is considered.
    """

    if not feedback:
        print("[ERROR][phase1_phase2_communication.build_phase2_feedback_score] cause=no_feedback_messages")
        raise RuntimeError("Phase 2 feedback score requires non-empty feedback")
    for index, row in enumerate(feedback):
        if not isinstance(row, Mapping):
            print(
                "[ERROR][phase1_phase2_communication.build_phase2_feedback_score] "
                f"cause=invalid_feedback_row index={index} row_type={type(row).__name__}"
            )
            raise RuntimeError(f"invalid Phase 2 feedback row at index={index}")
    try:
        summary = _build_summary([], feedback)
    except (KeyError, TypeError, ValueError) as exc:
        print(
            "[ERROR][phase1_phase2_communication.build_phase2_feedback_score] "
            f"cause=invalid_feedback_schema error={exc}"
        )
        raise RuntimeError("invalid Phase 2 feedback schema") from exc
    return (
        int(summary["hard_violation_count"]),
        int(summary["infeasible_count"]),
        int(summary["repair_required_count"]),
        int(summary["wide_bay_violation_count"]),
        int(summary["estimated_batch_count_total"]),
    )


def score_phase1_assignments_with_phase2_feedback(
    assignments: Mapping[str, str],
    jobs: Mapping[str, Any],
    bay_ids: Sequence[str],
    machines: Sequence[Any] | Mapping[str, Any],
    algorithm: str = "phase1_pair_candidate",
    wide_bth_threshold: float = DEFAULT_WIDE_BTH_THRESHOLD,
    batch_max_wo_count: int = DEFAULT_BATCH_MAX_WO_COUNT,
    batch_max_length_sum: float = DEFAULT_BATCH_MAX_LENGTH_SUM,
) -> tuple[int, int, int, int, int]:
    """Score one Phase 1 candidate with the Phase 2 feedback contract.

    This is the bridge used by Phase 1 self-labeling. It converts current
    Job-like/Machine-like runtime objects into the same scenario/plan shape used
    by the JSONL communication package, then returns `build_phase2_feedback_score`.
    """

    scenario = _scenario_from_runtime_objects(jobs=jobs, machines=machines)
    plan = _plan_from_assignment_map(
        assignments=assignments,
        jobs=jobs,
        bay_ids=bay_ids,
        algorithm=algorithm,
    )
    feedback = build_phase2_feedback_messages(
        scenario=scenario,
        plan=plan,
        wide_bth_threshold=wide_bth_threshold,
        batch_max_wo_count=batch_max_wo_count,
        batch_max_length_sum=batch_max_length_sum,
    )
    return build_phase2_feedback_score(feedback)


def load_communication_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read communication JSONL rows without inventing missing messages."""

    input_path = Path(path)
    if not input_path.exists():
        print(
            "[ERROR][phase1_phase2_communication.load_communication_jsonl] "
            f"cause=missing_jsonl path={input_path}"
        )
        raise FileNotFoundError(f"communication JSONL does not exist: {input_path}")
    rows: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                print(
                    "[ERROR][phase1_phase2_communication.load_communication_jsonl] "
                    f"cause=invalid_json line_no={line_no} path={input_path}"
                )
                raise RuntimeError(f"invalid communication JSON at line {line_no}: {input_path}") from exc
            if not isinstance(row, dict):
                print(
                    "[ERROR][phase1_phase2_communication.load_communication_jsonl] "
                    f"cause=non_object_json line_no={line_no} path={input_path}"
                )
                raise RuntimeError(f"communication JSONL row must be object at line {line_no}")
            rows.append(row)
    if not rows:
        print(
            "[ERROR][phase1_phase2_communication.load_communication_jsonl] "
            f"cause=no_messages path={input_path}"
        )
        raise RuntimeError(f"communication JSONL has no messages: {input_path}")
    return rows


def apply_phase1_messages_to_scenario(
    scenario: Mapping[str, Any],
    phase1_messages: Sequence[Mapping[str, Any]],
    assignment_mode: str = "cut_bay",
) -> Dict[str, Any]:
    """Apply Phase 1 block-Bay messages to a Phase 2 scenario.

    `cut_bay` fixes the Phase 2 Bay directly.
    `allowed_bay_ids` keeps `cut_bay` open but masks Phase 2 candidates to the
    Bay chosen by Phase 1.
    """

    if assignment_mode not in {"cut_bay", "allowed_bay_ids"}:
        print(
            "[ERROR][phase1_phase2_communication.apply_phase1_messages_to_scenario] "
            f"cause=unknown_assignment_mode assignment_mode={assignment_mode}"
        )
        raise ValueError(f"unknown Phase 1 message assignment mode: {assignment_mode}")
    assignment_by_block = _message_assignment_map(phase1_messages)
    phase2_scenario = copy.deepcopy(dict(scenario))
    jobs = phase2_scenario.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        print(
            "[ERROR][phase1_phase2_communication.apply_phase1_messages_to_scenario] "
            f"cause=no_scenario_jobs jobs_type={type(jobs).__name__}"
        )
        raise RuntimeError("Phase 1 message apply requires a scenario with non-empty jobs")

    assigned_job_count = 0
    assigned_block_ids: set[str] = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            print(
                "[ERROR][phase1_phase2_communication.apply_phase1_messages_to_scenario] "
                f"cause=invalid_job_row index={index} row_type={type(job).__name__}"
            )
            raise RuntimeError(f"invalid job row at index={index}")
        job_id = _job_id(job)
        block_set_id = _require_text(job.get("block_set_id"), "block_set_id", job_id)
        assigned_bay = assignment_by_block.get(block_set_id)
        if assigned_bay is None:
            print(
                "[ERROR][phase1_phase2_communication.apply_phase1_messages_to_scenario] "
                f"cause=missing_phase1_message job_id={job_id} block_set_id={block_set_id}"
            )
            raise RuntimeError(f"missing_phase1_message: {block_set_id}")
        if assignment_mode == "cut_bay":
            job["cut_bay"] = assigned_bay
        else:
            job["allowed_bay_ids"] = [assigned_bay]
        assigned_job_count += 1
        assigned_block_ids.add(block_set_id)

    metadata = phase2_scenario.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        print(
            "[ERROR][phase1_phase2_communication.apply_phase1_messages_to_scenario] "
            f"cause=invalid_metadata metadata_type={type(metadata).__name__}"
        )
        raise RuntimeError("scenario metadata must be a mapping")
    metadata["phase1_message_applied"] = True
    metadata["phase1_message_assignment_mode"] = assignment_mode
    metadata["phase1_message_assigned_block_count"] = len(assigned_block_ids)
    metadata["phase1_message_assigned_job_count"] = assigned_job_count

    return {
        "scenario": phase2_scenario,
        "summary": {
            "assignment_mode": assignment_mode,
            "assigned_job_count": assigned_job_count,
            "assigned_block_count": len(assigned_block_ids),
            "message_assignment_count": len(assignment_by_block),
        },
    }


def _build_summary(phase1_messages: Sequence[Mapping[str, Any]], feedback: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize communication artifacts without changing their row grain."""

    return {
        "block_message_count": len(phase1_messages),
        "feedback_message_count": len(feedback),
        "repair_required_count": sum(1 for row in feedback if row["status"] == "repair_required"),
        "infeasible_count": sum(1 for row in feedback if row["status"] == "infeasible"),
        "hard_violation_count": sum(int(row["hard_violation_count"]) for row in feedback),
        "wide_bay_violation_count": sum(int(row["wide_bay_violation_count"]) for row in feedback),
        "missing_machine_bay_count": sum(int(row["missing_machine_bay_count"]) for row in feedback),
        "single_wo_length_violation_count": sum(int(row["single_wo_length_violation_count"]) for row in feedback),
        "estimated_batch_count_total": sum(int(row["estimated_batch_count"]) for row in feedback),
    }


def _scenario_from_runtime_objects(
    jobs: Mapping[str, Any],
    machines: Sequence[Any] | Mapping[str, Any],
) -> Dict[str, Any]:
    """Convert in-memory Job/Machine objects into communication scenario rows."""

    if not isinstance(jobs, Mapping) or not jobs:
        print(
            "[ERROR][phase1_phase2_communication._scenario_from_runtime_objects] "
            f"cause=no_jobs jobs_type={type(jobs).__name__}"
        )
        raise RuntimeError("Phase 2 feedback scoring requires non-empty jobs")
    machine_values = list(machines.values()) if isinstance(machines, Mapping) else list(machines)
    if not machine_values:
        print("[ERROR][phase1_phase2_communication._scenario_from_runtime_objects] cause=no_machines")
        raise RuntimeError("Phase 2 feedback scoring requires non-empty machines")
    return {
        "metadata": {"scenario_name": "phase1_candidate_feedback_score"},
        "jobs": [_job_object_to_scenario_row(job_key, job) for job_key, job in sorted(jobs.items())],
        "machines": [_machine_object_to_scenario_row(index, machine) for index, machine in enumerate(machine_values)],
    }


def _plan_from_assignment_map(
    assignments: Mapping[str, str],
    jobs: Mapping[str, Any],
    bay_ids: Sequence[str],
    algorithm: str,
) -> Dict[str, Any]:
    """Convert a `block_set_id -> Bay` candidate into Phase 1 plan rows."""

    if not isinstance(assignments, Mapping) or not assignments:
        print(
            "[ERROR][phase1_phase2_communication._plan_from_assignment_map] "
            f"cause=no_assignments assignments_type={type(assignments).__name__}"
        )
        raise RuntimeError("Phase 2 feedback scoring requires non-empty assignments")
    normalized_bays = [str(bay_id) for bay_id in bay_ids]
    if not normalized_bays:
        print("[ERROR][phase1_phase2_communication._plan_from_assignment_map] cause=no_bay_ids")
        raise RuntimeError("Phase 2 feedback scoring requires non-empty bay_ids")

    block_rows: Dict[str, Dict[str, str]] = {}
    for job_key, job in jobs.items():
        job_id = _require_text(_object_field(job, "job_id"), "job_id", str(job_key))
        block_set_id = _require_text(_object_field(job, "block_set_id"), "block_set_id", job_id)
        if block_set_id in block_rows:
            continue
        project_no, block_no = _project_block_labels_from_object(block_set_id, job)
        block_rows[block_set_id] = {"project_no": project_no, "block_no": block_no}

    missing_blocks = sorted(set(block_rows) - set(assignments))
    extra_blocks = sorted(set(assignments) - set(block_rows))
    if missing_blocks or extra_blocks:
        print(
            "[ERROR][phase1_phase2_communication._plan_from_assignment_map] "
            f"cause=assignment_block_mismatch missing={missing_blocks[:5]} extra={extra_blocks[:5]}"
        )
        raise RuntimeError("Phase 2 feedback scoring assignments must cover exactly the runtime blocks")

    rows: List[Dict[str, Any]] = []
    for block_set_id in sorted(block_rows):
        assigned_bay = _require_text(assignments.get(block_set_id), "assigned_bay", block_set_id)
        if assigned_bay not in normalized_bays:
            print(
                "[ERROR][phase1_phase2_communication._plan_from_assignment_map] "
                f"cause=assignment_bay_not_in_candidate_bays block_set_id={block_set_id} "
                f"assigned_bay={assigned_bay} bay_ids={normalized_bays}"
            )
            raise RuntimeError(f"assignment_bay_not_in_candidate_bays: {block_set_id}")
        labels = block_rows[block_set_id]
        rows.append(
            {
                "block_set_id": block_set_id,
                "project_no": labels["project_no"],
                "block_no": labels["block_no"],
                "assigned_bay": assigned_bay,
                "candidate_bays": "|".join(normalized_bays),
            }
        )
    return {"algorithm": algorithm, "assignments": rows}


def _job_object_to_scenario_row(job_key: str, job: Any) -> Dict[str, Any]:
    """Return a scenario job row from a Job-like object without defaulting data."""

    job_id = _require_text(_object_field(job, "job_id"), "job_id", str(job_key))
    block_set_id = _require_text(_object_field(job, "block_set_id"), "block_set_id", job_id)
    extra = _object_extra(job)
    row = {
        "job_id": job_id,
        "source_wk_ord_no": _optional_text(extra.get("source_wk_ord_no")) or job_id,
        "block_set_id": block_set_id,
        "family": _optional_text(_object_field(job, "family") or extra.get("family")),
        "steel_quantity": _object_field(job, "steel_quantity"),
        "cut_length": _object_field(job, "cut_length"),
        "bevel_quantity": _object_field(job, "bevel_quantity"),
        "plate_length": _object_field(job, "plate_length"),
        "plate_width": _object_field(job, "plate_width"),
        "workday": _object_field(job, "workday") or extra.get("source_workday"),
        "source_project_no": extra.get("source_project_no"),
        "source_block_no": extra.get("source_block_no"),
    }
    return row


def _machine_object_to_scenario_row(index: int, machine: Any) -> Dict[str, Any]:
    """Return a scenario machine row from a Machine-like object."""

    machine_id = _require_text(_object_field(machine, "machine_id"), "machine_id", f"machine_{index}")
    bay_id = _require_text(_object_field(machine, "bay_id"), "bay_id", machine_id)
    row = {"machine_id": machine_id, "bay_id": bay_id}
    enabled = _object_field(machine, "enabled")
    if enabled is not None:
        row["enabled"] = bool(enabled)
    return row


def _project_block_labels_from_object(block_set_id: str, job: Any) -> tuple[str, str]:
    """Return project/block labels from job.extra or the block id."""

    extra = _object_extra(job)
    project_no = _optional_text(extra.get("source_project_no"))
    block_no = _optional_text(extra.get("source_block_no"))
    if (not project_no or not block_no) and "::" in block_set_id:
        project_no, block_no = block_set_id.split("::", 1)
    return project_no or "", block_no or ""


def _object_extra(item: Any) -> Mapping[str, Any]:
    """Read optional `.extra`/`['extra']` mapping from runtime objects."""

    extra = _object_field(item, "extra")
    return extra if isinstance(extra, Mapping) else {}


def _object_field(item: Any, field_name: str) -> Any:
    """Read a field from mapping or object attribute without alias guessing."""

    if isinstance(item, Mapping):
        return item.get(field_name)
    return getattr(item, field_name, None)


def _message_assignment_map(phase1_messages: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Return `block_set_id -> assigned_bay` from Phase 1 messages."""

    if not phase1_messages:
        print("[ERROR][phase1_phase2_communication._message_assignment_map] cause=no_phase1_messages")
        raise RuntimeError("Phase 1 message apply requires non-empty messages")
    result: Dict[str, str] = {}
    for index, row in enumerate(phase1_messages):
        if not isinstance(row, Mapping):
            print(
                "[ERROR][phase1_phase2_communication._message_assignment_map] "
                f"cause=invalid_message_row index={index} row_type={type(row).__name__}"
            )
            raise RuntimeError(f"invalid Phase 1 message row at index={index}")
        direction = _require_text(row.get("direction"), "direction", f"message_{index}")
        if direction != "phase1_to_phase2":
            print(
                "[ERROR][phase1_phase2_communication._message_assignment_map] "
                f"cause=unexpected_direction index={index} direction={direction}"
            )
            raise RuntimeError(f"expected phase1_to_phase2 message at index={index}")
        block_set_id = _require_text(row.get("block_set_id"), "block_set_id", f"message_{index}")
        assigned_bay = _require_text(row.get("assigned_bay"), "assigned_bay", block_set_id)
        if block_set_id in result:
            print(
                "[ERROR][phase1_phase2_communication._message_assignment_map] "
                f"cause=duplicate_block_message block_set_id={block_set_id}"
            )
            raise RuntimeError(f"duplicate Phase 1 message for block: {block_set_id}")
        result[block_set_id] = assigned_bay
    return result


def _assignment_rows_by_block(plan: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    """Return unique Phase 1 assignment rows keyed by block ID."""

    rows = plan.get("assignments")
    if not isinstance(rows, list) or not rows:
        print(
            "[ERROR][phase1_phase2_communication._assignment_rows_by_block] "
            f"cause=no_assignments rows_type={type(rows).__name__}"
        )
        raise RuntimeError("Phase 1 communication requires non-empty plan assignments")
    result: Dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            print(
                "[ERROR][phase1_phase2_communication._assignment_rows_by_block] "
                f"cause=invalid_assignment_row index={index} row_type={type(row).__name__}"
            )
            raise RuntimeError(f"invalid Phase 1 assignment row at index={index}")
        block_set_id = _require_text(row.get("block_set_id"), "block_set_id", f"assignment_{index}")
        if block_set_id in result:
            print(
                "[ERROR][phase1_phase2_communication._assignment_rows_by_block] "
                f"cause=duplicate_block_assignment block_set_id={block_set_id}"
            )
            raise RuntimeError(f"duplicate Phase 1 assignment: {block_set_id}")
        result[block_set_id] = row
    return result


def _jobs_by_block(scenario: Mapping[str, Any]) -> Dict[str, List[Mapping[str, Any]]]:
    """Group scenario jobs by `block_set_id`."""

    rows = scenario.get("jobs")
    if not isinstance(rows, list) or not rows:
        print(
            "[ERROR][phase1_phase2_communication._jobs_by_block] "
            f"cause=no_jobs rows_type={type(rows).__name__}"
        )
        raise RuntimeError("Phase 1/2 communication requires non-empty scenario jobs")
    result: Dict[str, List[Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            print(
                "[ERROR][phase1_phase2_communication._jobs_by_block] "
                f"cause=invalid_job_row index={index} row_type={type(row).__name__}"
            )
            raise RuntimeError(f"invalid scenario job row at index={index}")
        job_id = _job_id(row)
        block_set_id = _require_text(row.get("block_set_id"), "block_set_id", job_id)
        result.setdefault(block_set_id, []).append(row)
    return result


def _machines_by_bay(scenario: Mapping[str, Any]) -> Dict[str, List[str]]:
    """Group enabled scenario machines by cutting Bay."""

    rows = scenario.get("machines")
    if not isinstance(rows, list) or not rows:
        print(
            "[ERROR][phase1_phase2_communication._machines_by_bay] "
            f"cause=no_machines rows_type={type(rows).__name__}"
        )
        raise RuntimeError("Phase 2 feedback requires non-empty scenario machines")
    result: Dict[str, List[str]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            print(
                "[ERROR][phase1_phase2_communication._machines_by_bay] "
                f"cause=invalid_machine_row index={index} row_type={type(row).__name__}"
            )
            raise RuntimeError(f"invalid scenario machine row at index={index}")
        if row.get("enabled") is False:
            continue
        machine_id = _require_text(row.get("machine_id"), "machine_id", f"machine_{index}")
        bay_id = _require_text(row.get("bay_id"), "bay_id", machine_id)
        result.setdefault(bay_id, []).append(machine_id)
    if not result:
        print("[ERROR][phase1_phase2_communication._machines_by_bay] cause=no_enabled_machines")
        raise RuntimeError("Phase 2 feedback requires at least one enabled machine")
    return {bay_id: sorted(machine_ids) for bay_id, machine_ids in result.items()}


def _aggregate_block_jobs(
    block_set_id: str,
    block_jobs: Sequence[Mapping[str, Any]],
    wide_bth_threshold: float,
) -> Dict[str, Any]:
    """Aggregate only the block-owned features that Phase 2 needs."""

    job_ids: List[str] = []
    family_values: set[str] = set()
    workday_values: set[str] = set()
    steel_quantity_sum = 0
    cut_length_sum = 0.0
    bevel_quantity_sum = 0
    plate_length_sum = 0.0
    plate_length_max = 0.0
    plate_width_values: List[float] = []
    for job in block_jobs:
        job_id = _job_id(job)
        job_ids.append(job_id)
        family = _optional_text(job.get("family"))
        if family:
            family_values.add(family)
        workday = _optional_text(job.get("workday") or job.get("actual_workday") or job.get("planned_start_date"))
        if workday:
            workday_values.add(workday)
        steel_quantity_sum += _require_positive_int(job.get("steel_quantity"), "steel_quantity", job_id)
        cut_length_sum += _require_non_negative_float(job.get("cut_length"), "cut_length", job_id)
        bevel_quantity_sum += _require_non_negative_int(job.get("bevel_quantity"), "bevel_quantity", job_id)
        plate_length = _require_non_negative_float(job.get("plate_length"), "plate_length", job_id)
        plate_length_sum += plate_length
        plate_length_max = max(plate_length_max, plate_length)
        plate_width = _optional_non_negative_float(job.get("plate_width"), "plate_width", job_id)
        if plate_width is not None:
            plate_width_values.append(plate_width)
        if job.get("is_wide_plate") is True:
            plate_width_values.append(wide_bth_threshold + 1.0)

    plate_width_max = max(plate_width_values) if plate_width_values else None
    project_no, block_no = _project_block_labels(block_set_id, block_jobs)
    return {
        "project_no": project_no,
        "block_no": block_no,
        "job_ids": sorted(job_ids),
        "family_values": sorted(family_values),
        "workday_values": sorted(workday_values),
        "wo_count": len(block_jobs),
        "steel_quantity_sum": steel_quantity_sum,
        "cut_length_sum": round(cut_length_sum, 6),
        "bevel_quantity_sum": bevel_quantity_sum,
        "plate_length_sum": round(plate_length_sum, 6),
        "plate_length_max": round(plate_length_max, 6),
        "plate_width_max": None if plate_width_max is None else round(plate_width_max, 6),
        "width_data_present": 1 if plate_width_values else 0,
        "wide_plate_flag": None if plate_width_max is None else int(plate_width_max > wide_bth_threshold),
        "long_cut_over_1000": int(cut_length_sum > 1000.0),
    }


def _estimate_batching(
    plate_lengths: Sequence[float],
    batch_max_wo_count: int,
    batch_max_length_sum: float,
) -> Dict[str, Any]:
    """Estimate Phase 2 batch count by first-fit decreasing on W/O length."""

    single_wo_length_violation_count = sum(1 for length in plate_lengths if length > batch_max_length_sum)
    if single_wo_length_violation_count:
        return {
            "estimated_batch_count": 0,
            "batch_length_sums": [],
            "single_wo_length_violation_count": single_wo_length_violation_count,
        }
    batches: List[Dict[str, float | int]] = []
    for length in sorted(plate_lengths, reverse=True):
        placed = False
        for batch in batches:
            next_count = int(batch["count"]) + 1
            next_length_sum = float(batch["length_sum"]) + length
            if next_count <= batch_max_wo_count and next_length_sum <= batch_max_length_sum:
                batch["count"] = next_count
                batch["length_sum"] = next_length_sum
                placed = True
                break
        if not placed:
            batches.append({"count": 1, "length_sum": length})
    return {
        "estimated_batch_count": len(batches),
        "batch_length_sums": [float(batch["length_sum"]) for batch in batches],
        "single_wo_length_violation_count": 0,
    }


def _estimate_machine_load_gap(batch_lengths: Sequence[float], machine_ids: Sequence[str]) -> float:
    """Estimate machine load gap from batch lengths within the assigned Bay."""

    if not machine_ids:
        return 0.0
    loads = {machine_id: 0.0 for machine_id in machine_ids}
    for batch_length in sorted(batch_lengths, reverse=True):
        machine_id = min(loads, key=lambda key: (loads[key], key))
        loads[machine_id] += batch_length
    values = list(loads.values())
    return round(max(values) - min(values), 6) if values else 0.0


def _repair_priority(
    missing_machine_bay_count: int,
    length_violation_count: int,
    wide_bay_violation_count: int,
) -> int:
    """Return lower-is-more-urgent repair priority."""

    if missing_machine_bay_count:
        return 1
    if length_violation_count:
        return 2
    if wide_bay_violation_count:
        return 3
    return 0


def _candidate_bays(assignment: Mapping[str, Any]) -> List[str]:
    """Read candidate Bay list from a plan row."""

    raw = assignment.get("candidate_bays")
    if isinstance(raw, str) and raw.strip():
        return [part.strip() for part in raw.split("|") if part.strip()]
    assigned_bay = _require_text(assignment.get("assigned_bay"), "assigned_bay", str(assignment))
    return [assigned_bay]


def _candidate_bays_with_machines(candidate_bays: Sequence[str], machines_by_bay: Mapping[str, Sequence[str]]) -> List[str]:
    """Return candidate Bays that currently have at least one machine."""

    return [bay_id for bay_id in candidate_bays if bay_id in machines_by_bay]


def _require_block_jobs(
    jobs_by_block: Mapping[str, Sequence[Mapping[str, Any]]],
    block_set_id: str,
) -> Sequence[Mapping[str, Any]]:
    """Return jobs for one block or fail loudly."""

    block_jobs = jobs_by_block.get(block_set_id)
    if not block_jobs:
        print(
            "[ERROR][phase1_phase2_communication._require_block_jobs] "
            f"cause=missing_block_jobs block_set_id={block_set_id}"
        )
        raise RuntimeError(f"missing scenario jobs for Phase 1 block: {block_set_id}")
    return block_jobs


def _project_block_labels(block_set_id: str, block_jobs: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """Return project/block labels from source columns or `project::block` ID."""

    first_job = block_jobs[0]
    project_no = _optional_text(first_job.get("source_project_no"))
    block_no = _optional_text(first_job.get("source_block_no"))
    if (not project_no or not block_no) and "::" in block_set_id:
        project_no, block_no = block_set_id.split("::", 1)
    return project_no or "", block_no or ""


def _job_id(job: Mapping[str, Any]) -> str:
    """Return the visible W/O id used in reports and message joins."""

    return _require_text(job.get("source_wk_ord_no") or job.get("job_id"), "job_id", str(job))


def _optional_text(value: Any) -> str:
    """Return stripped text or empty string for optional display fields."""

    return "" if value in (None, "") else str(value).strip()


def _require_text(value: Any, field_name: str, row_key: str) -> str:
    """Require non-empty text without substituting another field."""

    text = _optional_text(value)
    if not text:
        print(
            "[ERROR][phase1_phase2_communication._require_text] "
            f"cause=missing_{field_name} row_key={row_key}"
        )
        raise RuntimeError(f"missing_{field_name}: {row_key}")
    return text


def _require_positive_int(value: Any, field_name: str, row_key: str) -> int:
    """Require a positive integer."""

    parsed = _parse_int(value, field_name, row_key)
    if parsed <= 0:
        print(
            "[ERROR][phase1_phase2_communication._require_positive_int] "
            f"cause=non_positive_{field_name} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"non_positive_{field_name}: {row_key}")
    return parsed


def _require_non_negative_int(value: Any, field_name: str, row_key: str) -> int:
    """Require a non-negative integer."""

    parsed = _parse_int(value, field_name, row_key)
    if parsed < 0:
        print(
            "[ERROR][phase1_phase2_communication._require_non_negative_int] "
            f"cause=negative_{field_name} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"negative_{field_name}: {row_key}")
    return parsed


def _parse_int(value: Any, field_name: str, row_key: str) -> int:
    """Parse an integer field and fail without fallback."""

    if value in (None, ""):
        print(
            "[ERROR][phase1_phase2_communication._parse_int] "
            f"cause=missing_{field_name} row_key={row_key}"
        )
        raise RuntimeError(f"missing_{field_name}: {row_key}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][phase1_phase2_communication._parse_int] "
            f"cause=invalid_{field_name} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"invalid_{field_name}: {row_key}") from exc


def _require_non_negative_float(value: Any, field_name: str, row_key: str) -> float:
    """Require a non-negative float."""

    if value in (None, ""):
        print(
            "[ERROR][phase1_phase2_communication._require_non_negative_float] "
            f"cause=missing_{field_name} row_key={row_key}"
        )
        raise RuntimeError(f"missing_{field_name}: {row_key}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][phase1_phase2_communication._require_non_negative_float] "
            f"cause=invalid_{field_name} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"invalid_{field_name}: {row_key}") from exc
    if parsed < 0 or math.isnan(parsed):
        print(
            "[ERROR][phase1_phase2_communication._require_non_negative_float] "
            f"cause=negative_or_nan_{field_name} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"negative_or_nan_{field_name}: {row_key}")
    return parsed


def _optional_non_negative_float(value: Any, field_name: str, row_key: str) -> float | None:
    """Parse optional non-negative float while preserving missing as None."""

    if value in (None, ""):
        return None
    return _require_non_negative_float(value, field_name, row_key)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write one JSON object per line."""

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_feedback_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the human-readable Phase 2 feedback table."""

    fields = [
        "block_set_id",
        "assigned_bay",
        "status",
        "reason_codes",
        "recommended_bays",
        "wo_count",
        "estimated_batch_count",
        "estimated_machine_count",
        "estimated_machine_load_gap",
        "hard_violation_count",
        "wide_bay_violation_count",
        "missing_machine_bay_count",
        "single_wo_length_violation_count",
        "batch_wo_count_violation_count",
        "repair_priority",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: "|".join(str(item) for item in row[field])
                    if isinstance(row.get(field), list)
                    else row.get(field)
                    for field in fields
                }
            )
