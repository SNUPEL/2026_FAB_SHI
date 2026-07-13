"""공통 계층 환경에서 Phase 2 set-pointer 입력 state를 만든다."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Sequence

from Environment.hierarchical import CommonHierarchicalEnvironment, OpenBatchState


PHASE2_STATE_SCHEMA_VERSION = "phase2_set_pointer_v1"
PHASE2_SET_POINTER_POLICY_TYPE = "phase2_set_pointer"
PHASE2_BAY_CONTEXT_FEATURE_NAMES = (
    "stage_select_machine",
    "stage_select_wo",
    "progress_ratio",
    "remaining_wo_ratio",
    "remaining_tact_ratio",
    "remaining_lth_ratio",
    "remaining_cut_ratio",
    "remaining_bevel_ratio",
    "normalized_wo_gap",
    "normalized_cut_gap",
    "normalized_bevel_gap",
    "normalized_occupancy_gap",
    "machine_count_ratio",
)
PHASE2_MACHINE_NODE_FEATURE_NAMES = (
    "clock_to_target_ratio",
    "wo_to_target_ratio",
    "cut_to_target_ratio",
    "bevel_to_target_ratio",
    "batch_to_expected_ratio",
    "feasible_remaining_wo_ratio",
    "is_selected_machine",
)
PHASE2_WO_NODE_FEATURE_NAMES = (
    "tact_ratio",
    "length_to_limit_ratio",
    "cut_ratio",
    "bevel_ratio",
    "is_in_open_batch",
)
PHASE2_OPEN_BATCH_FEATURE_NAMES = (
    "wo_count_ratio",
    "length_sum_ratio",
    "max_tact_ratio",
    "cut_sum_ratio",
    "bevel_sum_ratio",
    "remaining_slot_ratio",
    "remaining_length_ratio",
)
PHASE2_PROJECTED_FEATURE_NAMES = (
    "duration_increment_ratio",
    "projected_finish_ratio",
    "projected_makespan_ratio",
    "projected_normalized_wo_gap",
    "projected_normalized_cut_gap",
    "projected_normalized_bevel_gap",
    "projected_normalized_occupancy_gap",
)


def phase2_state_feature_schema() -> Dict[str, list[str]]:
    """Checkpoint와 evaluator가 공유하는 Phase 2 feature 계약을 반환한다.

    새 dict/list를 반환해 호출자가 값을 변경해도 모듈의 기준 계약이 오염되지
    않게 한다. 이 순서는 각 MLP 입력 열 순서이므로 checkpoint 호환성의 일부다.
    """

    return {
        "bay_context": list(PHASE2_BAY_CONTEXT_FEATURE_NAMES),
        "machine_node": list(PHASE2_MACHINE_NODE_FEATURE_NAMES),
        "wo_node": list(PHASE2_WO_NODE_FEATURE_NAMES),
        "open_batch": list(PHASE2_OPEN_BATCH_FEATURE_NAMES),
        "projected_action": list(PHASE2_PROJECTED_FEATURE_NAMES),
    }


@dataclass(frozen=True)
class Phase2PolicyState:
    schema_version: str
    stage: str
    bay_id: str
    bay_context_features: list[float]
    machine_ids: list[str]
    machine_node_features: list[list[float]]
    wo_ids: list[str]
    wo_node_features: list[list[float]]
    open_batch_features: list[float]
    action_ids: list[str]
    action_projected_features: list[list[float]]
    action_candidate_node_indices: list[int]
    selected_machine_node_index: int


def build_phase2_policy_state(
    *,
    environment: CommonHierarchicalEnvironment,
    bay_id: str,
    stage: str,
    actions: Sequence[Mapping[str, Any]],
    selected_machine_id: str | None = None,
    open_batch_id: str | None = None,
) -> Phase2PolicyState:
    """가변 Machine/W/O set과 action-conditioned projected feature를 만든다."""

    normalized_stage = str(stage).strip().upper()
    if normalized_stage not in {"SELECT_MACHINE", "SELECT_WO"}:
        print(f"[ERROR][Phase2.state.build_phase2_policy_state] cause=unknown_stage stage={stage}")
        raise RuntimeError(f"unknown Phase 2 policy stage: {stage}")
    if not actions:
        print("[ERROR][Phase2.state.build_phase2_policy_state] cause=no_actions")
        raise RuntimeError("Phase 2 policy state requires feasible actions")
    if normalized_stage == "SELECT_WO" and (not selected_machine_id or not open_batch_id):
        print(
            "[ERROR][Phase2.state.build_phase2_policy_state] "
            "cause=missing_selected_machine_or_open_batch"
        )
        raise RuntimeError("SELECT_WO state requires selected machine and open batch")

    normalized_bay = str(bay_id)
    view = environment.phase2_bay_view(normalized_bay)
    machine_ids = sorted(str(value) for value in view.machines)
    wo_ids = sorted(str(value) for value in view.jobs)
    if not machine_ids or not wo_ids:
        print(
            "[ERROR][Phase2.state.build_phase2_policy_state] "
            f"cause=empty_node_set bay_id={normalized_bay} machines={len(machine_ids)} wos={len(wo_ids)}"
        )
        raise RuntimeError("Phase 2 policy state requires non-empty machine and W/O sets")

    all_bay_jobs = {
        str(job_id): job
        for job_id, job in environment.jobs.items()
        if environment.state.planning.block_to_bay[_text_field(job, "block_set_id", str(job_id))] == normalized_bay
    }
    totals = _job_totals(all_bay_jobs)
    remaining_totals = _job_totals(view.jobs)
    machine_count = len(machine_ids)
    target = {
        "wo": totals["wo"] / machine_count,
        "cut": totals["cut"] / machine_count,
        "bevel": totals["bevel"] / machine_count,
        "occupancy": totals["tact"] / (machine_count * environment.max_batch_wo_count),
        "batch": math.ceil(totals["wo"] / environment.max_batch_wo_count) / machine_count,
    }
    current_gaps = _machine_gaps(view.machine_loads, view.machine_available_at)
    scheduled_count = len(all_bay_jobs) - len(view.jobs)
    bay_context = [
        1.0 if normalized_stage == "SELECT_MACHINE" else 0.0,
        1.0 if normalized_stage == "SELECT_WO" else 0.0,
        scheduled_count / totals["wo"],
        remaining_totals["wo"] / totals["wo"],
        remaining_totals["tact"] / totals["tact"],
        remaining_totals["length"] / totals["length"],
        remaining_totals["cut"] / totals["cut"],
        remaining_totals["bevel"] / max(1.0, totals["bevel"]),
        current_gaps["wo"],
        current_gaps["cut"],
        current_gaps["bevel"],
        current_gaps["occupancy"],
        machine_count / max(1.0, float(len(environment.machines))),
    ]

    selected_machine_index = -1
    machine_features: list[list[float]] = []
    for index, machine_id in enumerate(machine_ids):
        if selected_machine_id is not None and machine_id == str(selected_machine_id):
            selected_machine_index = index
        load = view.machine_loads[machine_id]
        feasible_count = sum(
            1 for job in view.jobs.values() if _job_allows_machine(job, machine_id)
        )
        machine_features.append(
            [
                float(view.machine_available_at[machine_id]) / max(1.0, target["occupancy"]),
                _machine_load_number(load, "wo_count", machine_id) / max(1.0, target["wo"]),
                _machine_load_number(load, "cut_length_sum", machine_id) / max(1.0, target["cut"]),
                _machine_load_number(load, "bevel_quantity_sum", machine_id) / max(1.0, target["bevel"]),
                _machine_load_number(load, "batch_count", machine_id) / max(1.0, target["batch"]),
                feasible_count / max(1.0, float(len(view.jobs))),
                1.0 if machine_id == str(selected_machine_id) else 0.0,
            ]
        )
    if normalized_stage == "SELECT_WO" and selected_machine_index < 0:
        print(
            "[ERROR][Phase2.state.build_phase2_policy_state] "
            f"cause=selected_machine_outside_bay machine_id={selected_machine_id} bay_id={normalized_bay}"
        )
        raise RuntimeError("selected machine is outside the Phase 2 Bay view")

    open_batch = _resolve_open_batch(environment, open_batch_id)
    open_job_ids = set(open_batch.job_ids) if open_batch is not None else set()
    wo_features = [
        [
            _processing_time(view.jobs[job_id], job_id) / totals["tact"],
            _number_field(view.jobs[job_id], "plate_length", job_id) / environment.max_batch_length_sum,
            _number_field(view.jobs[job_id], "cut_length", job_id) / totals["cut"],
            _number_field(view.jobs[job_id], "bevel_quantity", job_id) / max(1.0, totals["bevel"]),
            1.0 if job_id in open_job_ids else 0.0,
        ]
        for job_id in wo_ids
    ]
    open_features = _open_batch_features(environment, open_batch, totals)

    action_ids: list[str] = []
    projected_features: list[list[float]] = []
    candidate_indices: list[int] = []
    for action_index, action in enumerate(actions):
        action_type = str(action.get("action_type") or "").strip().lower()
        expected_type = "select_machine" if normalized_stage == "SELECT_MACHINE" else "select_wo"
        if action_type != expected_type:
            print(
                "[ERROR][Phase2.state.build_phase2_policy_state] "
                f"cause=action_stage_mismatch index={action_index} action_type={action_type} stage={normalized_stage}"
            )
            raise RuntimeError("Phase 2 action type does not match policy stage")
        machine_id = str(action.get("machine_id") or "")
        if machine_id not in machine_ids:
            print(
                "[ERROR][Phase2.state.build_phase2_policy_state] "
                f"cause=action_machine_outside_bay machine_id={machine_id}"
            )
            raise RuntimeError("Phase 2 action machine is outside Bay view")
        if normalized_stage == "SELECT_MACHINE":
            node_index = machine_ids.index(machine_id)
            action_id = str(action.get("action_id") or f"machine:{machine_id}")
        else:
            action_job_ids = tuple(str(value) for value in action.get("job_ids", ()))
            if len(action_job_ids) != 1 or action_job_ids[0] not in wo_ids:
                print(
                    "[ERROR][Phase2.state.build_phase2_policy_state] "
                    f"cause=invalid_wo_action index={action_index} job_ids={action_job_ids}"
                )
                raise RuntimeError("SELECT_WO action must reference one remaining W/O")
            node_index = wo_ids.index(action_job_ids[0])
            action_id = str(action.get("action_id") or f"wo:{action_job_ids[0]}")
        action_ids.append(action_id)
        candidate_indices.append(node_index)
        projected_features.append(
            _projected_action_features(
                action=action,
                machine_id=machine_id,
                machine_ids=machine_ids,
                machine_loads=view.machine_loads,
                machine_available_at=view.machine_available_at,
                totals=totals,
            )
        )

    return Phase2PolicyState(
        schema_version=PHASE2_STATE_SCHEMA_VERSION,
        stage=normalized_stage,
        bay_id=normalized_bay,
        bay_context_features=bay_context,
        machine_ids=machine_ids,
        machine_node_features=machine_features,
        wo_ids=wo_ids,
        wo_node_features=wo_features,
        open_batch_features=open_features,
        action_ids=action_ids,
        action_projected_features=projected_features,
        action_candidate_node_indices=candidate_indices,
        selected_machine_node_index=selected_machine_index,
    )


def _projected_action_features(
    *,
    action: Mapping[str, Any],
    machine_id: str,
    machine_ids: Sequence[str],
    machine_loads: Mapping[str, Mapping[str, int | float]],
    machine_available_at: Mapping[str, float],
    totals: Mapping[str, float],
) -> list[float]:
    projected_loads = {key: dict(value) for key, value in machine_loads.items()}
    projected_clock = {key: float(value) for key, value in machine_available_at.items()}
    if str(action["action_type"]) == "select_wo":
        projected_loads[machine_id]["wo_count"] += int(action["wo_count"])
        projected_loads[machine_id]["cut_length_sum"] += float(action["cut_length_sum"])
        projected_loads[machine_id]["bevel_quantity_sum"] += float(action["bevel_quantity_sum"])
        projected_clock[machine_id] = float(action["projected_finish_time"])
    gaps = _machine_gaps(projected_loads, projected_clock)
    time_scale = max(1.0, totals["tact"])
    return [
        float(action.get("duration_increment", 0.0)) / time_scale,
        float(action["projected_finish_time"]) / time_scale,
        float(action["projected_makespan"]) / time_scale,
        gaps["wo"],
        gaps["cut"],
        gaps["bevel"],
        gaps["occupancy"],
    ]


def _machine_gaps(
    machine_loads: Mapping[str, Mapping[str, int | float]],
    machine_available_at: Mapping[str, float],
) -> Dict[str, float]:
    return {
        "wo": _normalized_gap([float(row["wo_count"]) for row in machine_loads.values()]),
        "cut": _normalized_gap([float(row["cut_length_sum"]) for row in machine_loads.values()]),
        "bevel": _normalized_gap([float(row["bevel_quantity_sum"]) for row in machine_loads.values()]),
        "occupancy": _normalized_gap([float(machine_available_at[key]) for key in machine_loads]),
    }


def _normalized_gap(values: Sequence[float]) -> float:
    if not values:
        print("[ERROR][Phase2.state._normalized_gap] cause=no_values")
        raise RuntimeError("normalized gap requires values")
    mean = sum(values) / len(values)
    return 0.0 if mean == 0.0 else (max(values) - min(values)) / mean


def _open_batch_features(
    environment: CommonHierarchicalEnvironment,
    batch: OpenBatchState | None,
    totals: Mapping[str, float],
) -> list[float]:
    if batch is None:
        return [0.0] * len(PHASE2_OPEN_BATCH_FEATURE_NAMES)
    return [
        len(batch.job_ids) / environment.max_batch_wo_count,
        batch.length_sum / environment.max_batch_length_sum,
        batch.max_processing_time / totals["tact"],
        batch.cut_length_sum / totals["cut"],
        batch.bevel_quantity_sum / max(1.0, totals["bevel"]),
        (environment.max_batch_wo_count - len(batch.job_ids)) / environment.max_batch_wo_count,
        (environment.max_batch_length_sum - batch.length_sum) / environment.max_batch_length_sum,
    ]


def _resolve_open_batch(
    environment: CommonHierarchicalEnvironment,
    open_batch_id: str | None,
) -> OpenBatchState | None:
    if open_batch_id is None:
        return None
    if open_batch_id not in environment.state.runtime.open_batches:
        print(
            "[ERROR][Phase2.state._resolve_open_batch] "
            f"cause=unknown_open_batch batch_id={open_batch_id}"
        )
        raise RuntimeError(f"unknown open batch in Phase 2 state: {open_batch_id}")
    return environment.state.runtime.open_batches[open_batch_id]


def _job_totals(jobs: Mapping[str, object]) -> Dict[str, float]:
    if not jobs:
        print("[ERROR][Phase2.state._job_totals] cause=no_jobs")
        raise RuntimeError("Phase 2 state requires jobs")
    return {
        "wo": float(len(jobs)),
        "tact": sum(_processing_time(job, str(job_id)) for job_id, job in jobs.items()),
        "length": sum(_number_field(job, "plate_length", str(job_id)) for job_id, job in jobs.items()),
        "cut": sum(_number_field(job, "cut_length", str(job_id)) for job_id, job in jobs.items()),
        "bevel": sum(_number_field(job, "bevel_quantity", str(job_id)) for job_id, job in jobs.items()),
    }


def _job_allows_machine(job: object, machine_id: str) -> bool:
    allowed = tuple(str(value) for value in (_optional_field(job, "allowed_machine_ids") or ()))
    prohibited = tuple(str(value) for value in (_optional_field(job, "prohibited_machine_ids") or ()))
    return (not allowed or machine_id in allowed) and machine_id not in prohibited


def _processing_time(job: object, job_id: str) -> float:
    stages = _required_field(job, "base_stage_minutes", job_id)
    if not isinstance(stages, Mapping) or not stages:
        print(f"[ERROR][Phase2.state._processing_time] cause=invalid_stages job_id={job_id}")
        raise RuntimeError(f"invalid base_stage_minutes for {job_id}")
    try:
        total = sum(float(value) for value in stages.values())
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][Phase2.state._processing_time] "
            f"cause=non_numeric job_id={job_id} values={stages}"
        )
        raise RuntimeError(f"non-numeric base_stage_minutes for {job_id}") from exc
    if not math.isfinite(total) or total <= 0:
        print(f"[ERROR][Phase2.state._processing_time] cause=invalid job_id={job_id} value={total}")
        raise RuntimeError(f"invalid processing time for {job_id}")
    return total


def _number_field(job: object, name: str, job_id: str) -> float:
    raw_value = _required_field(job, name, job_id)
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        print(
            f"[ERROR][Phase2.state._number_field] cause=not_numeric "
            f"field={name} job_id={job_id} value={raw_value}"
        )
        raise RuntimeError(f"non-numeric {name} for {job_id}") from exc
    if not math.isfinite(value) or value < 0:
        print(f"[ERROR][Phase2.state._number_field] cause=invalid field={name} job_id={job_id} value={value}")
        raise RuntimeError(f"invalid {name} for {job_id}")
    return value


def _machine_load_number(load: Mapping[str, int | float], name: str, machine_id: str) -> float:
    """State schema의 필수 machine load를 기본값 없이 읽는다."""

    if name not in load:
        print(
            "[ERROR][Phase2.state._machine_load_number] "
            f"cause=missing_field machine_id={machine_id} field={name}"
        )
        raise RuntimeError(f"missing Phase 2 machine load field: {machine_id}.{name}")
    raw_value = load[name]
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][Phase2.state._machine_load_number] "
            f"cause=not_numeric machine_id={machine_id} field={name} value={raw_value}"
        )
        raise RuntimeError(f"non-numeric Phase 2 machine load: {machine_id}.{name}") from exc
    if not math.isfinite(value) or value < 0:
        print(
            "[ERROR][Phase2.state._machine_load_number] "
            f"cause=invalid_value machine_id={machine_id} field={name} value={raw_value}"
        )
        raise RuntimeError(f"invalid Phase 2 machine load: {machine_id}.{name}")
    return value


def _text_field(job: object, name: str, job_id: str) -> str:
    return str(_required_field(job, name, job_id))


def _required_field(value: object, name: str, key: str) -> Any:
    result = value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
    if result is None or (isinstance(result, str) and not result.strip()):
        print(f"[ERROR][Phase2.state._required_field] cause=missing field={name} key={key}")
        raise RuntimeError(f"missing {name} for {key}")
    return result


def _optional_field(value: object, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
