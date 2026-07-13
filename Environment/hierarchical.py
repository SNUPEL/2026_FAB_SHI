"""Phase 1 planning과 Phase 2 batch DES가 공유하는 계층 상태 소유자."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Any, Dict, Mapping

from Environment.constraints.profiles import (
    PhaseConstraintProfile,
    evaluate_phase2_action_constraints,
)
from Environment.events import (
    EVENT_CUT_BAY_ASSIGN,
    EVENT_MACHINE_ASSIGN,
    EVENT_PROCESS_FINISH,
    EVENT_PROCESS_START,
    FactoryEvent,
)


@dataclass
class HierarchicalPlanningState:
    block_to_bay: Dict[str, str] = field(default_factory=dict)
    bay_loads: Dict[str, Dict[str, int | float]] = field(default_factory=dict)
    phase1_completed: bool = False


@dataclass
class OpenBatchState:
    batch_id: str
    machine_id: str
    bay_id: str
    target_batch_size: int
    job_ids: list[str] = field(default_factory=list)
    length_sum: float = 0.0
    max_processing_time: float = 0.0
    processing_time_sum: float = 0.0
    cut_length_sum: float = 0.0
    bevel_quantity_sum: float = 0.0


@dataclass
class HierarchicalRuntimeState:
    machine_available_at: Dict[str, float]
    machine_loads: Dict[str, Dict[str, int | float]]
    open_batches: Dict[str, OpenBatchState] = field(default_factory=dict)
    completed_batches: list[Dict[str, Any]] = field(default_factory=list)
    scheduled_jobs: set[str] = field(default_factory=set)
    batch_seq: int = 0


@dataclass
class HierarchicalEventState:
    current_time: float = 0.0
    timeline: list[Dict[str, Any]] = field(default_factory=list)
    event_log: list[FactoryEvent] = field(default_factory=list)
    event_seq: int = 0


@dataclass
class HierarchicalState:
    planning: HierarchicalPlanningState
    runtime: HierarchicalRuntimeState
    events: HierarchicalEventState


@dataclass(frozen=True)
class HierarchicalSnapshot:
    state: HierarchicalState


@dataclass(frozen=True)
class Phase1View:
    jobs: Mapping[str, object]
    bay_ids: tuple[str, ...]
    bay_loads: Mapping[str, Mapping[str, int | float]]
    block_to_bay: Mapping[str, str]


@dataclass(frozen=True)
class Phase2BayView:
    bay_id: str
    jobs: Mapping[str, object]
    machines: Mapping[str, object]
    machine_available_at: Mapping[str, float]
    machine_loads: Mapping[str, Mapping[str, int | float]]


def create_phase1_planning_state(
    bay_capacity_weights: Mapping[str, int | float],
) -> HierarchicalPlanningState:
    """Standalone Phase 1과 full-flow가 공유할 planning state를 만든다."""

    if not bay_capacity_weights:
        print("[ERROR][Environment.hierarchical.create_phase1_planning_state] cause=no_bays")
        raise RuntimeError("Phase 1 planning state requires Bay capacity weights")
    normalized_weights: Dict[str, float] = {}
    for raw_bay_id, raw_weight in bay_capacity_weights.items():
        bay_id = str(raw_bay_id).strip()
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            print(
                "[ERROR][Environment.hierarchical.create_phase1_planning_state] "
                f"cause=non_numeric_capacity_weight bay_id={raw_bay_id} value={raw_weight}"
            )
            raise RuntimeError(f"Phase 1 Bay capacity weight is not numeric: {raw_bay_id}") from exc
        if not bay_id or not math.isfinite(weight) or weight <= 0:
            print(
                "[ERROR][Environment.hierarchical.create_phase1_planning_state] "
                f"cause=invalid_capacity_weight bay_id={raw_bay_id} value={raw_weight}"
            )
            raise RuntimeError("Phase 1 Bay capacity weights must be finite and positive")
        if bay_id in normalized_weights:
            print(
                "[ERROR][Environment.hierarchical.create_phase1_planning_state] "
                f"cause=duplicate_bay bay_id={bay_id}"
            )
            raise RuntimeError(f"duplicate Phase 1 Bay: {bay_id}")
        normalized_weights[bay_id] = weight
    return HierarchicalPlanningState(
        bay_loads={
            bay_id: {
                "block_count": 0,
                "wo_count": 0,
                "steel_quantity_sum": 0,
                "cut_length_sum": 0.0,
                "bevel_quantity_sum": 0,
                "long_cut_bay24_count": 0,
                "capacity_weight": normalized_weights[bay_id],
            }
            for bay_id in sorted(normalized_weights)
        }
    )


def apply_phase1_block_assignment(
    state: HierarchicalPlanningState,
    *,
    block_set_id: str,
    bay_id: str,
    steel_quantity_sum: int,
    cut_length_sum: float,
    bevel_quantity_sum: int,
    wo_count: int,
    long_cut_over_1000: int,
) -> None:
    """Block 하나의 Bay 배정과 누적 부하를 planning state에 원자적으로 반영한다."""

    normalized_block = str(block_set_id).strip()
    normalized_bay = str(bay_id).strip()
    if not normalized_block:
        print("[ERROR][Environment.hierarchical.apply_phase1_block_assignment] cause=empty_block_set_id")
        raise RuntimeError("Phase 1 block_set_id is required")
    if normalized_block in state.block_to_bay:
        print(
            "[ERROR][Environment.hierarchical.apply_phase1_block_assignment] "
            f"cause=duplicate_block block_set_id={normalized_block}"
        )
        raise RuntimeError(f"Phase 1 block already assigned: {normalized_block}")
    if normalized_bay not in state.bay_loads:
        print(
            "[ERROR][Environment.hierarchical.apply_phase1_block_assignment] "
            f"cause=unknown_bay block_set_id={normalized_block} bay_id={normalized_bay}"
        )
        raise RuntimeError(f"Phase 1 assignment references unknown Bay: {normalized_bay}")
    raw_values = {
        "steel_quantity_sum": steel_quantity_sum,
        "cut_length_sum": cut_length_sum,
        "bevel_quantity_sum": bevel_quantity_sum,
        "wo_count": wo_count,
        "long_cut_over_1000": long_cut_over_1000,
    }
    try:
        values = {key: float(value) for key, value in raw_values.items()}
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][Environment.hierarchical.apply_phase1_block_assignment] "
            f"cause=non_numeric_block_load block_set_id={normalized_block} values={raw_values}"
        )
        raise RuntimeError(f"Phase 1 block load is not numeric: {normalized_block}") from exc
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        print(
            "[ERROR][Environment.hierarchical.apply_phase1_block_assignment] "
            f"cause=invalid_block_load block_set_id={normalized_block} values={values}"
        )
        raise RuntimeError(f"Phase 1 block load must be finite and non-negative: {normalized_block}")
    discrete_fields = (
        "steel_quantity_sum",
        "bevel_quantity_sum",
        "wo_count",
        "long_cut_over_1000",
    )
    if any(
        isinstance(raw_values[field_name], bool) or not values[field_name].is_integer()
        for field_name in discrete_fields
    ):
        print(
            "[ERROR][Environment.hierarchical.apply_phase1_block_assignment] "
            f"cause=fractional_discrete_block_load block_set_id={normalized_block} values={raw_values}"
        )
        raise RuntimeError(f"Phase 1 discrete block load must be integer: {normalized_block}")
    if int(wo_count) <= 0 or int(steel_quantity_sum) <= 0 or int(long_cut_over_1000) not in {0, 1}:
        print(
            "[ERROR][Environment.hierarchical.apply_phase1_block_assignment] "
            f"cause=invalid_discrete_block_load block_set_id={normalized_block} values={values}"
        )
        raise RuntimeError(f"Phase 1 discrete block load is invalid: {normalized_block}")

    load = state.bay_loads[normalized_bay]
    load["steel_quantity_sum"] += int(steel_quantity_sum)
    load["cut_length_sum"] += float(cut_length_sum)
    load["bevel_quantity_sum"] += int(bevel_quantity_sum)
    load["long_cut_bay24_count"] += (
        1 if int(long_cut_over_1000) == 1 and normalized_bay == "24" else 0
    )
    load["wo_count"] += int(wo_count)
    load["block_count"] += 1
    state.block_to_bay[normalized_block] = normalized_bay


def complete_phase1_planning(
    state: HierarchicalPlanningState,
    expected_block_ids: set[str],
) -> None:
    """모든 block이 정확히 한 번 배정된 경우에만 Phase 1 완료를 표시한다."""

    normalized_expected = {str(value) for value in expected_block_ids}
    if set(state.block_to_bay) != normalized_expected:
        print(
            "[ERROR][Environment.hierarchical.complete_phase1_planning] "
            f"cause=assignment_mismatch missing={sorted(normalized_expected - set(state.block_to_bay))} "
            f"extra={sorted(set(state.block_to_bay) - normalized_expected)}"
        )
        raise RuntimeError("Phase 1 planning cannot complete with missing or extra blocks")
    state.phase1_completed = True


class CommonHierarchicalEnvironment:
    """Static scenario는 공유하고 planning/runtime/event state만 복제한다."""

    def __init__(
        self,
        *,
        jobs: Mapping[str, object],
        machines: Mapping[str, object],
        bay_capacity_weights: Mapping[str, int | float],
        constraint_profile: PhaseConstraintProfile,
        max_batch_wo_count: int,
        max_batch_length_sum: float,
    ) -> None:
        if not jobs or not machines:
            print("[ERROR][Environment.hierarchical.__init__] cause=empty_static_scenario")
            raise RuntimeError("hierarchical environment requires jobs and machines")
        if max_batch_wo_count <= 0 or max_batch_length_sum <= 0:
            print(
                "[ERROR][Environment.hierarchical.__init__] "
                f"cause=invalid_batch_limits count={max_batch_wo_count} length={max_batch_length_sum}"
            )
            raise RuntimeError("hierarchical environment batch limits must be positive")

        self.jobs = jobs
        self.machines = machines
        self.constraint_profile = constraint_profile
        self.max_batch_wo_count = int(max_batch_wo_count)
        self.max_batch_length_sum = float(max_batch_length_sum)
        self.machine_bay_ids = {
            str(machine_id): str(_required_field(machine, "bay_id", str(machine_id)))
            for machine_id, machine in machines.items()
        }
        self.bay_ids = tuple(sorted(set(self.machine_bay_ids.values())))
        if set(str(value) for value in bay_capacity_weights) != set(self.bay_ids):
            print(
                "[ERROR][Environment.hierarchical.__init__] "
                f"cause=capacity_weight_bay_mismatch expected={self.bay_ids} "
                f"actual={sorted(str(value) for value in bay_capacity_weights)}"
            )
            raise RuntimeError("Bay capacity weights do not match machine Bays")
        self.bay_capacity_weights = {str(key): float(value) for key, value in bay_capacity_weights.items()}
        self.state = self._new_state()

    def _new_state(self) -> HierarchicalState:
        planning = create_phase1_planning_state(self.bay_capacity_weights)
        return HierarchicalState(
            planning=planning,
            runtime=HierarchicalRuntimeState(
                machine_available_at={str(machine_id): 0.0 for machine_id in self.machines},
                machine_loads={
                    str(machine_id): {
                        "wo_count": 0,
                        "processing_time_sum": 0.0,
                        "cut_length_sum": 0.0,
                        "bevel_quantity_sum": 0,
                        "occupancy_time_sum": 0.0,
                        "batch_count": 0,
                    }
                    for machine_id in self.machines
                },
            ),
            events=HierarchicalEventState(),
        )

    def snapshot(self) -> HierarchicalSnapshot:
        return HierarchicalSnapshot(state=deepcopy(self.state))

    def restore(self, snapshot: HierarchicalSnapshot) -> None:
        if not isinstance(snapshot, HierarchicalSnapshot):
            print("[ERROR][Environment.hierarchical.restore] cause=invalid_snapshot")
            raise RuntimeError("invalid hierarchical environment snapshot")
        self.state = deepcopy(snapshot.state)

    def fork(self, snapshot: HierarchicalSnapshot | None = None) -> "CommonHierarchicalEnvironment":
        forked = CommonHierarchicalEnvironment(
            jobs=self.jobs,
            machines=self.machines,
            bay_capacity_weights=self.bay_capacity_weights,
            constraint_profile=self.constraint_profile,
            max_batch_wo_count=self.max_batch_wo_count,
            max_batch_length_sum=self.max_batch_length_sum,
        )
        forked.restore(snapshot or self.snapshot())
        return forked

    def phase1_view(self) -> Phase1View:
        return Phase1View(
            jobs=self.jobs,
            bay_ids=self.bay_ids,
            bay_loads=self.state.planning.bay_loads,
            block_to_bay=self.state.planning.block_to_bay,
        )

    def commit_phase1(
        self,
        *,
        block_to_bay: Mapping[str, str],
        bay_loads: Mapping[str, Mapping[str, int | float]],
    ) -> None:
        expected_blocks = {_block_set_id(job, str(job_id)) for job_id, job in self.jobs.items()}
        if set(block_to_bay) != expected_blocks:
            print(
                "[ERROR][Environment.hierarchical.commit_phase1] "
                f"cause=block_assignment_mismatch missing={sorted(expected_blocks - set(block_to_bay))} "
                f"extra={sorted(set(block_to_bay) - expected_blocks)}"
            )
            raise RuntimeError("Phase 1 block assignments do not match scenario blocks")
        if set(str(value) for value in bay_loads) != set(self.bay_ids):
            print("[ERROR][Environment.hierarchical.commit_phase1] cause=bay_load_mismatch")
            raise RuntimeError("Phase 1 Bay loads do not match environment Bays")
        invalid_bays = sorted({str(bay) for bay in block_to_bay.values()} - set(self.bay_ids))
        if invalid_bays:
            print(
                "[ERROR][Environment.hierarchical.commit_phase1] "
                f"cause=unknown_assigned_bay bay_ids={invalid_bays}"
            )
            raise RuntimeError(f"Phase 1 assigned unknown Bays: {invalid_bays}")

        normalized_loads = _validate_committed_phase1_loads(
            jobs=self.jobs,
            block_to_bay=block_to_bay,
            bay_loads=bay_loads,
            bay_capacity_weights=self.bay_capacity_weights,
        )
        self.state.planning.block_to_bay = {str(key): str(value) for key, value in block_to_bay.items()}
        self.state.planning.bay_loads = normalized_loads
        self.state.planning.phase1_completed = True

    def phase2_bay_view(self, bay_id: str) -> Phase2BayView:
        normalized_bay = str(bay_id)
        if not self.state.planning.phase1_completed:
            print("[ERROR][Environment.hierarchical.phase2_bay_view] cause=phase1_not_committed")
            raise RuntimeError("Phase 1 must be committed before Phase 2")
        if normalized_bay not in self.bay_ids:
            print(
                "[ERROR][Environment.hierarchical.phase2_bay_view] "
                f"cause=unknown_bay bay_id={normalized_bay}"
            )
            raise RuntimeError(f"unknown Phase 2 Bay: {normalized_bay}")
        jobs = {
            str(job_id): job
            for job_id, job in self.jobs.items()
            if self.state.planning.block_to_bay[_block_set_id(job, str(job_id))] == normalized_bay
            and str(job_id) not in self.state.runtime.scheduled_jobs
        }
        machines = {
            str(machine_id): machine
            for machine_id, machine in self.machines.items()
            if self.machine_bay_ids[str(machine_id)] == normalized_bay
        }
        return Phase2BayView(
            bay_id=normalized_bay,
            jobs=jobs,
            machines=machines,
            machine_available_at={key: self.state.runtime.machine_available_at[key] for key in machines},
            machine_loads={key: self.state.runtime.machine_loads[key] for key in machines},
        )

    def open_batch(self, machine_id: str, *, target_batch_size: int) -> str:
        normalized_machine = str(machine_id)
        if normalized_machine not in self.machines:
            print(
                "[ERROR][Environment.hierarchical.open_batch] "
                f"cause=unknown_machine machine_id={normalized_machine}"
            )
            raise RuntimeError(f"unknown machine: {normalized_machine}")
        if not 1 <= int(target_batch_size) <= self.max_batch_wo_count:
            print(
                "[ERROR][Environment.hierarchical.open_batch] "
                f"cause=invalid_target_batch_size value={target_batch_size}"
            )
            raise RuntimeError("target batch size is outside the confirmed limit")
        if any(batch.machine_id == normalized_machine for batch in self.state.runtime.open_batches.values()):
            print(
                "[ERROR][Environment.hierarchical.open_batch] "
                f"cause=machine_already_has_open_batch machine_id={normalized_machine}"
            )
            raise RuntimeError(f"machine already has an open batch: {normalized_machine}")
        self.state.runtime.batch_seq += 1
        batch_id = f"HB{self.state.runtime.batch_seq:06d}"
        self.state.runtime.open_batches[batch_id] = OpenBatchState(
            batch_id=batch_id,
            machine_id=normalized_machine,
            bay_id=self.machine_bay_ids[normalized_machine],
            target_batch_size=int(target_batch_size),
        )
        return batch_id

    def add_wo(self, batch_id: str, job_id: str) -> None:
        batch = self._open_batch(batch_id)
        normalized_job = str(job_id)
        if normalized_job not in self.jobs:
            print(f"[ERROR][Environment.hierarchical.add_wo] cause=unknown_job job_id={normalized_job}")
            raise RuntimeError(f"unknown job: {normalized_job}")
        if normalized_job in self.state.runtime.scheduled_jobs or any(
            normalized_job in current.job_ids for current in self.state.runtime.open_batches.values()
        ):
            print(f"[ERROR][Environment.hierarchical.add_wo] cause=duplicate_job job_id={normalized_job}")
            raise RuntimeError(f"job already selected or scheduled: {normalized_job}")
        if len(batch.job_ids) >= batch.target_batch_size:
            print(
                "[ERROR][Environment.hierarchical.add_wo] "
                f"cause=target_batch_full batch_id={batch_id} target={batch.target_batch_size}"
            )
            raise RuntimeError(f"target batch size already reached: {batch_id}")

        job = self.jobs[normalized_job]
        projected_ids = tuple([*batch.job_ids, normalized_job])
        projected_length = batch.length_sum + _non_negative_field(job, "plate_length", normalized_job)
        start_time = self.state.runtime.machine_available_at[batch.machine_id]
        result = evaluate_phase2_action_constraints(
            profile=self.constraint_profile,
            job=job,
            machine=self.machines[batch.machine_id],
            jobs=self.jobs,
            machines=self.machines,
            phase1_assignments=self.state.planning.block_to_bay,
            machine_available_at=self.state.runtime.machine_available_at,
            current_time=start_time,
            candidate_batch_job_ids=projected_ids,
            candidate_batch_length_sum=projected_length,
            max_wo_count=self.max_batch_wo_count,
            max_length_sum=self.max_batch_length_sum,
        )
        if not result.hard_passed:
            print(
                "[ERROR][Environment.hierarchical.add_wo] "
                f"cause=hard_constraint_failed batch_id={batch_id} job_id={normalized_job} "
                f"rules={result.hard_failed_rule_names} reasons={result.hard_reasons}"
            )
            raise RuntimeError(f"Phase 2 action violates hard constraints: {result.hard_failed_rule_names}")

        processing_time = _processing_time(job, normalized_job)
        batch.job_ids.append(normalized_job)
        batch.length_sum = projected_length
        batch.max_processing_time = max(batch.max_processing_time, processing_time)
        batch.processing_time_sum += processing_time
        batch.cut_length_sum += _non_negative_field(job, "cut_length", normalized_job)
        batch.bevel_quantity_sum += _non_negative_field(job, "bevel_quantity", normalized_job)

    def close_batch(self, batch_id: str) -> Dict[str, Any]:
        batch = self._open_batch(batch_id)
        if not batch.job_ids:
            print(f"[ERROR][Environment.hierarchical.close_batch] cause=empty_batch batch_id={batch_id}")
            raise RuntimeError(f"cannot close empty batch: {batch_id}")

        start_time = float(self.state.runtime.machine_available_at[batch.machine_id])
        finish_time = start_time + float(batch.max_processing_time)
        row = {
            "batch_id": batch.batch_id,
            "machine_id": batch.machine_id,
            "bay_id": batch.bay_id,
            "job_ids": tuple(batch.job_ids),
            "wo_count": len(batch.job_ids),
            "length_sum": round(batch.length_sum, 6),
            "batch_duration": round(batch.max_processing_time, 6),
            "duration": round(batch.max_processing_time, 6),
            "job_processing_time_sum": round(batch.processing_time_sum, 6),
            "cut_length_sum": round(batch.cut_length_sum, 6),
            "bevel_quantity_sum": round(batch.bevel_quantity_sum, 6),
            "start_time": round(start_time, 6),
            "finish_time": round(finish_time, 6),
            "duration_rule": "max_processing_time",
        }
        load = self.state.runtime.machine_loads[batch.machine_id]
        load["wo_count"] += len(batch.job_ids)
        load["processing_time_sum"] += batch.processing_time_sum
        load["cut_length_sum"] += batch.cut_length_sum
        load["bevel_quantity_sum"] += batch.bevel_quantity_sum
        load["occupancy_time_sum"] += batch.max_processing_time
        load["batch_count"] += 1
        self.state.runtime.machine_available_at[batch.machine_id] = finish_time
        self.state.runtime.scheduled_jobs.update(batch.job_ids)
        self.state.runtime.completed_batches.append(dict(row))
        self.state.events.timeline.append(dict(row))
        # 여러 설비의 batch를 계획 순서대로 넣어도 공통 DES horizon은 뒤로 가지 않는다.
        self.state.events.current_time = max(self.state.events.current_time, finish_time)
        for job_id in batch.job_ids:
            self._append_job_events(batch, job_id, start_time, finish_time)
        del self.state.runtime.open_batches[batch_id]
        return row

    def _append_job_events(self, batch: OpenBatchState, job_id: str, start_time: float, finish_time: float) -> None:
        operation_id = f"{batch.batch_id}:{job_id}"
        payload = {"batch_id": batch.batch_id, "batch_job_ids": tuple(batch.job_ids)}
        for event_type, event_time in (
            (EVENT_MACHINE_ASSIGN, start_time),
            (EVENT_CUT_BAY_ASSIGN, start_time),
            (EVENT_PROCESS_START, start_time),
            (EVENT_PROCESS_FINISH, finish_time),
        ):
            self.state.events.event_seq += 1
            self.state.events.event_log.append(
                FactoryEvent(
                    event_id=f"HE{self.state.events.event_seq:08d}",
                    event_type=event_type,
                    time_min=event_time,
                    job_id=job_id,
                    machine_id=batch.machine_id,
                    bay_id=batch.bay_id,
                    operation_id=operation_id,
                    payload=dict(payload),
                )
            )

    def _open_batch(self, batch_id: str) -> OpenBatchState:
        if batch_id not in self.state.runtime.open_batches:
            print(f"[ERROR][Environment.hierarchical._open_batch] cause=unknown_batch batch_id={batch_id}")
            raise RuntimeError(f"unknown open batch: {batch_id}")
        return self.state.runtime.open_batches[batch_id]


def _required_field(value: object, field_name: str, key: str) -> Any:
    if isinstance(value, Mapping):
        present = field_name in value
        result = value.get(field_name)
    else:
        present = hasattr(value, field_name)
        result = getattr(value, field_name, None)
    if not present or result is None or (isinstance(result, str) and not result.strip()):
        print(
            "[ERROR][Environment.hierarchical._required_field] "
            f"cause=missing_field field={field_name} key={key}"
        )
        raise RuntimeError(f"missing {field_name} for {key}")
    return result


def _validate_committed_phase1_loads(
    *,
    jobs: Mapping[str, object],
    block_to_bay: Mapping[str, str],
    bay_loads: Mapping[str, Mapping[str, int | float]],
    bay_capacity_weights: Mapping[str, float],
) -> Dict[str, Dict[str, int | float]]:
    """Phase 1이 넘긴 권위 load가 assignment와 모순되지 않는지 검사한다."""

    expected_block_counts = {bay_id: 0 for bay_id in bay_capacity_weights}
    for bay_id in block_to_bay.values():
        expected_block_counts[str(bay_id)] += 1
    expected_wo_counts = {bay_id: 0 for bay_id in bay_capacity_weights}
    for job_id, job in jobs.items():
        block_id = _block_set_id(job, str(job_id))
        bay_id = str(block_to_bay[block_id])
        expected_wo_counts[bay_id] += 1

    normalized: Dict[str, Dict[str, int | float]] = {}
    integer_fields = (
        "block_count",
        "wo_count",
        "steel_quantity_sum",
        "bevel_quantity_sum",
        "long_cut_bay24_count",
    )
    for bay_id in sorted(bay_capacity_weights):
        row = bay_loads[bay_id]
        values: Dict[str, int | float] = {}
        for field_name in integer_fields:
            values[field_name] = _required_non_negative_integer(row, field_name, bay_id)
        values["cut_length_sum"] = _required_non_negative_number(row, "cut_length_sum", bay_id)
        values["capacity_weight"] = _required_positive_number(row, "capacity_weight", bay_id)
        if values["block_count"] != expected_block_counts[bay_id]:
            print(
                "[ERROR][Environment.hierarchical._validate_committed_phase1_loads] "
                f"cause=block_count_mismatch bay_id={bay_id} "
                f"actual={values['block_count']} expected={expected_block_counts[bay_id]}"
            )
            raise RuntimeError(f"Phase 1 block_count mismatch for Bay {bay_id}")
        if values["wo_count"] != expected_wo_counts[bay_id]:
            print(
                "[ERROR][Environment.hierarchical._validate_committed_phase1_loads] "
                f"cause=wo_count_mismatch bay_id={bay_id} "
                f"actual={values['wo_count']} expected={expected_wo_counts[bay_id]}"
            )
            raise RuntimeError(f"Phase 1 wo_count mismatch for Bay {bay_id}")
        if not math.isclose(
            float(values["capacity_weight"]),
            float(bay_capacity_weights[bay_id]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            print(
                "[ERROR][Environment.hierarchical._validate_committed_phase1_loads] "
                f"cause=capacity_weight_mismatch bay_id={bay_id} "
                f"actual={values['capacity_weight']} expected={bay_capacity_weights[bay_id]}"
            )
            raise RuntimeError(f"Phase 1 capacity_weight mismatch for Bay {bay_id}")
        if int(values["long_cut_bay24_count"]) > int(values["block_count"]):
            print(
                "[ERROR][Environment.hierarchical._validate_committed_phase1_loads] "
                f"cause=invalid_long_cut_count bay_id={bay_id} "
                f"long_cut={values['long_cut_bay24_count']} blocks={values['block_count']}"
            )
            raise RuntimeError(f"Phase 1 long-cut count exceeds block count for Bay {bay_id}")
        normalized[bay_id] = values
    return normalized


def _required_non_negative_integer(
    row: Mapping[str, int | float], field_name: str, bay_id: str
) -> int:
    number = _required_non_negative_number(row, field_name, bay_id)
    if isinstance(row[field_name], bool) or not float(number).is_integer():
        print(
            "[ERROR][Environment.hierarchical._required_non_negative_integer] "
            f"cause=not_integer bay_id={bay_id} field={field_name} value={row[field_name]}"
        )
        raise RuntimeError(f"Phase 1 Bay load must be an integer: {bay_id}.{field_name}")
    return int(number)


def _required_non_negative_number(
    row: Mapping[str, int | float], field_name: str, bay_id: str
) -> float:
    if field_name not in row:
        print(
            "[ERROR][Environment.hierarchical._required_non_negative_number] "
            f"cause=missing_field bay_id={bay_id} field={field_name}"
        )
        raise RuntimeError(f"Phase 1 Bay load is missing: {bay_id}.{field_name}")
    try:
        number = float(row[field_name])
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][Environment.hierarchical._required_non_negative_number] "
            f"cause=not_numeric bay_id={bay_id} field={field_name} value={row[field_name]}"
        )
        raise RuntimeError(f"Phase 1 Bay load is not numeric: {bay_id}.{field_name}") from exc
    if not math.isfinite(number) or number < 0:
        print(
            "[ERROR][Environment.hierarchical._required_non_negative_number] "
            f"cause=invalid_value bay_id={bay_id} field={field_name} value={row[field_name]}"
        )
        raise RuntimeError(f"Phase 1 Bay load must be finite and non-negative: {bay_id}.{field_name}")
    return number


def _required_positive_number(
    row: Mapping[str, int | float], field_name: str, bay_id: str
) -> float:
    number = _required_non_negative_number(row, field_name, bay_id)
    if number <= 0:
        print(
            "[ERROR][Environment.hierarchical._required_positive_number] "
            f"cause=non_positive bay_id={bay_id} field={field_name} value={number}"
        )
        raise RuntimeError(f"Phase 1 Bay load must be positive: {bay_id}.{field_name}")
    return number


def _block_set_id(job: object, job_id: str) -> str:
    return str(_required_field(job, "block_set_id", job_id))


def _non_negative_field(job: object, field_name: str, job_id: str) -> float:
    raw_value = _required_field(job, field_name, job_id)
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][Environment.hierarchical._non_negative_field] "
            f"cause=not_numeric field={field_name} job_id={job_id} value={raw_value}"
        )
        raise RuntimeError(f"non-numeric {field_name} for {job_id}") from exc
    if not math.isfinite(value) or value < 0:
        print(
            "[ERROR][Environment.hierarchical._non_negative_field] "
            f"cause=invalid_value field={field_name} job_id={job_id} value={value}"
        )
        raise RuntimeError(f"invalid {field_name} for {job_id}")
    return value


def _processing_time(job: object, job_id: str) -> float:
    stages = _required_field(job, "base_stage_minutes", job_id)
    if not isinstance(stages, Mapping) or not stages:
        print(
            "[ERROR][Environment.hierarchical._processing_time] "
            f"cause=invalid_stage_minutes job_id={job_id}"
        )
        raise RuntimeError(f"invalid base_stage_minutes for {job_id}")
    try:
        total = sum(float(value) for value in stages.values())
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][Environment.hierarchical._processing_time] "
            f"cause=non_numeric_stage_minutes job_id={job_id} values={stages}"
        )
        raise RuntimeError(f"non-numeric base_stage_minutes for {job_id}") from exc
    if not math.isfinite(total) or total <= 0:
        print(
            "[ERROR][Environment.hierarchical._processing_time] "
            f"cause=invalid_processing_time job_id={job_id} value={total}"
        )
        raise RuntimeError(f"invalid processing time for {job_id}")
    return total
