"""Gymnasium 연결 전용 adapter와 검증 도구.

이 모듈의 핵심 목적은 DES core를 새로 만들지 않고,
`CuttingSimulation.get_candidates()`가 반환하는 hard-feasible action 후보를
그대로 학습용 action mask로 노출하는 것이다.

구조:
- `DESActionMaskAdapter`
  Gymnasium 설치 여부와 무관하게 동작한다. 현재 DES state에서
  action 후보, mask, action_id table, candidate feature matrix를 만든다.

- `HierarchicalDESActionAdapter`
  DES 내부 후보(`open_batch`, `add_to_batch`, `close_batch`)를
  RL이 이해하기 쉬운 `SELECT_MACHINE -> SELECT_WO -> COMMIT` 단계로
  다시 노출한다. hard-feasible 판정은 새로 만들지 않고 기존
  `CuttingSimulation.get_candidates()` 결과만 사용한다.

- `PMSPGymnasiumWrapper`
  실제 Gymnasium `Env` 형태의 얇은 wrapper다. 외부 `gymnasium` 패키지가
  없으면 조용히 대체하지 않고 원인을 출력한 뒤 실패한다.

- `run_wrapper_equivalence()`
  `batch_fill_spt` 같은 기존 휴리스틱을 wrapper action index로 재현해서
  기존 DES 직접 실행과 결과가 같은지 확인한다.

예시:
```python
env = CuttingShopEnvironment(config=config, scenario=scenario)
wrapper = PMSPGymnasiumWrapper(env, max_actions=4096)
obs, info = wrapper.reset()
mask = wrapper.action_masks()
action_index = int(mask.nonzero()[0][0])
obs, reward, terminated, truncated, info = wrapper.step(action_index)
```
"""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from Agent.heuristics import select_action_by_rule
from Environment.environment import CuttingShopEnvironment
from Environment.reward import compute_load_imbalance
from Environment.simulation import ActionCandidate

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError as exc:
    gym = None
    spaces = None
    GYMNASIUM_AVAILABLE = False
    _GYMNASIUM_IMPORT_ERROR = exc
else:
    _GYMNASIUM_IMPORT_ERROR = None


CANDIDATE_FEATURE_NAMES: Tuple[str, ...] = (
    "action_kind_code",
    "estimated_minutes",
    "processing_minutes",
    "soft_penalty",
    "priority_weight",
    "batch_wo_count",
    "batch_length_sum_ratio",
    "job_plate_length_ratio",
    "machine_load_ratio",
    "machine_available_delay_ratio",
    "remaining_job_ratio_after_action",
)

ACTION_KIND_TO_CODE: Dict[str, float] = {
    "open_batch": 0.0,
    "add_to_batch": 1.0,
    "close_batch": 2.0,
    "single_dispatch": 3.0,
}

HIERARCHICAL_PHASE_SELECT_MACHINE = "SELECT_MACHINE"
HIERARCHICAL_PHASE_SELECT_WO = "SELECT_WO"

HIERARCHICAL_ACTION_KIND_TO_CODE: Dict[str, float] = {
    "select_machine": 0.0,
    "select_wo": 1.0,
    "commit_batch": 2.0,
}

HIERARCHICAL_FEATURE_NAMES: Tuple[str, ...] = (
    "hierarchical_action_kind_code",
    "internal_action_kind_code",
    "candidate_count_for_machine",
    "batch_wo_count",
    "batch_length_sum_ratio",
    "estimated_minutes_ratio",
    "machine_load_ratio",
    "remaining_job_ratio",
)


@dataclass
class ActionMaskSnapshot:
    """한 decision epoch에서 학습기가 봐야 하는 action mask 상태."""

    candidates: List[ActionCandidate]
    action_id_table: List[Dict]
    action_mask: np.ndarray
    candidate_features: np.ndarray


@dataclass
class HierarchicalActionMaskSnapshot:
    """계층형 action adapter가 한 decision epoch에서 노출하는 상태.

    `candidates`는 현재 DES core가 반환한 원본 hard-feasible 후보 목록이다.
    `action_id_table`은 RL이 선택하는 계층형 action index 표다.
    """

    phase: str
    selected_machine_id: Optional[str]
    selected_batch_id: Optional[str]
    current_batch_job_ids: Tuple[str, ...]
    candidates: List[ActionCandidate]
    action_id_table: List[Dict]
    action_mask: np.ndarray
    action_features: np.ndarray


def require_gymnasium() -> None:
    """Gymnasium 패키지가 실제로 사용 가능한지 확인한다."""

    if GYMNASIUM_AVAILABLE:
        return
    print(
        "[ERROR][gym_wrapper.require_gymnasium] "
        "cause=gymnasium_not_installed "
        f"error={_GYMNASIUM_IMPORT_ERROR} "
        "fix=install gymnasium before constructing PMSPGymnasiumWrapper"
    )
    raise RuntimeError("gymnasium is required for PMSPGymnasiumWrapper") from _GYMNASIUM_IMPORT_ERROR


def _action_kind_code(action_kind: str) -> float:
    """action kind를 고정 numeric code로 바꾼다."""

    if action_kind not in ACTION_KIND_TO_CODE:
        print(
            "[ERROR][gym_wrapper._action_kind_code] "
            f"cause=unknown_action_kind action_kind={action_kind}"
        )
        raise KeyError(f"unknown action_kind: {action_kind}")
    return ACTION_KIND_TO_CODE[action_kind]


def _candidate_to_table_row(candidate: ActionCandidate, index: int) -> Dict:
    """candidate 객체 1개를 사람이 읽을 수 있는 action table row로 바꾼다."""

    return {
        "index": index,
        "action_id": candidate.action_id,
        "action_kind": candidate.action_kind,
        "job_id": candidate.job.job_id,
        "machine_id": candidate.machine.machine_id,
        "machine_bay_id": candidate.machine.bay_id or "",
        "batch_id": candidate.batch_id or "",
        "batch_job_ids": list(candidate.batch_job_ids),
        "batch_wo_count": int(candidate.batch_wo_count),
        "batch_length_sum": float(candidate.batch_length_sum),
        "estimated_minutes": float(candidate.estimated_minutes),
        "processing_minutes": float(candidate.processing_minutes),
        "soft_penalty": float(candidate.soft_penalty),
    }


def _candidate_feature_vector(
    candidate: ActionCandidate,
    env: CuttingShopEnvironment,
) -> List[float]:
    """candidate 객체 1개를 고정 길이 숫자 feature로 바꾼다.

    이 feature는 Gymnasium wrapper smoke용 최소 관측값이다.
    기존 `Train/*` 네트워크는 아직 core observation의 `available_actions`를
    직접 쓰므로, 본 feature는 신규 Gym 실험을 위한 별도 경로다.
    """

    simulation = env.simulation
    horizon = max(float(simulation.config["simulation"]["horizon_minutes"]), 1.0)
    total_jobs = max(len(simulation.jobs), 1)
    remaining_jobs = len(simulation.state.unscheduled_jobs)
    machine_id = candidate.machine.machine_id
    machine_load = float(simulation.state.machine_loads.get(machine_id, 0.0))
    machine_available_at = float(simulation.state.machine_available_at.get(machine_id, 0.0))
    machine_delay = max(0.0, machine_available_at - float(simulation.state.current_time))
    max_batch_length = candidate.machine.max_batch_length_sum
    if max_batch_length is None:
        max_batch_length = simulation._get_machine_day_length_sum_limit(machine_id)
    if max_batch_length is None:
        max_batch_length = candidate.machine.table_length_limit
    if max_batch_length is None:
        print(
            "[ERROR][gym_wrapper._candidate_feature_vector] "
            "cause=missing_batch_length_limit "
            f"machine_id={machine_id} action_id={candidate.action_id}"
        )
        raise RuntimeError(f"missing batch length limit for machine_id={machine_id}")

    return [
        _action_kind_code(candidate.action_kind),
        float(candidate.estimated_minutes) / horizon,
        float(candidate.processing_minutes) / horizon,
        float(candidate.soft_penalty),
        float(candidate.job.priority_weight),
        float(candidate.batch_wo_count),
        float(candidate.batch_length_sum) / max(float(max_batch_length), 1.0),
        float(candidate.job.plate_length) / max(float(max_batch_length), 1.0),
        machine_load / horizon,
        machine_delay / horizon,
        max(float(remaining_jobs - candidate.batch_wo_count), 0.0) / total_jobs,
    ]


def _machine_batch_length_limit(env: CuttingShopEnvironment, machine_id: str) -> float:
    """계층형 feature 정규화에 쓸 machine별 batch 길이 한계를 읽는다."""

    if machine_id not in env.simulation.machines:
        print(
            "[ERROR][gym_wrapper._machine_batch_length_limit] "
            f"cause=unknown_machine machine_id={machine_id}"
        )
        raise KeyError(f"unknown machine_id: {machine_id}")

    machine = env.simulation.machines[machine_id]
    limit = machine.max_batch_length_sum
    if limit is None:
        limit = env.simulation._get_machine_day_length_sum_limit(machine_id)
    if limit is None:
        limit = machine.table_length_limit
    if limit in (None, ""):
        print(
            "[ERROR][gym_wrapper._machine_batch_length_limit] "
            f"cause=missing_batch_length_limit machine_id={machine_id}"
        )
        raise RuntimeError(f"missing batch length limit for machine_id={machine_id}")
    return float(limit)


def _hierarchical_action_kind_code(action_kind: str) -> float:
    """계층형 action kind를 고정 numeric code로 바꾼다."""

    if action_kind not in HIERARCHICAL_ACTION_KIND_TO_CODE:
        print(
            "[ERROR][gym_wrapper._hierarchical_action_kind_code] "
            f"cause=unknown_hierarchical_action_kind action_kind={action_kind}"
        )
        raise KeyError(f"unknown hierarchical action_kind: {action_kind}")
    return HIERARCHICAL_ACTION_KIND_TO_CODE[action_kind]


def _hierarchical_feature_vector(row: Dict, env: CuttingShopEnvironment) -> List[float]:
    """계층형 action table row 1개를 고정 길이 숫자 feature로 바꾼다."""

    simulation = env.simulation
    horizon = max(float(simulation.config["simulation"]["horizon_minutes"]), 1.0)
    total_jobs = max(len(simulation.jobs), 1)
    remaining_jobs = len(simulation.state.unscheduled_jobs)
    machine_id = str(row.get("machine_id") or "")
    machine_load = float(simulation.state.machine_loads.get(machine_id, 0.0)) if machine_id else 0.0
    max_batch_length = _machine_batch_length_limit(env, machine_id) if machine_id else 1.0
    internal_action_kind = str(row.get("internal_action_kind") or "")
    internal_kind_code = _action_kind_code(internal_action_kind) if internal_action_kind else -1.0

    return [
        _hierarchical_action_kind_code(str(row["action_kind"])),
        internal_kind_code,
        float(row.get("candidate_count_for_machine", 0.0)),
        float(row.get("batch_wo_count", 0.0)),
        float(row.get("batch_length_sum", 0.0)) / max(max_batch_length, 1.0),
        float(row.get("estimated_minutes", 0.0)) / horizon,
        machine_load / horizon,
        float(remaining_jobs) / total_jobs,
    ]


def _global_feature_vector(env: CuttingShopEnvironment) -> np.ndarray:
    """현재 DES state를 고정 길이 global feature로 바꾼다."""

    simulation = env.simulation
    horizon = max(float(simulation.config["simulation"]["horizon_minutes"]), 1.0)
    total_jobs = max(len(simulation.jobs), 1)
    machine_count = max(len(simulation.machines), 1)
    machine_loads = simulation.state.machine_loads
    return np.asarray(
        [
            float(simulation.state.current_time) / horizon,
            float(simulation.get_makespan()) / horizon,
            len(simulation.state.unscheduled_jobs) / total_jobs,
            len(simulation.state.open_batches) / machine_count,
            float(compute_load_imbalance(machine_loads)) / horizon,
            sum(1 for machine_id in simulation.machines if simulation._is_machine_available(machine_id))
            / machine_count,
        ],
        dtype=np.float32,
    )


def _hierarchical_phase_code(phase: str) -> float:
    """계층형 phase를 observation용 숫자 code로 바꾼다."""

    if phase == HIERARCHICAL_PHASE_SELECT_MACHINE:
        return 0.0
    if phase == HIERARCHICAL_PHASE_SELECT_WO:
        return 1.0
    print(
        "[ERROR][gym_wrapper._hierarchical_phase_code] "
        f"cause=unknown_hierarchical_phase phase={phase}"
    )
    raise KeyError(f"unknown hierarchical phase: {phase}")


class DESActionMaskAdapter:
    """DES candidate list를 Gym/RL용 mask와 action table로 변환한다.

    이 class는 외부 `gymnasium` 패키지를 요구하지 않는다.
    따라서 CI나 현재 개발 PC에 Gymnasium이 없어도 action mask 동일성 검증은 가능하다.
    """

    def __init__(self, env: CuttingShopEnvironment, max_actions: int):
        if max_actions <= 0:
            print(
                "[ERROR][DESActionMaskAdapter.__init__] "
                f"cause=invalid_max_actions max_actions={max_actions}"
            )
            raise ValueError("max_actions must be positive")
        self.env = env
        self.max_actions = int(max_actions)
        self._snapshot: Optional[ActionMaskSnapshot] = None

    @property
    def snapshot(self) -> ActionMaskSnapshot:
        """가장 최근 action mask snapshot을 반환한다."""

        if self._snapshot is None:
            print("[ERROR][DESActionMaskAdapter.snapshot] cause=snapshot_not_initialized")
            raise RuntimeError("action mask snapshot is not initialized")
        return self._snapshot

    def refresh(self) -> ActionMaskSnapshot:
        """현재 DES state에서 candidate, mask, table, feature를 새로 만든다."""

        candidates = self.env.get_action_candidates()
        candidate_count = len(candidates)
        if candidate_count > self.max_actions:
            print(
                "[ERROR][DESActionMaskAdapter.refresh] "
                "cause=candidate_count_exceeds_max_actions "
                f"candidate_count={candidate_count} max_actions={self.max_actions}"
            )
            raise RuntimeError("candidate count exceeds max_actions")

        action_ids = [candidate.action_id for candidate in candidates]
        duplicate_count = len(action_ids) - len(set(action_ids))
        if duplicate_count:
            print(
                "[ERROR][DESActionMaskAdapter.refresh] "
                f"cause=duplicate_action_id duplicate_count={duplicate_count}"
            )
            raise RuntimeError("duplicate action_id detected in candidates")

        action_mask = np.zeros(self.max_actions, dtype=np.bool_)
        action_mask[:candidate_count] = True
        candidate_features = np.zeros((self.max_actions, len(CANDIDATE_FEATURE_NAMES)), dtype=np.float32)
        action_id_table: List[Dict] = []
        for index, candidate in enumerate(candidates):
            action_id_table.append(_candidate_to_table_row(candidate, index))
            candidate_features[index, :] = np.asarray(
                _candidate_feature_vector(candidate, self.env),
                dtype=np.float32,
            )

        self._snapshot = ActionMaskSnapshot(
            candidates=candidates,
            action_id_table=action_id_table,
            action_mask=action_mask,
            candidate_features=candidate_features,
        )
        return self._snapshot

    def action_masks(self) -> np.ndarray:
        """학습 라이브러리에서 흔히 기대하는 bool mask를 반환한다."""

        return self.snapshot.action_mask.copy()

    def current_action_table(self) -> List[Dict]:
        """현재 index -> action_id mapping table을 반환한다."""

        return copy.deepcopy(self.snapshot.action_id_table)

    def select_candidate_by_index(self, action_index: int) -> ActionCandidate:
        """wrapper가 받은 action index를 DES `ActionCandidate`로 되돌린다."""

        if not isinstance(action_index, (int, np.integer)):
            print(
                "[ERROR][DESActionMaskAdapter.select_candidate_by_index] "
                f"cause=non_integer_action_index action_index={action_index}"
            )
            raise TypeError("action_index must be an integer")
        if action_index < 0 or action_index >= self.max_actions:
            print(
                "[ERROR][DESActionMaskAdapter.select_candidate_by_index] "
                f"cause=action_index_out_of_range action_index={action_index} max_actions={self.max_actions}"
            )
            raise IndexError("action_index out of range")
        if not bool(self.snapshot.action_mask[int(action_index)]):
            print(
                "[ERROR][DESActionMaskAdapter.select_candidate_by_index] "
                "cause=masked_or_padded_action_selected "
                f"action_index={action_index} available_action_count={len(self.snapshot.candidates)}"
            )
            raise RuntimeError("selected action index is masked/padded")
        return self.snapshot.candidates[int(action_index)]


class HierarchicalDESActionAdapter:
    """DES batch 후보를 `SELECT_MACHINE -> SELECT_WO -> COMMIT` 형태로 노출한다.

    설계 기준:
    - hard 제약은 이 class에서 새로 계산하지 않는다.
    - 항상 `CuttingSimulation.get_candidates()`가 만든 후보만 사용한다.
    - `select_machine`은 학습용 context 선택이므로 DES state를 바꾸지 않는다.
    - `select_wo`와 `commit_batch`만 DES `step_candidate()`를 호출한다.

    예시:
    ```python
    adapter = HierarchicalDESActionAdapter(env, max_actions=4096)
    adapter.refresh()
    adapter.step(machine_index)  # SELECT_MACHINE, DES state 불변
    adapter.step(job_index)      # SELECT_WO, open/add batch 실행
    adapter.step(commit_index)   # COMMIT, close batch 실행
    ```
    """

    def __init__(self, env: CuttingShopEnvironment, max_actions: int):
        if max_actions <= 0:
            print(
                "[ERROR][HierarchicalDESActionAdapter.__init__] "
                f"cause=invalid_max_actions max_actions={max_actions}"
            )
            raise ValueError("max_actions must be positive")
        self.env = env
        self.max_actions = int(max_actions)
        self.selected_machine_id: Optional[str] = None
        self._snapshot: Optional[HierarchicalActionMaskSnapshot] = None

    @property
    def snapshot(self) -> HierarchicalActionMaskSnapshot:
        """가장 최근 계층형 action mask snapshot을 반환한다."""

        if self._snapshot is None:
            print("[ERROR][HierarchicalDESActionAdapter.snapshot] cause=snapshot_not_initialized")
            raise RuntimeError("hierarchical action mask snapshot is not initialized")
        return self._snapshot

    def refresh(self) -> HierarchicalActionMaskSnapshot:
        """현재 DES state에서 계층형 action mask를 새로 만든다."""

        candidates = self.env.get_action_candidates()
        if self.selected_machine_id is None:
            rows = self._build_select_machine_rows(candidates)
            phase = HIERARCHICAL_PHASE_SELECT_MACHINE
            selected_batch_id = None
            current_batch_job_ids: Tuple[str, ...] = ()
        else:
            rows = self._build_selected_machine_rows(candidates, self.selected_machine_id)
            if not rows:
                print(
                    "[ERROR][HierarchicalDESActionAdapter.refresh] "
                    "cause=selected_machine_has_no_feasible_rows "
                    f"selected_machine_id={self.selected_machine_id} "
                    f"candidate_count={len(candidates)}"
                )
                raise RuntimeError(
                    f"selected machine has no feasible hierarchical rows: {self.selected_machine_id}"
                )
            else:
                phase = HIERARCHICAL_PHASE_SELECT_WO
                selected_batch_id, current_batch_job_ids = self._open_batch_summary_for_machine(
                    self.selected_machine_id
                )

        if len(rows) > self.max_actions:
            print(
                "[ERROR][HierarchicalDESActionAdapter.refresh] "
                "cause=hierarchical_action_count_exceeds_max_actions "
                f"action_count={len(rows)} max_actions={self.max_actions} phase={phase}"
            )
            raise RuntimeError("hierarchical action count exceeds max_actions")

        action_ids = [row["action_id"] for row in rows]
        duplicate_count = len(action_ids) - len(set(action_ids))
        if duplicate_count:
            print(
                "[ERROR][HierarchicalDESActionAdapter.refresh] "
                f"cause=duplicate_hierarchical_action_id duplicate_count={duplicate_count}"
            )
            raise RuntimeError("duplicate hierarchical action_id detected")

        action_mask = np.zeros(self.max_actions, dtype=np.bool_)
        action_mask[: len(rows)] = True
        action_features = np.zeros((self.max_actions, len(HIERARCHICAL_FEATURE_NAMES)), dtype=np.float32)
        for index, row in enumerate(rows):
            row["index"] = index
            action_features[index, :] = np.asarray(
                _hierarchical_feature_vector(row, self.env),
                dtype=np.float32,
            )

        self._snapshot = HierarchicalActionMaskSnapshot(
            phase=phase,
            selected_machine_id=self.selected_machine_id,
            selected_batch_id=selected_batch_id,
            current_batch_job_ids=current_batch_job_ids,
            candidates=candidates,
            action_id_table=rows,
            action_mask=action_mask,
            action_features=action_features,
        )
        return self._snapshot

    def action_masks(self) -> np.ndarray:
        """학습 라이브러리에서 기대하는 bool mask를 반환한다."""

        return self.snapshot.action_mask.copy()

    def current_action_table(self) -> List[Dict]:
        """현재 계층형 index -> action 의미 table을 반환한다."""

        return copy.deepcopy(self.snapshot.action_id_table)

    def step(self, action_index: int) -> Tuple[HierarchicalActionMaskSnapshot, float, bool, Dict]:
        """계층형 action index를 선택해 adapter 또는 DES state를 한 단계 전진한다."""

        row = self._select_row_by_index(action_index)
        action_kind = str(row["action_kind"])
        if action_kind == "select_machine":
            self.selected_machine_id = str(row["machine_id"])
            self.refresh()
            return self.snapshot, 0.0, False, {
                "hierarchical_action_kind": action_kind,
                "phase": self.snapshot.phase,
                "selected_machine_id": self.selected_machine_id,
                "des_state_mutated": False,
            }

        if action_kind not in {"select_wo", "commit_batch"}:
            print(
                "[ERROR][HierarchicalDESActionAdapter.step] "
                f"cause=unsupported_hierarchical_action_kind action_kind={action_kind}"
            )
            raise ValueError(f"unsupported hierarchical action_kind: {action_kind}")

        internal_action_id = str(row.get("internal_action_id") or "")
        selected_candidate = self._candidate_by_action_id(internal_action_id)
        _, reward, terminated, truncated, step_info = self.env.simulation.step_candidate(
            selected_candidate,
            build_observation_result=False,
            advance_decision_epoch=True,
        )
        if action_kind == "commit_batch" or terminated or truncated:
            self.selected_machine_id = None
        else:
            self.selected_machine_id = selected_candidate.machine.machine_id
        self.refresh()
        done = bool(terminated or truncated)
        info = {
            "hierarchical_action_kind": action_kind,
            "internal_action_id": internal_action_id,
            "internal_action_kind": selected_candidate.action_kind,
            "phase": self.snapshot.phase,
            "selected_machine_id": self.selected_machine_id,
            "des_state_mutated": True,
        }
        info.update(step_info)
        return self.snapshot, float(reward), done, info

    def _select_row_by_index(self, action_index: int) -> Dict:
        """계층형 action index를 현재 table row로 변환한다."""

        if not isinstance(action_index, (int, np.integer)):
            print(
                "[ERROR][HierarchicalDESActionAdapter._select_row_by_index] "
                f"cause=non_integer_action_index action_index={action_index}"
            )
            raise TypeError("action_index must be an integer")
        if action_index < 0 or action_index >= self.max_actions:
            print(
                "[ERROR][HierarchicalDESActionAdapter._select_row_by_index] "
                f"cause=action_index_out_of_range action_index={action_index} max_actions={self.max_actions}"
            )
            raise IndexError("action_index out of range")
        if not bool(self.snapshot.action_mask[int(action_index)]):
            print(
                "[ERROR][HierarchicalDESActionAdapter._select_row_by_index] "
                "cause=masked_or_padded_action_selected "
                f"action_index={action_index} available_action_count={len(self.snapshot.action_id_table)}"
            )
            raise RuntimeError("selected hierarchical action index is masked/padded")
        return self.snapshot.action_id_table[int(action_index)]

    def _candidate_by_action_id(self, action_id: str) -> ActionCandidate:
        """현재 snapshot의 원본 DES 후보에서 action_id를 찾는다."""

        for candidate in self.snapshot.candidates:
            if candidate.action_id == action_id:
                return candidate
        print(
            "[ERROR][HierarchicalDESActionAdapter._candidate_by_action_id] "
            f"cause=internal_action_not_found action_id={action_id}"
        )
        raise KeyError(f"internal action not found: {action_id}")

    def _build_select_machine_rows(self, candidates: Sequence[ActionCandidate]) -> List[Dict]:
        """원본 DES 후보 목록에서 machine 선택 row를 만든다."""

        grouped: Dict[str, Dict] = {}
        for candidate in candidates:
            machine_id = candidate.machine.machine_id
            if machine_id not in grouped:
                grouped[machine_id] = {
                    "index": -1,
                    "action_id": f"select_machine:{machine_id}",
                    "action_kind": "select_machine",
                    "machine_id": machine_id,
                    "machine_bay_id": candidate.machine.bay_id or "",
                    "candidate_count_for_machine": 0,
                    "internal_action_id": "",
                    "internal_action_kind": "",
                    "job_id": "",
                    "batch_id": "",
                    "batch_job_ids": [],
                    "batch_wo_count": 0,
                    "batch_length_sum": 0.0,
                    "estimated_minutes": 0.0,
                    "processing_minutes": 0.0,
                    "soft_penalty": 0.0,
                }
            grouped[machine_id]["candidate_count_for_machine"] += 1
        return [grouped[machine_id] for machine_id in sorted(grouped)]

    def _build_selected_machine_rows(
        self,
        candidates: Sequence[ActionCandidate],
        machine_id: str,
    ) -> List[Dict]:
        """선택된 machine에서 가능한 W/O 추가 또는 commit row를 만든다."""

        selected = [candidate for candidate in candidates if candidate.machine.machine_id == machine_id]
        rows: List[Dict] = []
        candidate_count = len(selected)
        for candidate in selected:
            if candidate.action_kind in {"open_batch", "add_to_batch", "single_dispatch"}:
                hierarchical_kind = "select_wo"
            elif candidate.action_kind == "close_batch":
                hierarchical_kind = "commit_batch"
            else:
                print(
                    "[ERROR][HierarchicalDESActionAdapter._build_selected_machine_rows] "
                    "cause=unsupported_internal_action_kind "
                    f"action_kind={candidate.action_kind} action_id={candidate.action_id}"
                )
                raise ValueError(f"unsupported internal action_kind: {candidate.action_kind}")
            rows.append(
                {
                    "index": -1,
                    "action_id": f"{hierarchical_kind}:{candidate.action_id}",
                    "action_kind": hierarchical_kind,
                    "machine_id": candidate.machine.machine_id,
                    "machine_bay_id": candidate.machine.bay_id or "",
                    "candidate_count_for_machine": candidate_count,
                    "internal_action_id": candidate.action_id,
                    "internal_action_kind": candidate.action_kind,
                    "job_id": candidate.job.job_id,
                    "batch_id": candidate.batch_id or "",
                    "batch_job_ids": list(candidate.batch_job_ids),
                    "batch_wo_count": int(candidate.batch_wo_count),
                    "batch_length_sum": float(candidate.batch_length_sum),
                    "estimated_minutes": float(candidate.estimated_minutes),
                    "processing_minutes": float(candidate.processing_minutes),
                    "soft_penalty": float(candidate.soft_penalty),
                }
            )
        return rows

    def _open_batch_summary_for_machine(self, machine_id: str) -> Tuple[Optional[str], Tuple[str, ...]]:
        """선택된 machine에 열린 batch가 있으면 batch id와 W/O 목록을 반환한다."""

        matches = [
            (batch_id, batch)
            for batch_id, batch in self.env.simulation.state.open_batches.items()
            if batch.machine_id == machine_id
        ]
        if not matches:
            return None, ()
        if len(matches) > 1:
            print(
                "[ERROR][HierarchicalDESActionAdapter._open_batch_summary_for_machine] "
                f"cause=multiple_open_batches_for_machine machine_id={machine_id} count={len(matches)}"
            )
            raise RuntimeError(f"multiple open batches for machine_id={machine_id}")
        batch_id, batch = matches[0]
        return batch_id, tuple(batch.job_ids)


class PMSPHierarchicalGymnasiumWrapper(gym.Env if GYMNASIUM_AVAILABLE else object):
    """계층형 action adapter를 실제 Gymnasium `Env` 형태로 감싼 wrapper.

    이 wrapper의 action index는 현재 phase에 따라 의미가 달라진다.
    - `SELECT_MACHINE`: index는 machine 선택 row를 가리킨다. DES state는 바뀌지 않는다.
    - `SELECT_WO`: index는 W/O 추가 또는 batch commit row를 가리킨다. 기존 DES 후보를 실행한다.
    """

    metadata = {"render_modes": []}

    def __init__(self, env: CuttingShopEnvironment, max_actions: int = 4096):
        require_gymnasium()
        self.core_env = env
        self.max_actions = int(max_actions)
        self.adapter = HierarchicalDESActionAdapter(env=env, max_actions=self.max_actions)
        self.action_space = spaces.Discrete(self.max_actions)
        self.observation_space = spaces.Dict(
            {
                "global_features": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
                "phase_code": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "action_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_actions, len(HIERARCHICAL_FEATURE_NAMES)),
                    dtype=np.float32,
                ),
                "action_mask": spaces.MultiBinary(self.max_actions),
            }
        )
        self._terminated = False
        self._truncated = False

    def _build_gym_observation(self) -> Dict[str, np.ndarray]:
        """Gymnasium observation_space와 일치하는 계층형 observation을 만든다."""

        snapshot = self.adapter.snapshot
        return {
            "global_features": _global_feature_vector(self.core_env),
            "phase_code": np.asarray([_hierarchical_phase_code(snapshot.phase)], dtype=np.float32),
            "action_features": snapshot.action_features.copy(),
            "action_mask": snapshot.action_mask.astype(np.int8, copy=True),
        }

    def _build_info(self) -> Dict:
        """계층형 action table과 core observation을 Gym info에 담는다."""

        snapshot = self.adapter.snapshot
        return {
            "phase": snapshot.phase,
            "selected_machine_id": snapshot.selected_machine_id or "",
            "selected_batch_id": snapshot.selected_batch_id or "",
            "current_batch_job_ids": tuple(snapshot.current_batch_job_ids),
            "available_action_count": len(snapshot.action_id_table),
            "action_id_table": self.adapter.current_action_table(),
            "action_feature_names": list(HIERARCHICAL_FEATURE_NAMES),
            "core_observation": self.core_env.simulation.build_observation(),
        }

    def reset(self, *, seed=None, options=None):
        """Gymnasium reset API. 초기 계층형 phase와 action mask를 반환한다."""

        super().reset(seed=seed)
        self.core_env.reset()
        self.adapter.selected_machine_id = None
        self.adapter.refresh()
        self._terminated = not self.core_env.simulation._has_pending_work()
        self._truncated = False
        return self._build_gym_observation(), self._build_info()

    def step(self, action_index: int):
        """현재 phase의 action index를 선택해 계층형 환경을 한 step 전진한다."""

        if self._terminated or self._truncated:
            print(
                "[ERROR][PMSPHierarchicalGymnasiumWrapper.step] "
                f"cause=step_called_after_done terminated={self._terminated} truncated={self._truncated}"
            )
            raise RuntimeError("cannot step after terminated/truncated; call reset first")

        _, reward, _, step_info = self.adapter.step(action_index)
        self._terminated = not self.core_env.simulation._has_pending_work()
        self._truncated = (
            self.core_env.simulation.state.current_time
            > self.core_env.simulation.config["simulation"]["horizon_minutes"]
        )
        info = self._build_info()
        info.update(step_info)
        return self._build_gym_observation(), float(reward), self._terminated, self._truncated, info

    def action_masks(self) -> np.ndarray:
        """Stable-Baselines 계열에서 기대하는 계층형 action mask helper."""

        return self.adapter.action_masks()


class PMSPGymnasiumWrapper(gym.Env if GYMNASIUM_AVAILABLE else object):
    """PMSP DES core 위에 얇게 붙는 Gymnasium wrapper."""

    metadata = {"render_modes": []}

    def __init__(self, env: CuttingShopEnvironment, max_actions: int = 4096):
        require_gymnasium()
        self.core_env = env
        self.max_actions = int(max_actions)
        self.adapter = DESActionMaskAdapter(env=env, max_actions=self.max_actions)
        self.action_space = spaces.Discrete(self.max_actions)
        self.observation_space = spaces.Dict(
            {
                "global_features": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
                "candidate_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_actions, len(CANDIDATE_FEATURE_NAMES)),
                    dtype=np.float32,
                ),
                "action_mask": spaces.MultiBinary(self.max_actions),
            }
        )
        self._terminated = False
        self._truncated = False

    def _build_gym_observation(self) -> Dict[str, np.ndarray]:
        """Gymnasium `observation_space`와 일치하는 숫자 observation을 만든다."""

        snapshot = self.adapter.snapshot
        return {
            "global_features": _global_feature_vector(self.core_env),
            "candidate_features": snapshot.candidate_features.copy(),
            "action_mask": snapshot.action_mask.astype(np.int8, copy=True),
        }

    def _build_info(self) -> Dict:
        """문자열 action_id table과 core observation은 Gym info에 담는다."""

        return {
            "available_action_count": len(self.adapter.snapshot.candidates),
            "action_id_table": self.adapter.current_action_table(),
            "candidate_feature_names": list(CANDIDATE_FEATURE_NAMES),
            "core_observation": self.core_env.simulation.build_observation(),
        }

    def reset(self, *, seed=None, options=None):
        """Gymnasium reset API. 초기 state와 action mask를 반환한다."""

        super().reset(seed=seed)
        self.core_env.reset()
        self.adapter.refresh()
        self._terminated = not self.core_env.simulation._has_pending_work()
        self._truncated = False
        return self._build_gym_observation(), self._build_info()

    def step(self, action_index: int):
        """현재 action mask의 index를 선택해 DES state를 한 step 전진한다."""

        if self._terminated or self._truncated:
            print(
                "[ERROR][PMSPGymnasiumWrapper.step] "
                f"cause=step_called_after_done terminated={self._terminated} truncated={self._truncated}"
            )
            raise RuntimeError("cannot step after terminated/truncated; call reset first")

        selected = self.adapter.select_candidate_by_index(action_index)
        _, reward, terminated, truncated, step_info = self.core_env.simulation.step_candidate(
            selected,
            build_observation_result=False,
            advance_decision_epoch=True,
        )
        self.adapter.refresh()
        self._terminated = bool(terminated)
        self._truncated = bool(truncated)
        info = self._build_info()
        info.update(step_info)
        return self._build_gym_observation(), float(reward), self._terminated, self._truncated, info

    def action_masks(self) -> np.ndarray:
        """Stable-Baselines 계열에서 기대하는 action mask helper."""

        return self.adapter.action_masks()


def _simulation_summary(env: CuttingShopEnvironment) -> Dict:
    """직접 DES 실행과 wrapper 실행을 비교하기 위한 summary를 만든다."""

    return {
        "scheduled_jobs": len(env.simulation.state.schedule),
        "unscheduled_jobs": sorted(env.simulation.state.unscheduled_jobs),
        "makespan": float(env.simulation.get_makespan()),
        "machine_loads": {key: float(value) for key, value in sorted(env.simulation.state.machine_loads.items())},
        "event_count": len(env.simulation.export_event_log()),
        "load_imbalance": float(compute_load_imbalance(env.simulation.state.machine_loads)),
    }


def _run_direct_heuristic_trace(env: CuttingShopEnvironment, heuristic_name: str) -> Tuple[List[Dict], Dict]:
    """기존 DES core를 action_id 직접 방식으로 실행한다."""

    env.reset()
    trace: List[Dict] = []
    step_index = 0
    while env.simulation._has_pending_work():
        candidates = env.get_action_candidates()
        if not candidates:
            print(
                "[ERROR][gym_wrapper._run_direct_heuristic_trace] "
                "cause=no_candidates_with_pending_work "
                f"step={step_index} current_time={env.simulation.state.current_time}"
            )
            raise RuntimeError("direct heuristic has pending work but no candidates")
        selected = select_action_by_rule(candidates, env.simulation, heuristic_name)
        if selected is None:
            print(
                "[ERROR][gym_wrapper._run_direct_heuristic_trace] "
                f"cause=heuristic_returned_none heuristic={heuristic_name} step={step_index}"
            )
            raise RuntimeError("heuristic returned None")
        trace.append(
            {
                "step": step_index,
                "path": "direct",
                "candidate_count": len(candidates),
                "action_id": selected.action_id,
                "action_kind": selected.action_kind,
                "job_id": selected.job.job_id,
                "machine_id": selected.machine.machine_id,
                "batch_id": selected.batch_id or "",
                "batch_job_ids": "|".join(selected.batch_job_ids),
            }
        )
        env.simulation.step_candidate(selected, build_observation_result=False, advance_decision_epoch=True)
        step_index += 1
    return trace, _simulation_summary(env)


def _find_action_index(action_id_table: Sequence[Dict], action_id: str) -> int:
    """action_id table에서 특정 action_id의 index를 찾는다."""

    for row in action_id_table:
        if row["action_id"] == action_id:
            return int(row["index"])
    print(
        "[ERROR][gym_wrapper._find_action_index] "
        f"cause=action_id_not_found action_id={action_id} available_count={len(action_id_table)}"
    )
    raise KeyError(f"action_id not found in wrapper table: {action_id}")


def _run_wrapper_heuristic_trace(
    env: CuttingShopEnvironment,
    heuristic_name: str,
    max_actions: int,
) -> Tuple[List[Dict], Dict]:
    """Gymnasium 없이 adapter index 방식으로 기존 휴리스틱을 재현한다."""

    env.reset()
    adapter = DESActionMaskAdapter(env=env, max_actions=max_actions)
    adapter.refresh()
    trace: List[Dict] = []
    step_index = 0
    while env.simulation._has_pending_work():
        candidates = adapter.snapshot.candidates
        if not candidates:
            print(
                "[ERROR][gym_wrapper._run_wrapper_heuristic_trace] "
                "cause=no_candidates_with_pending_work "
                f"step={step_index} current_time={env.simulation.state.current_time}"
            )
            raise RuntimeError("wrapper has pending work but no candidates")
        before_mask = adapter.action_masks()
        before_table = adapter.current_action_table()
        after_mask = adapter.action_masks()
        after_table = adapter.current_action_table()
        if not np.array_equal(before_mask, after_mask) or before_table != after_table:
            print(
                "[ERROR][gym_wrapper._run_wrapper_heuristic_trace] "
                f"cause=unstable_action_mask step={step_index}"
            )
            raise RuntimeError("action mask is unstable within one decision epoch")

        selected = select_action_by_rule(candidates, env.simulation, heuristic_name)
        if selected is None:
            print(
                "[ERROR][gym_wrapper._run_wrapper_heuristic_trace] "
                f"cause=heuristic_returned_none heuristic={heuristic_name} step={step_index}"
            )
            raise RuntimeError("heuristic returned None")
        action_index = _find_action_index(before_table, selected.action_id)
        selected_by_index = adapter.select_candidate_by_index(action_index)
        trace.append(
            {
                "step": step_index,
                "path": "wrapper",
                "candidate_count": len(candidates),
                "mask_true_count": int(before_mask.sum()),
                "action_index": action_index,
                "action_id": selected_by_index.action_id,
                "action_kind": selected_by_index.action_kind,
                "job_id": selected_by_index.job.job_id,
                "machine_id": selected_by_index.machine.machine_id,
                "batch_id": selected_by_index.batch_id or "",
                "batch_job_ids": "|".join(selected_by_index.batch_job_ids),
            }
        )
        env.simulation.step_candidate(selected_by_index, build_observation_result=False, advance_decision_epoch=True)
        adapter.refresh()
        step_index += 1
    return trace, _simulation_summary(env)


def _candidate_pool_from_hierarchical_rows(snapshot: HierarchicalActionMaskSnapshot) -> List[ActionCandidate]:
    """현재 계층형 phase에서 실제로 선택 가능한 DES 후보만 추린다.

    `SELECT_WO` phase에서는 action table에 노출된 machine의 `select_wo` 또는
    `commit_batch` 후보만 학습기가 고를 수 있다. 따라서 pseudo-label을 만들 때도
    전체 DES 후보가 아니라 현재 action table에 들어 있는 내부 후보만 사용한다.
    """

    action_ids = [
        str(row.get("internal_action_id"))
        for row in snapshot.action_id_table
        if str(row.get("internal_action_id") or "").strip()
    ]
    if not action_ids:
        print(
            "[ERROR][gym_wrapper._candidate_pool_from_hierarchical_rows] "
            f"cause=empty_internal_action_ids phase={snapshot.phase}"
        )
        raise RuntimeError("hierarchical SELECT_WO phase has no internal action ids")

    candidate_by_id = {candidate.action_id: candidate for candidate in snapshot.candidates}
    missing = sorted({action_id for action_id in action_ids if action_id not in candidate_by_id})
    if missing:
        print(
            "[ERROR][gym_wrapper._candidate_pool_from_hierarchical_rows] "
            f"cause=action_table_candidate_missing missing={missing[:5]} missing_count={len(missing)}"
        )
        raise KeyError("hierarchical action table references candidates not present in snapshot")
    return [candidate_by_id[action_id] for action_id in action_ids]


def _find_hierarchical_row(
    action_id_table: Sequence[Dict],
    *,
    action_kind: str,
    machine_id: Optional[str] = None,
    internal_action_id: Optional[str] = None,
) -> Dict:
    """현재 계층형 action table에서 선택할 row를 하나만 찾는다."""

    matches = []
    for row in action_id_table:
        if row.get("action_kind") != action_kind:
            continue
        if machine_id is not None and row.get("machine_id") != machine_id:
            continue
        if internal_action_id is not None and row.get("internal_action_id") != internal_action_id:
            continue
        matches.append(row)

    if len(matches) != 1:
        print(
            "[ERROR][gym_wrapper._find_hierarchical_row] "
            f"cause=unexpected_match_count count={len(matches)} action_kind={action_kind} "
            f"machine_id={machine_id} internal_action_id={internal_action_id} "
            f"available_count={len(action_id_table)}"
        )
        raise RuntimeError("hierarchical row lookup must match exactly one row")
    return matches[0]


def _append_hierarchical_trace_record(
    trace: List[Dict],
    *,
    hierarchical_step: int,
    flat_decision: int,
    snapshot: HierarchicalActionMaskSnapshot,
    row: Dict,
    target_internal_action_id: str,
    reward: float,
    terminated: bool,
    truncated: bool,
    des_state_mutated: bool,
    env: CuttingShopEnvironment,
) -> None:
    """선택된 계층형 action 1개를 CSV row 형태로 기록한다."""

    trace.append(
        {
            "hierarchical_step": hierarchical_step,
            "flat_decision": flat_decision,
            "phase": snapshot.phase,
            "action_index": int(row["index"]),
            "mask_true_count": int(snapshot.action_mask.sum()),
            "available_action_count": len(snapshot.action_id_table),
            "selected_hierarchical_action_id": row["action_id"],
            "hierarchical_action_kind": row["action_kind"],
            "target_internal_action_id": target_internal_action_id,
            "internal_action_id": row.get("internal_action_id", ""),
            "internal_action_kind": row.get("internal_action_kind", ""),
            "machine_id": row.get("machine_id", ""),
            "machine_bay_id": row.get("machine_bay_id", ""),
            "job_id": row.get("job_id", ""),
            "batch_id": row.get("batch_id", ""),
            "batch_job_ids": "|".join(row.get("batch_job_ids", [])),
            "batch_wo_count": row.get("batch_wo_count", 0),
            "batch_length_sum": round(float(row.get("batch_length_sum", 0.0)), 6),
            "reward": round(float(reward), 6),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "des_state_mutated": bool(des_state_mutated),
            "remaining_jobs": len(env.simulation.state.unscheduled_jobs),
            "current_time_min": round(float(env.simulation.state.current_time), 6),
        }
    )


def _append_hierarchical_action_table_record(
    records: List[Dict],
    *,
    hierarchical_step: int,
    flat_decision: int,
    snapshot: HierarchicalActionMaskSnapshot,
    observation: Dict[str, np.ndarray],
) -> None:
    """해당 phase에서 학습기가 본 action table과 mask를 JSONL용으로 저장한다."""

    records.append(
        {
            "hierarchical_step": hierarchical_step,
            "flat_decision": flat_decision,
            "phase": snapshot.phase,
            "selected_machine_id": snapshot.selected_machine_id or "",
            "selected_batch_id": snapshot.selected_batch_id or "",
            "current_batch_job_ids": list(snapshot.current_batch_job_ids),
            "mask_true_indices": np.flatnonzero(observation["action_mask"]).astype(int).tolist(),
            "global_features": observation["global_features"].astype(float).tolist(),
            "phase_code": observation["phase_code"].astype(float).tolist(),
            "action_feature_names": list(HIERARCHICAL_FEATURE_NAMES),
            "action_id_table": copy.deepcopy(snapshot.action_id_table),
        }
    )


def _write_hierarchical_trace_outputs(
    output_dir: str,
    trace: Sequence[Dict],
    action_table_records: Sequence[Dict],
    summary: Dict,
) -> Dict:
    """계층형 trace CSV, action table JSONL, summary JSON을 저장한다."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trace_path = output_path / "hierarchical_trace.csv"
    table_path = output_path / "hierarchical_action_table.jsonl"
    summary_path = output_path / "summary.json"

    fieldnames = [
        "hierarchical_step",
        "flat_decision",
        "phase",
        "action_index",
        "mask_true_count",
        "available_action_count",
        "selected_hierarchical_action_id",
        "hierarchical_action_kind",
        "target_internal_action_id",
        "internal_action_id",
        "internal_action_kind",
        "machine_id",
        "machine_bay_id",
        "job_id",
        "batch_id",
        "batch_job_ids",
        "batch_wo_count",
        "batch_length_sum",
        "reward",
        "terminated",
        "truncated",
        "des_state_mutated",
        "remaining_jobs",
        "current_time_min",
    ]
    with trace_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in trace:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    with table_path.open("w", encoding="utf-8") as handle:
        for record in action_table_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary["trace_csv"] = str(trace_path)
    summary["action_table_jsonl"] = str(table_path)
    summary["summary_json"] = str(summary_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def run_hierarchical_trace_export(
    env: CuttingShopEnvironment,
    heuristic_name: str,
    max_actions: int,
    output_dir: str,
) -> Dict:
    """계층형 Gym wrapper action trace를 학습/디버깅용 파일로 저장한다.

    이 함수는 새 제약을 계산하지 않는다. 현재 phase의 wrapper action table에
    노출된 후보만 휴리스틱 label 후보로 사용한다.
    """

    print(
        "[CHECK][gym_wrapper.run_hierarchical_trace_export] "
        f"heuristic={heuristic_name} max_actions={max_actions} output_dir={output_dir}"
    )
    wrapper = PMSPHierarchicalGymnasiumWrapper(env=env, max_actions=max_actions)
    observation, _ = wrapper.reset()
    trace: List[Dict] = []
    action_table_records: List[Dict] = []
    hierarchical_step = 0
    flat_decision = 0
    phase_counts: Dict[str, int] = {}
    terminated = False
    truncated = False

    while not (terminated or truncated):
        snapshot = wrapper.adapter.snapshot
        if not env.simulation._has_pending_work():
            break
        phase_counts[snapshot.phase] = phase_counts.get(snapshot.phase, 0) + 1
        _append_hierarchical_action_table_record(
            action_table_records,
            hierarchical_step=hierarchical_step,
            flat_decision=flat_decision,
            snapshot=snapshot,
            observation=observation,
        )

        if snapshot.phase == HIERARCHICAL_PHASE_SELECT_MACHINE:
            candidates = snapshot.candidates
            if not candidates:
                print(
                    "[ERROR][gym_wrapper.run_hierarchical_trace_export] "
                    f"cause=no_candidates_in_select_machine step={hierarchical_step}"
                )
                raise RuntimeError("SELECT_MACHINE phase has no candidates")
            selected = select_action_by_rule(candidates, env.simulation, heuristic_name)
            if selected is None:
                print(
                    "[ERROR][gym_wrapper.run_hierarchical_trace_export] "
                    f"cause=heuristic_returned_none phase={snapshot.phase} heuristic={heuristic_name}"
                )
                raise RuntimeError("heuristic returned None in SELECT_MACHINE phase")
            row = _find_hierarchical_row(
                snapshot.action_id_table,
                action_kind="select_machine",
                machine_id=selected.machine.machine_id,
            )
            target_internal_action_id = selected.action_id
        elif snapshot.phase == HIERARCHICAL_PHASE_SELECT_WO:
            candidates = _candidate_pool_from_hierarchical_rows(snapshot)
            selected = select_action_by_rule(candidates, env.simulation, heuristic_name)
            if selected is None:
                print(
                    "[ERROR][gym_wrapper.run_hierarchical_trace_export] "
                    f"cause=heuristic_returned_none phase={snapshot.phase} heuristic={heuristic_name}"
                )
                raise RuntimeError("heuristic returned None in SELECT_WO phase")
            row = _find_hierarchical_row(
                snapshot.action_id_table,
                action_kind="commit_batch" if selected.action_kind == "close_batch" else "select_wo",
                internal_action_id=selected.action_id,
            )
            target_internal_action_id = selected.action_id
        else:
            print(
                "[ERROR][gym_wrapper.run_hierarchical_trace_export] "
                f"cause=unsupported_phase phase={snapshot.phase}"
            )
            raise RuntimeError(f"unsupported hierarchical phase: {snapshot.phase}")

        observation, reward, terminated, truncated, info = wrapper.step(int(row["index"]))
        _append_hierarchical_trace_record(
            trace,
            hierarchical_step=hierarchical_step,
            flat_decision=flat_decision,
            snapshot=snapshot,
            row=row,
            target_internal_action_id=target_internal_action_id,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            des_state_mutated=bool(info.get("des_state_mutated", False)),
            env=env,
        )
        if bool(info.get("des_state_mutated", False)):
            flat_decision += 1
        hierarchical_step += 1

    summary = _simulation_summary(env)
    summary.update(
        {
            "heuristic": heuristic_name,
            "max_actions": int(max_actions),
            "hierarchical_step_count": len(trace),
            "flat_decision_count": flat_decision,
            "phase_counts": phase_counts,
            "gymnasium_available": GYMNASIUM_AVAILABLE,
        }
    )
    result = _write_hierarchical_trace_outputs(output_dir, trace, action_table_records, summary)
    print(
        "[VALIDATION][gym_wrapper.run_hierarchical_trace_export] "
        "passed=true "
        f"hierarchical_steps={result['hierarchical_step_count']} "
        f"flat_decisions={result['flat_decision_count']} "
        f"scheduled_jobs={result['scheduled_jobs']}"
    )
    return result


def _compare_equivalence(direct_trace: Sequence[Dict], wrapper_trace: Sequence[Dict], direct_summary: Dict, wrapper_summary: Dict) -> Dict:
    """직접 실행과 wrapper index 실행이 같은지 비교한다."""

    direct_actions = [row["action_id"] for row in direct_trace]
    wrapper_actions = [row["action_id"] for row in wrapper_trace]
    action_sequence_match = direct_actions == wrapper_actions
    summary_match = (
        direct_summary["scheduled_jobs"] == wrapper_summary["scheduled_jobs"]
        and direct_summary["unscheduled_jobs"] == wrapper_summary["unscheduled_jobs"]
        and abs(direct_summary["makespan"] - wrapper_summary["makespan"]) <= 1e-9
        and direct_summary["machine_loads"] == wrapper_summary["machine_loads"]
        and direct_summary["event_count"] == wrapper_summary["event_count"]
    )
    passed = action_sequence_match and summary_match
    if not passed:
        print(
            "[ERROR][gym_wrapper._compare_equivalence] "
            "cause=equivalence_failed "
            f"action_sequence_match={action_sequence_match} summary_match={summary_match}"
        )
    return {
        "passed": passed,
        "action_sequence_match": action_sequence_match,
        "summary_match": summary_match,
        "direct_step_count": len(direct_trace),
        "wrapper_step_count": len(wrapper_trace),
        "direct_summary": direct_summary,
        "wrapper_summary": wrapper_summary,
        "gymnasium_available": GYMNASIUM_AVAILABLE,
    }


def _write_trace_csv(path: Path, direct_trace: Sequence[Dict], wrapper_trace: Sequence[Dict]) -> None:
    """직접 실행과 wrapper 실행 trace를 한 CSV에 저장한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "path",
        "candidate_count",
        "mask_true_count",
        "action_index",
        "action_id",
        "action_kind",
        "job_id",
        "machine_id",
        "batch_id",
        "batch_job_ids",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in list(direct_trace) + list(wrapper_trace):
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_wrapper_equivalence(
    direct_env: CuttingShopEnvironment,
    wrapper_env: CuttingShopEnvironment,
    heuristic_name: str,
    max_actions: int,
    output_dir: str,
) -> Dict:
    """기존 DES 실행과 wrapper action-index 실행을 비교하고 산출물을 저장한다."""

    print(
        "[CHECK][gym_wrapper.run_wrapper_equivalence] "
        f"heuristic={heuristic_name} max_actions={max_actions} output_dir={output_dir}"
    )
    direct_trace, direct_summary = _run_direct_heuristic_trace(direct_env, heuristic_name)
    wrapper_trace, wrapper_summary = _run_wrapper_heuristic_trace(wrapper_env, heuristic_name, max_actions)
    summary = _compare_equivalence(direct_trace, wrapper_trace, direct_summary, wrapper_summary)

    output_path = Path(output_dir)
    trace_path = output_path / "action_trace.csv"
    summary_path = output_path / "summary.json"
    _write_trace_csv(trace_path, direct_trace, wrapper_trace)
    output_path.mkdir(parents=True, exist_ok=True)
    summary["trace_csv"] = str(trace_path)
    summary["summary_json"] = str(summary_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if not summary["passed"]:
        print(
            "[VALIDATION][gym_wrapper.run_wrapper_equivalence] "
            "passed=false violations=1"
        )
        raise RuntimeError("wrapper equivalence validation failed")
    print(
        "[VALIDATION][gym_wrapper.run_wrapper_equivalence] "
        "passed=true violations=0 "
        f"steps={summary['direct_step_count']} scheduled_jobs={direct_summary['scheduled_jobs']} "
        f"makespan={direct_summary['makespan']:.6f}"
    )
    return summary
