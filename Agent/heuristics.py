"""비교/검증용 기본 휴리스틱.

이 파일은 강화학습이 없더라도 환경을 바로 실행해볼 수 있게 해줍니다.

현재 제공하는 휴리스틱:
- spt: 처리시간이 짧은 작업 우선
- priority: 우선순위가 높은 작업 우선
- load_balance: 현재 부하가 낮은 설비를 선호

입력/출력 흐름:
- 입력: `CuttingSimulation.get_candidates()`가 만든 `ActionCandidate` 목록
- 출력: 후보 중 선택된 `ActionCandidate` 1개 또는 후보가 없을 때 None

중요한 전제:
- 후보 목록은 이미 hard constraint를 통과한 action만 포함해야 한다.
- 따라서 이 파일은 제약 위반을 직접 고치는 곳이 아니다.
- 제약을 추가하려면 `Environment/constraints/`와 registry를 수정해야 한다.
"""

# LINE-BY-LINE: `typing` 모듈에서 `Any, Dict, Sequence`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Dict, Sequence


def _load_range(loads: Dict[str, float]) -> float:
    """부하평준화 비교용 max-min 지표를 계산한다."""

    if not loads:
        print("[ERROR][Agent.heuristics._load_range] cause=empty_loads")
        raise ValueError("cannot compute load range from empty loads")
    values = [float(value) for value in loads.values()]
    return max(values) - min(values)


def _open_batch_jobs(simulation, batch_id: str, job_ids: Sequence[str], action_id: str) -> tuple:
    """open batch에 들어 있는 W/O 객체를 오류 출력과 함께 조회한다."""

    jobs = []
    for job_id in job_ids:
        if job_id not in simulation.jobs:
            print(
                "[ERROR][Agent.heuristics._open_batch_jobs] "
                f"cause=unknown_open_batch_job batch_id={batch_id} "
                f"job_id={job_id} action_id={action_id}"
            )
            raise KeyError(f"unknown open-batch job: {job_id}")
        jobs.append(simulation.jobs[job_id])
    if not jobs:
        print(
            "[ERROR][Agent.heuristics._open_batch_jobs] "
            f"cause=empty_open_batch batch_id={batch_id} action_id={action_id}"
        )
        raise ValueError(f"empty open batch: {batch_id}")
    return tuple(jobs)


def _project_machine_loads(simulation, candidate) -> Dict[str, float]:
    """candidate를 지금 batch로 닫는다고 가정했을 때의 설비별 누적 부하."""

    machine_id = candidate.machine.machine_id
    if machine_id not in simulation.state.machine_loads:
        print(
            "[ERROR][Agent.heuristics._project_machine_loads] "
            f"cause=unknown_machine_load machine_id={machine_id} action_id={candidate.action_id}"
        )
        raise KeyError(f"unknown machine load: {machine_id}")
    projected = {key: float(value) for key, value in simulation.state.machine_loads.items()}
    for batch_id, batch in simulation.state.open_batches.items():
        if batch_id == candidate.batch_id:
            continue
        batch_machine_id = batch.machine_id
        if batch_machine_id not in simulation.machines:
            print(
                "[ERROR][Agent.heuristics._project_machine_loads] "
                f"cause=unknown_open_batch_machine batch_id={batch_id} "
                f"machine_id={batch_machine_id} action_id={candidate.action_id}"
            )
            raise KeyError(f"unknown open-batch machine: {batch_machine_id}")
        if batch_machine_id not in projected:
            print(
                "[ERROR][Agent.heuristics._project_machine_loads] "
                f"cause=open_batch_machine_missing_load batch_id={batch_id} "
                f"machine_id={batch_machine_id} action_id={candidate.action_id}"
            )
            raise KeyError(f"missing open-batch machine load: {batch_machine_id}")
        jobs = _open_batch_jobs(simulation, batch_id, tuple(batch.job_ids), candidate.action_id)
        projected[batch_machine_id] += float(
            simulation._batch_processing_minutes_for(jobs, simulation.machines[batch_machine_id])
        )
    projected[machine_id] += float(candidate.processing_minutes)
    return projected


def _project_bay_loads(simulation, machine_loads: Dict[str, float]) -> Dict[str, float]:
    """설비별 부하를 Machine.bay_id 기준으로 집계한다."""

    bay_loads: Dict[str, float] = {}
    for machine_id, load in machine_loads.items():
        if machine_id not in simulation.machines:
            print(
                "[ERROR][Agent.heuristics._project_bay_loads] "
                f"cause=unknown_machine machine_id={machine_id}"
            )
            raise KeyError(f"unknown machine: {machine_id}")
        bay_id = simulation.machines[machine_id].bay_id
        if bay_id is None or str(bay_id).strip() == "":
            print(
                "[ERROR][Agent.heuristics._project_bay_loads] "
                f"cause=missing_machine_bay machine_id={machine_id}"
            )
            raise ValueError(f"missing bay_id for machine: {machine_id}")
        bay_key = str(bay_id)
        bay_loads[bay_key] = bay_loads.get(bay_key, 0.0) + float(load)
    return bay_loads


def _candidate_job_count(candidate) -> int:
    """후보 action이 설비에 배정하려는 W/O 수를 읽는다."""

    batch_wo_count = int(candidate.batch_wo_count or 0)
    if batch_wo_count > 0:
        return batch_wo_count
    batch_job_ids = tuple(candidate.batch_job_ids or ())
    if batch_job_ids:
        return len(batch_job_ids)
    return 1


def _max_batch_wo_count_for_candidate(simulation, candidate) -> int:
    """close_batch 후보가 채울 수 있는 최대 W/O 수를 읽는다.

    generated planning에서 batch 한계는 hard constraint의 핵심 입력이다. 따라서
    close_batch인데 open batch나 machine 어디에도 한계가 없으면 조용히 1로
    대체하지 않고 실패시킨다.
    """

    if candidate.action_kind != "close_batch":
        return _candidate_job_count(candidate)

    if candidate.batch_id in simulation.state.open_batches:
        open_batch = simulation.state.open_batches[candidate.batch_id]
        max_wo_count = getattr(open_batch, "max_wo_count", None)
        if max_wo_count not in (None, ""):
            return int(max_wo_count)

    max_wo_count = getattr(candidate.machine, "max_batch_wo_count", None)
    if max_wo_count in (None, ""):
        print(
            "[ERROR][Agent.heuristics._max_batch_wo_count_for_candidate] "
            f"cause=missing_max_batch_wo_count action_id={candidate.action_id} "
            f"machine_id={candidate.machine.machine_id} batch_id={candidate.batch_id}"
        )
        raise ValueError(f"missing max_batch_wo_count for close_batch candidate: {candidate.action_id}")
    return int(max_wo_count)


def _close_batch_fill_deficit(simulation, candidate) -> int:
    """close_batch가 최대 3개 중 몇 자리를 비운 채 닫는지 계산한다."""

    if candidate.action_kind != "close_batch":
        return 0
    current_count = _candidate_job_count(candidate)
    max_wo_count = _max_batch_wo_count_for_candidate(simulation, candidate)
    return max(max_wo_count - current_count, 0)


def _project_machine_job_counts(simulation, candidate) -> Dict[str, float]:
    """candidate 선택 후 설비별 할당 W/O 개수를 예측한다."""

    machine_id = candidate.machine.machine_id
    if machine_id not in simulation.machines:
        print(
            "[ERROR][Agent.heuristics._project_machine_job_counts] "
            f"cause=unknown_candidate_machine machine_id={machine_id} action_id={candidate.action_id}"
        )
        raise KeyError(f"unknown candidate machine: {machine_id}")

    projected = {machine_key: 0.0 for machine_key in simulation.machines}
    for operation in simulation.state.schedule:
        operation_machine_id = operation.machine_id
        if operation_machine_id not in projected:
            print(
                "[ERROR][Agent.heuristics._project_machine_job_counts] "
                f"cause=unknown_scheduled_machine machine_id={operation_machine_id}"
            )
            raise KeyError(f"unknown scheduled machine: {operation_machine_id}")
        projected[operation_machine_id] += 1.0

    for batch_id, batch in simulation.state.open_batches.items():
        if batch_id == candidate.batch_id:
            continue
        batch_machine_id = batch.machine_id
        if batch_machine_id not in projected:
            print(
                "[ERROR][Agent.heuristics._project_machine_job_counts] "
                f"cause=unknown_open_batch_machine batch_id={batch_id} machine_id={batch_machine_id}"
            )
            raise KeyError(f"unknown open-batch machine: {batch_machine_id}")
        projected[batch_machine_id] += float(len(tuple(batch.job_ids)))

    projected[machine_id] += float(_candidate_job_count(candidate))
    return projected


def _project_bay_job_counts(simulation, machine_job_counts: Dict[str, float]) -> Dict[str, float]:
    """설비별 할당 W/O 수를 Machine.bay_id 기준 Bay별 W/O 수로 집계한다."""

    bay_counts: Dict[str, float] = {}
    for machine_id, job_count in machine_job_counts.items():
        if machine_id not in simulation.machines:
            print(
                "[ERROR][Agent.heuristics._project_bay_job_counts] "
                f"cause=unknown_machine machine_id={machine_id}"
            )
            raise KeyError(f"unknown machine: {machine_id}")
        bay_id = simulation.machines[machine_id].bay_id
        if bay_id is None or str(bay_id).strip() == "":
            print(
                "[ERROR][Agent.heuristics._project_bay_job_counts] "
                f"cause=missing_machine_bay machine_id={machine_id}"
            )
            raise ValueError(f"missing bay_id for machine: {machine_id}")
        bay_key = str(bay_id)
        bay_counts[bay_key] = bay_counts.get(bay_key, 0.0) + float(job_count)
    return bay_counts


def _project_makespan(simulation, candidate) -> float:
    """candidate를 close/schedule했을 때의 예상 전체 완료시각."""

    slot_times = [
        float(slot_time)
        for slot_list in simulation.state.machine_slot_available_at.values()
        for slot_time in slot_list
    ]
    pending_finish_times = []
    for batch_id, batch in simulation.state.open_batches.items():
        if batch_id == candidate.batch_id:
            continue
        if batch.machine_id not in simulation.machines:
            print(
                "[ERROR][Agent.heuristics._project_makespan] "
                f"cause=unknown_open_batch_machine batch_id={batch_id} "
                f"machine_id={batch.machine_id} action_id={candidate.action_id}"
            )
            raise KeyError(f"unknown open-batch machine: {batch.machine_id}")
        jobs = _open_batch_jobs(simulation, batch_id, tuple(batch.job_ids), candidate.action_id)
        timing = simulation._batch_candidate_timing(jobs, simulation.machines[batch.machine_id])
        pending_finish_times.append(float(timing["finish_time"]))
    current_makespan = max(slot_times) if slot_times else float(simulation.state.current_time)
    return max([current_makespan, float(candidate.finish_time), *pending_finish_times])


def _close_underfill_penalty(simulation, candidate) -> float:
    """1~2개 batch close를 허용하되, 이유 없는 single close는 약하게 억제한다."""

    if candidate.action_kind != "close_batch":
        return 0.0
    missing_count = _close_batch_fill_deficit(simulation, candidate)
    return missing_count * max(float(candidate.processing_minutes), 1.0) * 0.15


def _balanced_batch_score(simulation, candidate) -> tuple:
    """makespan, 설비 부하편차, Bay 부하편차를 함께 보는 batch 후보 점수."""

    action_rank = {
        "add_to_batch": 0,
        "close_batch": 1,
        "open_batch": 2,
        "single_dispatch": 3,
    }
    if candidate.action_kind not in action_rank:
        print(
            "[ERROR][Agent.heuristics._balanced_batch_score] "
            f"cause=unsupported_action_kind action_kind={candidate.action_kind} "
            f"action_id={candidate.action_id}"
        )
        raise ValueError(f"unsupported action kind for balanced_batch: {candidate.action_kind}")

    projected_machine_loads = _project_machine_loads(simulation, candidate)
    projected_bay_loads = _project_bay_loads(simulation, projected_machine_loads)
    projected_makespan = _project_makespan(simulation, candidate)
    machine_imbalance = _load_range(projected_machine_loads)
    bay_imbalance = _load_range(projected_bay_loads)
    underfill_penalty = _close_underfill_penalty(simulation, candidate)

    total_score = (
        projected_makespan
        + 0.75 * machine_imbalance
        + 0.75 * bay_imbalance
        + underfill_penalty
        + 0.01 * float(candidate.soft_penalty)
    )
    return (
        round(total_score, 6),
        round(projected_makespan, 6),
        round(machine_imbalance, 6),
        round(bay_imbalance, 6),
        action_rank[candidate.action_kind],
        -int(candidate.batch_wo_count or 1),
        round(float(candidate.estimated_minutes), 6),
        candidate.machine.machine_id,
        candidate.batch_id or "",
        candidate.job.job_id,
        candidate.action_id,
    )


def _balanced_batch_count_score(simulation, candidate) -> tuple:
    """할당 W/O 개수 기준으로 batch 후보를 평가한다."""

    action_rank = {
        "add_to_batch": 0,
        "close_batch": 1,
        "open_batch": 2,
        "single_dispatch": 3,
    }
    if candidate.action_kind not in action_rank:
        print(
            "[ERROR][Agent.heuristics._balanced_batch_count_score] "
            f"cause=unsupported_action_kind action_kind={candidate.action_kind} "
            f"action_id={candidate.action_id}"
        )
        raise ValueError(f"unsupported action kind for balanced_batch_count: {candidate.action_kind}")

    projected_machine_job_counts = _project_machine_job_counts(simulation, candidate)
    projected_bay_job_counts = _project_bay_job_counts(simulation, projected_machine_job_counts)
    projected_makespan = _project_makespan(simulation, candidate)
    machine_count_imbalance = _load_range(projected_machine_job_counts)
    bay_count_imbalance = _load_range(projected_bay_job_counts)
    fill_deficit = _close_batch_fill_deficit(simulation, candidate)
    count_imbalance = machine_count_imbalance + bay_count_imbalance

    return (
        round(float(fill_deficit), 6),
        round(count_imbalance, 6),
        round(bay_count_imbalance, 6),
        round(machine_count_imbalance, 6),
        round(projected_makespan, 6),
        action_rank[candidate.action_kind],
        -int(candidate.batch_wo_count or 1),
        round(float(candidate.estimated_minutes), 6),
        candidate.machine.machine_id,
        candidate.batch_id or "",
        candidate.job.job_id,
        candidate.action_id,
    )


# LINE-BY-LINE: `select_action_by_rule(candidates: Sequence, simulation, rule_name: str)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다. 예: action_id는 `job_001@PLS21` 또는 `open_batch:...` 형태.
def select_action_by_rule(candidates: Sequence, simulation, rule_name: str):
    """현재 후보 액션 중 하나를 고릅니다.

    candidates는 이미 하드 제약을 통과한 후보들입니다.
    따라서 이 함수는 주로
    - 처리시간
    - 우선순위
    - 부하 편차
    - 소프트 penalty
    를 기준으로 선택합니다.
    """

    # 후보가 없으면 정책이 선택할 수 있는 action이 없다.
    # 이 경우 simulation이 다음 decision epoch로 시간을 전진시킨다.
    # LINE-BY-LINE: 조건 `not candidates`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not candidates:
        # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return None

    # LINE-BY-LINE: 조건 `rule_name == "batch_fill_spt"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if rule_name == "batch_fill_spt":
        """open-batch 전용 확인 휴리스틱.

        import/호출 흐름:
        - `main.py`의 `command_simulate()`가 `env.run_with_policy(...)`를 호출한다.
        - `env.run_with_policy()`는 `CuttingSimulation.get_candidates()`에서 후보를 받는다.
        - action_space.mode가 `batch_open`이면 후보에 `open_batch`, `add_to_batch`, `close_batch`가 섞여 있다.

        선택 규칙:
        1. 가능한 `add_to_batch`가 있으면 먼저 선택해서 열린 batch를 채운다.
        2. add할 batch가 없으면 새 `open_batch`를 선택한다.
        3. open/add가 없고 close만 남으면 `close_batch`를 선택한다.

        이 휴리스틱은 성능 최적화용이 아니라 DES 동작 검증용이다.
        목적은 "open -> add -> close" state transition이 실제로 되는지 확인하는 것이다.
        """

        # LINE-BY-LINE: `close_candidates`에 `[candidate for candidate in candidates if candidate.action_kind == "close_batch"]` 결과를 저장합니다. 의미/사용: `close_candidates` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        close_candidates = [candidate for candidate in candidates if candidate.action_kind == "close_batch"]
        # LINE-BY-LINE: `add_candidates`에 `[candidate for candidate in candidates if candidate.action_kind == "add_to_batch"]` 결과를 저장합니다. 의미/사용: `add_candidates` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        add_candidates = [candidate for candidate in candidates if candidate.action_kind == "add_to_batch"]
        # LINE-BY-LINE: `open_candidates`에 `[candidate for candidate in candidates if candidate.action_kind == "open_batch"]` 결과를 저장합니다. 의미/사용: `open_candidates` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        open_candidates = [candidate for candidate in candidates if candidate.action_kind == "open_batch"]

        # LINE-BY-LINE: 조건 `add_candidates`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if add_candidates:
            # LINE-BY-LINE: 호출자에게 `min(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return min(
                # LINE-BY-LINE: `min(...)` 호출에 `add_candidates` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                add_candidates,
                # LINE-BY-LINE: `key`에 `lambda c: (` 결과를 저장합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                key=lambda c: (
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.estimated_minutes` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    c.estimated_minutes,
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.soft_penalty` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    c.soft_penalty,
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.batch_id or ""` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    c.batch_id or "",
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.job.job_id` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    c.job.job_id,
                ),
            )
        # LINE-BY-LINE: 조건 `open_candidates`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if open_candidates:
            # LINE-BY-LINE: 호출자에게 `min(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return min(
                # LINE-BY-LINE: `min(...)` 호출에 `open_candidates` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                open_candidates,
                # LINE-BY-LINE: `key`에 `lambda c: (` 결과를 저장합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                key=lambda c: (
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.estimated_minutes` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    c.estimated_minutes,
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.soft_penalty` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    c.soft_penalty,
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.machine.machine_id` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    c.machine.machine_id,
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.job.job_id` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    c.job.job_id,
                ),
            )
        # LINE-BY-LINE: 조건 `close_candidates`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if close_candidates:
            # LINE-BY-LINE: 호출자에게 `min(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return min(
                # LINE-BY-LINE: `min(...)` 호출에 `close_candidates` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                close_candidates,
                # LINE-BY-LINE: `key`에 `lambda c: (` 결과를 저장합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                key=lambda c: (
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.estimated_minutes` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    c.estimated_minutes,
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.soft_penalty` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    c.soft_penalty,
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.batch_id or ""` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    c.batch_id or "",
                ),
            )

    if rule_name == "balanced_batch":
        """batch_open 전용 균형형 휴리스틱.

        목적:
        - `batch_fill_spt`처럼 가능한 add를 무조건 먼저 고르지 않는다.
        - 각 후보가 지금 batch로 확정됐다고 가정한 projected 지표를 비교한다.
        - hard 제약은 여기서 새로 만들지 않고 `get_candidates()` 결과만 사용한다.

        점수에 들어가는 값:
        - 예상 makespan
        - 설비별 누적 부하 max-min
        - Bay별 누적 부하 max-min
        - 1~2개 batch close를 허용하되 single close 남발을 막는 약한 underfill penalty
        """

        return min(candidates, key=lambda candidate: _balanced_batch_score(simulation, candidate))

    if rule_name == "balanced_batch_count":
        """batch_open 전용 작업 개수 균형형 휴리스틱.

        목적:
        - 처리시간 합이 아니라 설비/Bay별 할당 W/O 개수 max-min을 우선 줄인다.
        - makespan은 count imbalance가 같은 후보의 tie-breaker로 사용한다.
        - hard 제약은 기존 `get_candidates()`가 통과시킨 후보만 사용한다.
        """

        return min(candidates, key=lambda candidate: _balanced_batch_count_score(simulation, candidate))

    # LINE-BY-LINE: 조건 `rule_name == "spt"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if rule_name == "spt":
        # 가장 짧은 처리시간 우선.
        # 다만 soft penalty가 낮은 후보를 약하게 선호합니다.
        # 같은 처리시간이면 priority가 높은 W/O를 먼저 선택한다.
        # LINE-BY-LINE: 호출자에게 `min(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return min(
            # LINE-BY-LINE: `min(...)` 호출에 `candidates` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            candidates,
            # LINE-BY-LINE: `key`에 `lambda c: (` 결과를 저장합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            key=lambda c: (
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.estimated_minutes` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                c.estimated_minutes,
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.soft_penalty` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                c.soft_penalty,
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `-c.job.priority_weight` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                -c.job.priority_weight,
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.job.job_id` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                c.job.job_id,
            ),
        )

    # LINE-BY-LINE: 조건 `rule_name == "priority"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if rule_name == "priority":
        # 숫자가 클수록 더 중요한 작업이라고 가정합니다.
        # 따라서 정렬 key에서는 -priority_weight를 사용합니다.
        # LINE-BY-LINE: 호출자에게 `min(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return min(
            # LINE-BY-LINE: `min(...)` 호출에 `candidates` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            candidates,
            # LINE-BY-LINE: `key`에 `lambda c: (` 결과를 저장합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            key=lambda c: (
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `-c.job.priority_weight` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                -c.job.priority_weight,
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.soft_penalty` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                c.soft_penalty,
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.estimated_minutes` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                c.estimated_minutes,
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.job.job_id` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                c.job.job_id,
            ),
        )

    # LINE-BY-LINE: 조건 `rule_name == "load_balance"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if rule_name == "load_balance":
        # 현재 누적 부하가 낮은 설비를 우선 사용해서
        # 설비 간 workload가 한쪽으로 치우치지 않게 유도합니다.
        # 목적은 optimal 해를 찾는 것이 아니라 baseline 비교값을 만드는 것이다.
        # LINE-BY-LINE: 호출자에게 `min(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return min(
            # LINE-BY-LINE: `min(...)` 호출에 `candidates` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            candidates,
            # LINE-BY-LINE: `key`에 `lambda c: (` 결과를 저장합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            key=lambda c: (
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `simulation.machine_loads[c.machine.machine_id]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                simulation.machine_loads[c.machine.machine_id],
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.soft_penalty` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                c.soft_penalty,
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `c.estimated_minutes` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                c.estimated_minutes,
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `-c.job.priority_weight` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                -c.job.priority_weight,
            ),
        )

    # LINE-BY-LINE: `ValueError(f"unknown heuristic: {rule_name}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
    raise ValueError(f"unknown heuristic: {rule_name}")
