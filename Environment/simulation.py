"""절단 공정 스케줄링 시뮬레이션 코어.

이 파일이 사실상 환경의 중심입니다.

역할:
- 현재 시뮬레이션 시간이 어디인지 관리
- 어떤 job-machine pair가 가능한지 계산
- 하드/소프트 제약을 평가
- action을 적용해서 상태를 전진
- observation을 RL 네트워크가 읽을 수 있는 dict로 변환

초보자 입장에서 가장 중요한 포인트는 두 가지입니다.

1. `get_candidates()`
   - 지금 선택 가능한 action 후보를 만드는 함수

2. `step_action(action_id)`
   - 선택한 action 하나를 실제로 적용하는 함수
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

from .constraints import ConstraintContext, ConstraintManager
from .constraints.registry import CandidateConstraintBundle
from .data import DownstreamBay, Job, Machine, ScheduledOperation
from .reward import compute_load_imbalance, compute_step_reward
from .state import SimulationState


@dataclass
class ActionCandidate:
    """현재 step에서 선택 가능한 action 1개.

    action_id는 외부에서 step(action_id)로 호출하기 위한 식별자입니다.
    """

    action_id: str
    job: Job
    machine: Machine
    estimated_minutes: float
    hard_reasons: List[str] = field(default_factory=list)
    soft_reasons: List[str] = field(default_factory=list)
    soft_penalty: float = 0.0


class CuttingSimulation:
    """PMSP형 절단 스케줄링 시뮬레이션.

    현재 버전의 핵심 설계:
    - action = 작업-설비 쌍 1개
    - 하드 제약은 마스킹
    - 소프트 제약은 penalty
    - 가능한 액션이 없으면 다음 완료 시점까지 자동 전진
    """

    def __init__(
        self,
        jobs: Dict[str, Job],
        machines: Dict[str, Machine],
        bays: Dict[str, DownstreamBay],
        layout,
        config: Dict,
    ):
        self.jobs = jobs
        self.machines = machines
        self.bays = bays
        self.layout = layout
        self.config = config
        self.minutes_per_day = int(config["simulation"].get("minutes_per_day", 1440))
        self.start_date = datetime.strptime(config["simulation"].get("start_date", "2026-01-01"), "%Y-%m-%d").date()
        self.category_flags = config["constraints"].get("categories", {})
        self.override_config = config["constraints"].get("overrides", {})
        self.family_to_index = {name: idx for idx, name in enumerate(sorted({job.family for job in jobs.values()}))}
        self.machine_type_to_index = {
            name: idx for idx, name in enumerate(sorted({machine.machine_type for machine in machines.values()}))
        }
        self.bay_to_index = {name: idx for idx, name in enumerate(sorted(bays.keys()))}
        self.constraint_manager = ConstraintManager(
            hard_enabled=config["constraints"]["hard_enabled"],
            soft_enabled=config["constraints"]["soft_enabled"],
            soft_weights=config["constraints"]["soft_penalty_weights"],
            category_flags=self.category_flags,
        )
        self.state = self._build_initial_state()

    def _build_initial_state(self) -> SimulationState:
        """에피소드 시작 상태를 만듭니다."""

        return SimulationState(
            current_time=0.0,
            unscheduled_jobs=set(self.jobs.keys()),
            machine_available_at={machine_id: 0.0 for machine_id in self.machines},
            machine_loads={machine_id: 0.0 for machine_id in self.machines},
            downstream_loads={bay_id: 0 for bay_id in self.bays},
            machine_daily_loads={},
            scheduled_job_count_by_day={},
        )

    @staticmethod
    def _normalize_date_key(date_key: str) -> str:
        """override key를 YYYY-MM-DD 형태로 통일합니다."""

        if len(date_key) == 8 and date_key.isdigit():
            return f"{date_key[0:4]}-{date_key[4:6]}-{date_key[6:8]}"
        return date_key

    def _current_day_index(self, at_time: Optional[float] = None) -> int:
        """현재 시각이 시작일로부터 며칠째인지 계산합니다."""

        target_time = self.state.current_time if at_time is None else at_time
        return int(target_time // self.minutes_per_day)

    def _day_key_from_index(self, day_index: int) -> str:
        """day index를 실제 날짜 문자열로 바꿉니다."""

        return (self.start_date + timedelta(days=day_index)).isoformat()

    def _current_day_key(self, at_time: Optional[float] = None) -> str:
        """현재 시각 기준 날짜 key를 반환합니다."""

        return self._day_key_from_index(self._current_day_index(at_time))

    @staticmethod
    def _minute_of_day(at_time: float, minutes_per_day: int) -> int:
        """현재 시각을 하루 기준 분(minute-of-day)로 변환합니다."""

        return int(at_time % minutes_per_day)

    @staticmethod
    def _parse_time_string(time_string: str) -> int:
        """HH:MM 문자열을 분 단위 정수로 바꿉니다."""

        hour, minute = time_string.split(":")
        return int(hour) * 60 + int(minute)

    def _is_in_time_window(self, minute_of_day: int, window: str) -> bool:
        """현재 분이 특정 시간 구간 안에 들어오는지 확인합니다.

        지원 형식:
        - 12:00-13:00
        - 22:00-02:00  (자정 넘김)
        """

        start_raw, end_raw = window.split("-")
        start_minute = self._parse_time_string(start_raw)
        end_minute = self._parse_time_string(end_raw)

        if start_minute <= end_minute:
            return start_minute <= minute_of_day < end_minute
        return minute_of_day >= start_minute or minute_of_day < end_minute

    def _matches_date_spec(self, day_key: str, date_spec: str) -> bool:
        """날짜가 단일 날짜 또는 범위 지정과 일치하는지 확인합니다.

        지원 형식:
        - 2026-01-01
        - 20260101
        - 2026-01-01~2026-01-03
        - 20260101..20260103
        """

        normalized_spec = self._normalize_date_key(date_spec)
        if "~" in normalized_spec:
            start_key, end_key = normalized_spec.split("~", 1)
            return start_key <= day_key <= self._normalize_date_key(end_key)
        if ".." in normalized_spec:
            start_key, end_key = normalized_spec.split("..", 1)
            return start_key <= day_key <= self._normalize_date_key(end_key)
        return day_key == normalized_spec

    def _day_matches_any_spec(self, day_key: str, date_specs: List[str]) -> bool:
        """현재 날짜가 date spec 목록 중 하나와 일치하는지 확인합니다."""

        return any(self._matches_date_spec(day_key, str(spec)) for spec in date_specs)

    def _ensure_day_state(self, day_key: str) -> None:
        """해당 날짜에 대한 상태 dict가 없으면 기본값으로 만듭니다."""

        if day_key not in self.state.machine_daily_loads:
            self.state.machine_daily_loads[day_key] = {
                machine_id: 0.0 for machine_id in self.machines
            }
        if day_key not in self.state.scheduled_job_count_by_day:
            self.state.scheduled_job_count_by_day[day_key] = 0

    def _get_daily_override_map(self, flag_name: str, map_name: str) -> Dict:
        """override 활성화 여부와 실제 override map을 함께 읽습니다."""

        if not self.override_config.get(flag_name, False):
            return {}
        return self.override_config.get(map_name, {}) or {}

    def _get_machine_enabled_flag(self, machine_id: str) -> bool:
        """기본 enabled + 날짜별 machine enable override를 반영합니다."""

        base_enabled = bool(self.machines[machine_id].enabled)
        override_map = self._get_daily_override_map(
            "enable_daily_machine_enable_overrides",
            "daily_machine_enable_overrides",
        )
        current_day_key = self._current_day_key()
        normalized = {
            self._normalize_date_key(day_key): value
            for day_key, value in override_map.items()
        }
        day_override = normalized.get(current_day_key, {})
        if machine_id in day_override:
            return bool(day_override[machine_id])
        return base_enabled

    def _is_global_calendar_open(self) -> tuple[bool, str]:
        """현재 시각에 공장 전체가 가동 가능한지 판단합니다."""

        calendar_cfg = self.config.get("calendar", {})
        current_day_key = self._current_day_key()
        minute_of_day = self._minute_of_day(self.state.current_time, self.minutes_per_day)

        if calendar_cfg.get("enable_holidays_off", False):
            holiday_specs = calendar_cfg.get("holidays_off", []) or []
            if self._day_matches_any_spec(current_day_key, holiday_specs):
                return False, f"holiday off on {current_day_key}"

        if calendar_cfg.get("enable_half_day_off", False):
            half_day_specs = calendar_cfg.get("half_day_off", []) or []
            half_day_window = calendar_cfg.get("half_day_off_window", "12:00-24:00")
            if self._day_matches_any_spec(current_day_key, half_day_specs):
                if self._is_in_time_window(minute_of_day, half_day_window):
                    return False, f"half day off active ({half_day_window})"

        if calendar_cfg.get("enable_lunch_break", False):
            lunch_window = calendar_cfg.get("lunch_break")
            lunch_dates = calendar_cfg.get("lunch_break_dates", []) or []
            lunch_applies = (not lunch_dates) or self._day_matches_any_spec(current_day_key, lunch_dates)
            if lunch_window and lunch_applies and self._is_in_time_window(minute_of_day, lunch_window):
                return False, f"lunch break active ({lunch_window})"

        if calendar_cfg.get("enable_global_shutdown_windows", False):
            shutdown_map = calendar_cfg.get("global_shutdown_windows", {}) or {}
            normalized = {
                self._normalize_date_key(day_key): windows
                for day_key, windows in shutdown_map.items()
            }
            day_windows = normalized.get(current_day_key, [])
            for window in day_windows:
                if self._is_in_time_window(minute_of_day, str(window)):
                    return False, f"global shutdown active ({window})"

        return True, ""

    def _machine_windows_for_day(self, config_key: str, machine_id: str) -> List[str]:
        """설비별 운영/정지 시간 구간을 읽습니다.

        지원 형태:
        - machine_operating_windows:
            default:
              laser_01: ["08:00-18:00"]
            "2026-01-01":
              laser_01: ["10:00-16:00"]

        우선순위:
        1. 해당 날짜 key
        2. default
        """

        calendar_cfg = self.config.get("calendar", {})
        raw_map = calendar_cfg.get(config_key, {}) or {}
        current_day_key = self._current_day_key()

        normalized = {}
        for day_key, machine_map in raw_map.items():
            if day_key == "default":
                normalized["default"] = machine_map
            else:
                normalized[self._normalize_date_key(day_key)] = machine_map

        if current_day_key in normalized and machine_id in normalized[current_day_key]:
            windows = normalized[current_day_key][machine_id]
        else:
            windows = normalized.get("default", {}).get(machine_id, [])

        if isinstance(windows, str):
            return [windows]
        return [str(window) for window in windows]

    def _machine_calendar_status(self, machine_id: str) -> tuple[bool, str]:
        """현재 시각에 특정 설비가 계획된 운영시간 안에 있는지 판단합니다.

        이 함수는 "고장"이 아니라 "계획된 운영 캘린더"를 다룹니다.
        """

        calendar_cfg = self.config.get("calendar", {})
        current_day_key = self._current_day_key()
        minute_of_day = self._minute_of_day(self.state.current_time, self.minutes_per_day)

        if calendar_cfg.get("enable_machine_operating_windows", False):
            windows = self._machine_windows_for_day("machine_operating_windows", machine_id)
            if windows and not any(self._is_in_time_window(minute_of_day, window) for window in windows):
                return False, f"{machine_id} outside operating window on {current_day_key}"

        if calendar_cfg.get("enable_machine_shutdown_windows", False):
            windows = self._machine_windows_for_day("machine_shutdown_windows", machine_id)
            for window in windows:
                if self._is_in_time_window(minute_of_day, window):
                    return False, f"{machine_id} planned shutdown active ({window})"

        return True, ""

    def _machine_breakdown_status(self, machine_id: str) -> tuple[bool, str]:
        """현재 시각에 특정 설비가 고장/정지 상태인지 판단합니다."""

        calendar_cfg = self.config.get("calendar", {})
        if not calendar_cfg.get("enable_machine_breakdowns", False):
            return False, ""

        current_day_key = self._current_day_key()
        minute_of_day = self._minute_of_day(self.state.current_time, self.minutes_per_day)
        breakdown_map = calendar_cfg.get("machine_breakdowns", {}) or {}
        normalized = {
            self._normalize_date_key(day_key): machine_map
            for day_key, machine_map in breakdown_map.items()
        }
        day_breakdowns = normalized.get(current_day_key, {})
        machine_windows = day_breakdowns.get(machine_id)

        if not machine_windows:
            return False, ""

        # full_day / true / 00:00-24:00 같은 간단 표기도 허용합니다.
        if isinstance(machine_windows, str):
            if machine_windows in {"full_day", "FULL_DAY", "true", "True"}:
                return True, f"{machine_id} breakdown full day"
            machine_windows = [machine_windows]
        elif machine_windows is True:
            return True, f"{machine_id} breakdown full day"

        for window in machine_windows:
            if self._is_in_time_window(minute_of_day, str(window)):
                return True, f"{machine_id} breakdown active ({window})"

        return False, ""

    def _has_temporary_blocking_condition(self) -> bool:
        """현재 후보가 없는 이유가 "시간이 지나면 풀릴 가능성이 있는가"를 판단합니다.

        왜 필요한가:
        - 단순 PMSP라면 후보가 없을 때 다음 설비 완료 시점으로만 가도 됩니다.
        - 하지만 현재 프로젝트에는 점심시간, 휴무일, 설비 고장, 일일 용량 제한처럼
          "같은 설비라도 나중에는 다시 가능해질 수 있는" 제약이 있습니다.
        - 그래서 후보가 없는 순간을 무조건 dead-end로 처리하면 안 됩니다.

        현재는 아래 상황을 "임시 차단"으로 봅니다.
        - 공장 전체 휴무 / 점심 / shutdown window
        - 오늘 일일 전체 작업 수 제한 도달
        - 특정 설비의 현재 고장/정지
        - 특정 설비의 오늘 일일 용량 초과

        반대로, 아래는 임시 차단으로 보지 않습니다.
        - 계열 미일치
        - 두께 미일치
        - 정반 길이 미일치
        즉, 시간이 지나도 풀리지 않는 정적 제약은 여기서 제외합니다.
        """

        current_day_key = self._current_day_key()
        self._ensure_day_state(current_day_key)

        global_calendar_open, _ = self._is_global_calendar_open()
        if not global_calendar_open:
            return True

        daily_job_cap = self._get_daily_job_cap()
        if daily_job_cap is not None and self.state.scheduled_job_count_by_day[current_day_key] >= daily_job_cap:
            return True

        override_enabled = self.override_config.get("enable_daily_machine_enable_overrides", False)

        for machine_id in self.machines:
            if not self._is_machine_available(machine_id):
                continue

            machine_enabled_flag = self._get_machine_enabled_flag(machine_id)
            if not machine_enabled_flag and override_enabled:
                return True

            machine_calendar_open, _ = self._machine_calendar_status(machine_id)
            if not machine_calendar_open:
                return True

            breakdown_active, _ = self._machine_breakdown_status(machine_id)
            if breakdown_active:
                return True

            daily_load = self.state.machine_daily_loads[current_day_key][machine_id]
            daily_capacity = self._get_machine_daily_capacity_limit(machine_id)
            if daily_load >= daily_capacity:
                return True

        return False

    def _get_machine_daily_capacity_limit(self, machine_id: str) -> float:
        """기본 일일 용량 + 날짜별 machine capacity override를 반영합니다."""

        base_capacity = float(self.machines[machine_id].daily_capacity_minutes)
        override_map = self._get_daily_override_map(
            "enable_daily_machine_capacity_overrides",
            "daily_machine_capacity_overrides",
        )
        current_day_key = self._current_day_key()
        normalized = {
            self._normalize_date_key(day_key): value
            for day_key, value in override_map.items()
        }
        day_override = normalized.get(current_day_key, {})
        if machine_id in day_override:
            return float(day_override[machine_id])
        return base_capacity

    def _get_daily_job_cap(self) -> Optional[int]:
        """오늘 전체 작업 수 제한이 있으면 반환하고, 없으면 None을 반환합니다."""

        override_map = self._get_daily_override_map(
            "enable_daily_job_cap_overrides",
            "daily_job_cap_overrides",
        )
        current_day_key = self._current_day_key()
        normalized = {
            self._normalize_date_key(day_key): value
            for day_key, value in override_map.items()
        }
        if current_day_key in normalized:
            return int(normalized[current_day_key])
        return None

    @property
    def machine_loads(self) -> Dict[str, float]:
        """휴리스틱에서 현재 설비 부하를 쉽게 참조할 수 있게 합니다."""

        return self.state.machine_loads

    def reset(self) -> None:
        """새 에피소드 시작."""

        self.state = self._build_initial_state()
        self._ensure_day_state(self._current_day_key(at_time=0.0))
        self._advance_to_decision_epoch()

    def _is_machine_available(self, machine_id: str) -> bool:
        """현재 시각에 해당 설비가 비어 있는지 확인합니다."""

        return self.state.machine_available_at[machine_id] <= self.state.current_time

    def _build_constraint_context(self, job: Job, machine: Machine) -> ConstraintContext:
        """제약 평가에 필요한 문맥 객체를 만듭니다."""

        current_day_key = self._current_day_key()
        self._ensure_day_state(current_day_key)
        global_calendar_open, global_calendar_reason = self._is_global_calendar_open()
        machine_calendar_open, machine_calendar_reason = self._machine_calendar_status(machine.machine_id)
        machine_breakdown_active, machine_breakdown_reason = self._machine_breakdown_status(machine.machine_id)
        return ConstraintContext(
            job=job,
            machine=machine,
            state=self.state,
            jobs=self.jobs,
            machines=self.machines,
            bays=self.bays,
            layout=self.layout,
            config=self.config,
            current_day_key=current_day_key,
            current_day_index=self._current_day_index(),
            machine_enabled_flag=self._get_machine_enabled_flag(machine.machine_id),
            machine_daily_capacity_limit=self._get_machine_daily_capacity_limit(machine.machine_id),
            machine_daily_load=self.state.machine_daily_loads[current_day_key][machine.machine_id],
            scheduled_job_count_today=self.state.scheduled_job_count_by_day[current_day_key],
            daily_job_cap=self._get_daily_job_cap(),
            global_calendar_open=global_calendar_open,
            global_calendar_reason=global_calendar_reason,
            machine_calendar_open=machine_calendar_open,
            machine_calendar_reason=machine_calendar_reason,
            machine_breakdown_active=machine_breakdown_active,
            machine_breakdown_reason=machine_breakdown_reason,
        )

    def _evaluate_candidate(self, job: Job, machine: Machine) -> CandidateConstraintBundle:
        """후보 action 1개에 대해 하드/소프트 제약을 함께 평가합니다."""

        context = self._build_constraint_context(job, machine)
        return self.constraint_manager.evaluate_candidate(context)

    def get_candidates(self) -> List[ActionCandidate]:
        """현재 의사결정 시점에서 가능한 action 후보 목록을 만듭니다."""

        candidates: List[ActionCandidate] = []
        current_day_key = self._current_day_key()
        self._ensure_day_state(current_day_key)

        # 현재 비어 있는 설비들에 대해서만 후보를 만듭니다.
        for machine in self.machines.values():
            if not self._is_machine_available(machine.machine_id):
                continue

            for job_id in sorted(self.state.unscheduled_jobs):
                job = self.jobs[job_id]
                bundle = self._evaluate_candidate(job, machine)

                # 하드 제약 하나라도 실패하면 이 action은 후보에서 제거합니다.
                if not bundle.hard_passed:
                    continue

                candidates.append(
                    ActionCandidate(
                        action_id=f"{job.job_id}@{machine.machine_id}",
                        job=job,
                        machine=machine,
                        estimated_minutes=job.estimate_total_minutes(machine),
                        hard_reasons=bundle.hard_reasons,
                        soft_reasons=bundle.soft_reasons,
                        soft_penalty=bundle.soft_penalty,
                    )
                )

        return candidates

    def get_action_map(self) -> Dict[str, ActionCandidate]:
        """action_id -> candidate 매핑."""

        return {candidate.action_id: candidate for candidate in self.get_candidates()}

    def _current_completion_time(self) -> float:
        """현재까지의 예상 전체 완료 시점을 계산합니다."""

        return max(self.state.machine_available_at.values(), default=self.state.current_time)

    def get_makespan(self) -> float:
        """현재 스케줄의 makespan을 반환합니다.

        주의:
        - current_time은 "현재 의사결정 시점"입니다.
        - makespan은 "모든 설비 중 가장 늦은 완료 시각"입니다.
        최종 성능 평가는 makespan으로 보는 것이 맞습니다.
        """

        return self._current_completion_time()

    def _build_scheduled_operation(self, candidate: ActionCandidate) -> ScheduledOperation:
        """선택한 action으로부터 schedule 레코드를 만듭니다."""

        start_time = self.state.current_time
        finish_time = start_time + candidate.estimated_minutes

        return ScheduledOperation(
            job_id=candidate.job.job_id,
            machine_id=candidate.machine.machine_id,
            start_time=start_time,
            finish_time=finish_time,
            downstream_bay=candidate.job.downstream_bay,
            estimated_minutes=candidate.estimated_minutes,
            stage_minutes={
                "setup": candidate.job.base_stage_minutes.get("setup", 0.0),
                "cut": candidate.job.base_stage_minutes.get("cut", 0.0) * candidate.machine.cut_speed_factor,
                "finish": candidate.job.base_stage_minutes.get("finish", 0.0),
            },
        )

    def _apply_operation(self, operation: ScheduledOperation) -> None:
        """선택 결과를 상태에 반영합니다."""

        operation_day_key = self._current_day_key(at_time=operation.start_time)
        self._ensure_day_state(operation_day_key)
        self.state.schedule.append(operation)
        self.state.unscheduled_jobs.remove(operation.job_id)
        self.state.machine_available_at[operation.machine_id] = operation.finish_time
        self.state.machine_loads[operation.machine_id] += operation.estimated_minutes
        self.state.machine_daily_loads[operation_day_key][operation.machine_id] += operation.estimated_minutes
        self.state.scheduled_job_count_by_day[operation_day_key] += 1
        self.state.downstream_loads[operation.downstream_bay] += 1

    def advance_time(self) -> bool:
        """다음 설비 완료 시점으로 시간을 넘깁니다."""

        future_times = sorted(
            {
                available_time
                for available_time in self.state.machine_available_at.values()
                if available_time > self.state.current_time
            }
        )
        if not future_times:
            return False
        self.state.current_time = future_times[0]
        self._ensure_day_state(self._current_day_key())
        return True

    def _advance_to_decision_epoch(self) -> None:
        """가능한 action이 나올 때까지 다음 의사결정 시점으로 전진합니다.

        전진 규칙:
        1. 후보가 있으면 즉시 멈춤
        2. 후보가 없고, 원인이 "바쁜 설비"면 다음 완료 시점으로 점프
        3. 후보가 없고, 원인이 "캘린더/고장/일일 용량"이면 1분씩 전진
           - 이 방식은 느릴 수 있지만 초보자에게 가장 직관적입니다.
           - 또한 현재 구현은 preemption이 없기 때문에,
             시작 가능 시점만 찾으면 충분합니다.
        4. 후보가 없고, 시간 지나도 안 풀리는 정적 제약뿐이면 중단
        """

        horizon_minutes = float(self.config["simulation"]["horizon_minutes"])

        while self.state.unscheduled_jobs:
            if self.get_candidates():
                return

            if not self.config["simulation"]["auto_advance_time"]:
                return

            if self.state.current_time >= horizon_minutes:
                return

            if self._has_temporary_blocking_condition():
                self.state.current_time = min(self.state.current_time + 1.0, horizon_minutes)
                self._ensure_day_state(self._current_day_key())
                continue

            moved = self.advance_time()
            if not moved:
                return

    def build_observation(self) -> Dict:
        """현재 상태를 외부 정책이 읽기 쉬운 dict 형태로 반환합니다."""

        candidates = self.get_candidates()
        current_day_key = self._current_day_key()
        self._ensure_day_state(current_day_key)
        total_jobs = len(self.jobs)
        machine_count = max(len(self.machines), 1)
        average_machine_load = sum(self.state.machine_loads.values()) / machine_count
        average_machine_capacity = sum(
            self._get_machine_daily_capacity_limit(machine.machine_id)
            for machine in self.machines.values()
        ) / machine_count
        max_downstream_ratio = 0.0
        for bay_id, bay in self.bays.items():
            ratio = self.state.downstream_loads[bay_id] / max(bay.capacity_limit, 1)
            max_downstream_ratio = max(max_downstream_ratio, ratio)

        return {
            "current_time": self.state.current_time,
            "current_day_key": current_day_key,
            "makespan": self.get_makespan(),
            "remaining_job_count": len(self.state.unscheduled_jobs),
            "total_job_count": total_jobs,
            "remaining_jobs": sorted(self.state.unscheduled_jobs),
            "machine_available_at": dict(self.state.machine_available_at),
            "machine_loads": dict(self.state.machine_loads),
            "downstream_loads": dict(self.state.downstream_loads),
            "env_features": {
                "current_time_norm": self.state.current_time / max(self.config["simulation"]["horizon_minutes"], 1),
                "makespan_norm": self.get_makespan() / max(self.config["simulation"]["horizon_minutes"], 1),
                "remaining_job_ratio": len(self.state.unscheduled_jobs) / max(total_jobs, 1),
                "available_machine_ratio": sum(self._is_machine_available(machine_id) for machine_id in self.machines) / machine_count,
                "calendar_open_flag": 1.0 if self._is_global_calendar_open()[0] else 0.0,
                "machine_calendar_open_ratio": sum(
                    1
                    for machine_id in self.machines
                    if self._machine_calendar_status(machine_id)[0]
                ) / machine_count,
                "broken_machine_ratio": sum(
                    1
                    for machine_id in self.machines
                    if self._machine_breakdown_status(machine_id)[0]
                ) / machine_count,
                "average_machine_load_ratio": average_machine_load / max(average_machine_capacity, 1.0),
                "max_downstream_ratio": max_downstream_ratio,
                "available_action_ratio": len(candidates) / max(len(self.state.unscheduled_jobs) * machine_count, 1),
            },
            "vocab_sizes": {
                "family": len(self.family_to_index),
                "machine_type": len(self.machine_type_to_index),
                "bay": len(self.bay_to_index),
            },
            "available_actions": [
                {
                    "action_id": candidate.action_id,
                    "job_id": candidate.job.job_id,
                    "machine_id": candidate.machine.machine_id,
                    "machine_type": candidate.machine.machine_type,
                    "machine_type_index": self.machine_type_to_index[candidate.machine.machine_type],
                    "estimated_minutes": candidate.estimated_minutes,
                    "priority_weight": candidate.job.priority_weight,
                    "soft_penalty": candidate.soft_penalty,
                    "soft_reasons": candidate.soft_reasons,
                    "family": candidate.job.family,
                    "family_index": self.family_to_index[candidate.job.family],
                    "thickness": candidate.job.thickness,
                    "plate_length": candidate.job.plate_length,
                    "downstream_bay": candidate.job.downstream_bay,
                    "downstream_bay_index": self.bay_to_index[candidate.job.downstream_bay],
                    "downstream_priority_rank": self.bays[candidate.job.downstream_bay].priority_rank,
                    "downstream_load_ratio": self.state.downstream_loads[candidate.job.downstream_bay]
                    / max(self.bays[candidate.job.downstream_bay].capacity_limit, 1),
                    "machine_speed_factor": candidate.machine.cut_speed_factor,
                    "machine_load_ratio": self.state.machine_daily_loads[current_day_key][candidate.machine.machine_id]
                    / max(self._get_machine_daily_capacity_limit(candidate.machine.machine_id), 1.0),
                    "remaining_machine_capacity_ratio": max(
                        0.0,
                        self._get_machine_daily_capacity_limit(candidate.machine.machine_id)
                        - self.state.machine_daily_loads[current_day_key][candidate.machine.machine_id],
                    )
                    / max(self._get_machine_daily_capacity_limit(candidate.machine.machine_id), 1.0),
                    "due_date_slack_norm": (
                        0.0
                        if candidate.job.due_date_minutes is None
                        else (candidate.job.due_date_minutes - (self.state.current_time + candidate.estimated_minutes))
                        / max(self.config["simulation"]["horizon_minutes"], 1)
                    ),
                }
                for candidate in candidates
            ],
        }

    def step_action(self, action_id: str):
        """외부에서 action_id를 받아 1 step 진행합니다."""

        action_map = self.get_action_map()
        if action_id not in action_map:
            raise ValueError(f"invalid action_id: {action_id}")

        selected = action_map[action_id]

        # 보상 계산을 위해 step 전 상태를 저장합니다.
        before_completion = self._current_completion_time()
        before_imbalance = compute_load_imbalance(self.state.machine_loads)

        operation = self._build_scheduled_operation(selected)
        self._apply_operation(operation)

        # step 후 상태 기준 지표를 다시 계산합니다.
        after_completion = self._current_completion_time()
        after_imbalance = compute_load_imbalance(self.state.machine_loads)

        reward = compute_step_reward(
            before_completion=before_completion,
            after_completion=after_completion,
            before_imbalance=before_imbalance,
            after_imbalance=after_imbalance,
            priority_weight=selected.job.priority_weight,
            soft_penalty=selected.soft_penalty,
            reward_config=self.config["reward"],
        )

        self._advance_to_decision_epoch()
        terminated = len(self.state.unscheduled_jobs) == 0
        truncated = self.state.current_time > self.config["simulation"]["horizon_minutes"]
        observation = self.build_observation()
        info = {
            "selected_action": action_id,
            "job_id": selected.job.job_id,
            "machine_id": selected.machine.machine_id,
            "operation": operation,
            "soft_penalty": selected.soft_penalty,
            "soft_reasons": selected.soft_reasons,
        }
        return observation, reward, terminated, truncated, info

    def run(self, policy_fn: Callable[[List[ActionCandidate], "CuttingSimulation"], Optional[ActionCandidate]]) -> Dict:
        """휴리스틱 또는 간단 정책으로 에피소드를 끝까지 실행합니다."""

        self.reset()
        while self.state.unscheduled_jobs and self.state.current_time <= self.config["simulation"]["horizon_minutes"]:
            candidates = self.get_candidates()
            if not candidates:
                break

            selected = policy_fn(candidates, self)
            if selected is None:
                break

            self.step_action(selected.action_id)

        return {
            "current_time": self.state.current_time,
            "makespan": self.get_makespan(),
            "scheduled_jobs": len(self.state.schedule),
            "unscheduled_jobs": sorted(self.state.unscheduled_jobs),
            "downstream_loads": dict(self.state.downstream_loads),
            "machine_loads": dict(self.state.machine_loads),
            "schedule": self.state.schedule,
        }
