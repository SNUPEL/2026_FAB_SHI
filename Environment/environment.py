"""Gym 스타일과 유사한 reset/step 인터페이스를 제공하는 환경 래퍼.

이 파일은 "입력 시나리오"와 "실제 시뮬레이션 코어" 사이의 연결 계층입니다.

즉:
- YAML -> 내부 데이터 객체 변환
- 외부에 reset / step 제공
를 담당합니다.
"""

from typing import Dict

from .data import AuxiliaryResource, DownstreamBay, Job, Machine
from .simulation import CuttingSimulation


class CuttingShopEnvironment:
    """절단 공정 스케줄링 환경.

    역할:
    - 시나리오 파일을 읽어서 내부 데이터 구조로 변환
    - simulation 객체를 감싸서 외부에 reset/step 형태로 제공
    """

    def __init__(self, config: Dict, scenario: Dict):
        self.config = config
        # 시나리오의 raw dict를 내부 dataclass 구조로 바꿉니다.
        self.jobs = self._build_jobs(scenario["jobs"])
        self.machines = self._build_machines(scenario["machines"])
        self.bays = self._build_bays(scenario["bays"])
        self.resources = self._build_resources(scenario.get("resources", []))

        # 현재는 위치 정보만 보관합니다.
        # 설비 크기, 이동 거리, 상세 레이아웃 모델은 이후 데이터가 들어오면 확장합니다.
        self.layout = {
            machine["machine_id"]: tuple(machine.get("position", (0.0, 0.0))) for machine in scenario["machines"]
        }

        self.simulation = CuttingSimulation(
            jobs=self.jobs,
            machines=self.machines,
            bays=self.bays,
            resources=self.resources,
            layout=self.layout,
            config=self.config,
        )

    def _build_jobs(self, raw_jobs):
        """시나리오의 작업 정의를 내부 Job 객체로 변환합니다."""

        return {
            job["job_id"]: Job(
                job_id=job["job_id"],
                family=job["family"],
                thickness=job["thickness"],
                plate_length=job["plate_length"],
                downstream_bay=job["downstream_bay"],
                priority_weight=job["priority_weight"],
                base_stage_minutes=job["base_stage_minutes"],
                due_date_minutes=job.get("due_date_minutes"),
                preferred_machine_types=tuple(job.get("preferred_machine_types", [])),
                required_resource_ids=tuple(job.get("required_resource_ids", [])),
            )
            for job in raw_jobs
        }

    def _build_machines(self, raw_machines):
        """시나리오의 설비 정의를 내부 Machine 객체로 변환합니다."""

        return {
            machine["machine_id"]: Machine(
                machine_id=machine["machine_id"],
                machine_type=machine["machine_type"],
                enabled=machine["enabled"],
                eligible_families=machine["eligible_families"],
                min_thickness=machine["min_thickness"],
                max_thickness=machine["max_thickness"],
                table_length_limit=machine["table_length_limit"],
                cut_speed_factor=machine["cut_speed_factor"],
                daily_capacity_minutes=machine["daily_capacity_minutes"],
                parallel_capacity=int(machine.get("parallel_capacity", 1)),
                required_resource_ids=tuple(machine.get("required_resource_ids", [])),
                position=tuple(machine.get("position", (0.0, 0.0))),
            )
            for machine in raw_machines
        }

    def _build_bays(self, raw_bays):
        """시나리오의 후공정 베이 정의를 내부 객체로 변환합니다."""

        return {
            bay["bay_id"]: DownstreamBay(
                bay_id=bay["bay_id"],
                priority_rank=bay["priority_rank"],
                capacity_limit=bay["capacity_limit"],
                transfer_time_minutes=float(bay.get("transfer_time_minutes", 0.0)),
                release_delay_minutes=(
                    None if bay.get("release_delay_minutes") is None else float(bay.get("release_delay_minutes"))
                ),
            )
            for bay in raw_bays
        }

    def _build_resources(self, raw_resources):
        """시나리오의 보조자원 정의를 내부 객체로 변환합니다."""

        return {
            resource["resource_id"]: AuxiliaryResource(
                resource_id=resource["resource_id"],
                capacity=int(resource["capacity"]),
            )
            for resource in raw_resources
        }

    def reset(self):
        """환경 초기화.

        반환 형식은 Gym 스타일을 따라
        `(observation, info)` 형태로 둡니다.
        """

        self.simulation.reset()
        return self.simulation.build_observation(), {"message": "reset complete"}

    def get_action_candidates(self):
        """현재 가능한 action 후보 목록을 반환합니다."""

        return self.simulation.get_candidates()

    def step(self, action_id: str):
        """선택한 action을 환경에 적용합니다."""

        return self.simulation.step_action(action_id)

    def run_with_policy(self, policy_fn):
        """휴리스틱 또는 간단 정책 함수로 한 에피소드를 끝까지 실행합니다."""

        return self.simulation.run(policy_fn)
