"""택트타임 분석/추정.

실제 데이터가 들어오면 이 파일에서 아래 역할을 맡게 됩니다.
- 장비별 실적 데이터 분석
- 작업 특성별 표준 택트타임 산출
- 보정계수 계산
- 설비-작업 조합별 처리시간 추정
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class TactTimeRecord:
    """실적 데이터 1건을 표현하는 예시 자료형."""

    machine_type: str
    family: str
    thickness: float
    plate_length: float
    actual_minutes: float


class TactTimeEstimator:
    """택트타임 추정기.

    현재는 간단한 통계 기반 placeholder이지만,
    인터페이스는 실제 추정기처럼 유지합니다.
    """

    def __init__(self):
        # 실제 구현에서는 여기서 통계량, 회귀계수, 모델 객체 등을 보관하면 됩니다.
        self.fitted = False
        self.summary = {}

    def fit(self, records: Iterable[TactTimeRecord]) -> None:
        """실적 데이터를 받아 기본 통계량을 계산합니다.

        현재는 평균 수준의 요약만 저장합니다.
        """

        records = list(records)
        if not records:
            self.summary = {"count": 0}
            self.fitted = True
            return

        avg_minutes = sum(record.actual_minutes for record in records) / len(records)
        self.summary = {
            "count": len(records),
            "average_actual_minutes": avg_minutes,
        }
        self.fitted = True

    def estimate(self, job: Dict, machine: Dict) -> float:
        """작업-설비 조합의 처리시간을 추정합니다.

        현재는 실제 모델이 없으므로 매우 단순한 placeholder만 둡니다.
        향후에는 아래 항목을 반영하면 됩니다.
        - 장비 타입
        - 계열
        - 두께
        - 길이 / 면적
        - 셋업 조건
        """

        base_minutes = float(job.get("base_cut_minutes", 60.0))
        speed_factor = float(machine.get("cut_speed_factor", 1.0))
        return base_minutes * speed_factor
