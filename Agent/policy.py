"""정책 에이전트 추상 자리.

현재 학습은 Train/algorithm 아래 실제 구현을 사용합니다.
이 파일은 향후 별도 inference wrapper가 필요할 때 확장합니다.
"""


# LINE-BY-LINE: `PolicyAgent` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 프로젝트의 해당 모듈에서 재사용됩니다.
class PolicyAgent:
    """가장 단순한 baseline 정책 에이전트.

    실제 학습 정책은 `Train/algorithm` + `Train/network`에서 동작합니다.
    이 클래스는 "정책 에이전트라는 객체를 나중에 따로 두고 싶을 때"
    확장하기 위한 placeholder입니다.

    주의:
    - 이 클래스는 연구용 baseline 자리일 뿐이다.
    - 데이터 오류나 제약 오류를 조용히 대체하는 fallback 용도가 아니다.
    """

    # LINE-BY-LINE: `select_action(self, candidates, simulation)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다. 예: action_id는 `job_001@PLS21` 또는 `open_batch:...` 형태.
    def select_action(self, candidates, simulation):
        """첫 번째 후보를 그대로 반환한다.

        입력:
        - candidates: 이미 hard constraint를 통과한 action 후보 목록
        - simulation: 현재 `CuttingSimulation` 객체

        출력:
        - 후보가 있으면 candidates[0]
        - 후보가 없으면 None
        """

        # LINE-BY-LINE: 조건 `not candidates`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not candidates:
            # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return None
        # LINE-BY-LINE: 호출자에게 `candidates[0]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return candidates[0]
