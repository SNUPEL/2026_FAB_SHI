"""정책 에이전트 추상 자리.

현재 학습은 Train/algorithm 아래 실제 구현을 사용합니다.
이 파일은 향후 별도 inference wrapper가 필요할 때 확장합니다.
"""


class PolicyAgent:
    """가장 단순한 fallback 정책 에이전트.

    실제 학습 정책은 `Train/algorithm` + `Train/network`에서 동작합니다.
    이 클래스는 "정책 에이전트라는 객체를 나중에 따로 두고 싶을 때"
    확장하기 위한 placeholder입니다.
    """

    def select_action(self, candidates, simulation):
        if not candidates:
            return None
        return candidates[0]
