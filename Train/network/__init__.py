"""Train network 패키지 공개 API."""

# LINE-BY-LINE: `.feature_builder` 모듈에서 `build_tensor_observation`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .feature_builder import build_tensor_observation
# LINE-BY-LINE: `.policy_value` 모듈에서 `CandidateGraphPolicyValueNet`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .policy_value import CandidateGraphPolicyValueNet

# LINE-BY-LINE: `__all__`에 `["build_tensor_observation", "CandidateGraphPolicyValueNet"]` 결과를 저장합니다. 의미/사용: `__all__` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
__all__ = ["build_tensor_observation", "CandidateGraphPolicyValueNet"]
