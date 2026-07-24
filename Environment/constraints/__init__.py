"""제약 패키지 공개 API."""

# LINE-BY-LINE: `.base` 모듈에서 `ConstraintContext, ConstraintResult`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .base import ConstraintContext, ConstraintResult
# LINE-BY-LINE: `.registry` 모듈에서 `CandidateConstraintBundle, ConstraintManager`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .registry import CandidateConstraintBundle, ConstraintManager

# LINE-BY-LINE: `__all__`에 `["ConstraintContext", "ConstraintResult", "ConstraintManager", "CandidateConstraintBundle"]` 결과를 저장합니다. 의미/사용: `__all__` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
__all__ = ["ConstraintContext", "ConstraintResult", "ConstraintManager", "CandidateConstraintBundle"]
