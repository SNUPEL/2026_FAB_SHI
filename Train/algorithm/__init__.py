"""학습 알고리즘 팩토리."""

# LINE-BY-LINE: `.factory` 모듈에서 `build_trainer`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .factory import build_trainer

# LINE-BY-LINE: `__all__`에 `["build_trainer"]` 결과를 저장합니다. 의미/사용: `__all__` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
__all__ = ["build_trainer"]
