"""Merged Phase 2 batch-machine 패키지.

공개 심볼을 여기서 re-export하지 않는다. `Phase1/__init__.py`와 같은 이유로
eager import가 torch를 끌어온다. 호출부는 `from Phase2.merged import ...`처럼
서브모듈을 직접 import한다.
"""

__all__: list[str] = []
