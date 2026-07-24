"""Phase 1 블록-Bay pair-pointer 패키지.

공개 심볼을 여기서 re-export하지 않는다. eager import는 `pair_self_labeling`과
`pointer_policy`를 통해 torch를 끌어오므로, torch가 필요 없는 CLI
(`phase1-plan-multi-series`, `generate-phase1-blocks`)까지 약 1.1초를 부담하게 된다.
호출부는 `from Phase1.heuristics import ...`처럼 서브모듈을 직접 import한다.
"""

__all__: list[str] = []
