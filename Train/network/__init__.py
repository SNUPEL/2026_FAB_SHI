"""Train network 패키지 공개 API."""

# Phase 1/2 scorer가 공유하는 최소 MLP builder만 공개합니다.
from .mlp import build_mlp

__all__ = ["build_mlp"]
