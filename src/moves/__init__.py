from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Move:
    kind: str
    params: tuple


@dataclass(frozen=True)
class MoveOperator:
    sample: Callable
    apply: Callable
    delta: Callable
    is_legal: Callable


__all__ = ["Move", "MoveOperator"]
