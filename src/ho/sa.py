from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional

import numpy as np

from moves import MoveOperator


@dataclass(frozen=True)
class SASchedule:
    t0: float
    alpha: float
    t_min: float = 1e-6


def run_sa(
    instance,
    solution_start,
    steps: Optional[int],
    schedule: SASchedule,
    move_operator: MoveOperator,
    rng: Optional[np.random.Generator] = None,
    max_steps: Optional[int] = None,
    stall_steps: int = 0,
    log_interval: int = 0,
    log_prefix: str = "",
) -> tuple[Any, float, Dict[str, float]]:
    if rng is None:
        rng = np.random.default_rng()
    if steps is None:
        if max_steps is None:
            raise ValueError("max_steps must be set when steps is None.")
        steps = max_steps
    current = solution_start.copy()
    best = current.copy()
    best_cost = float(current.cost)
    accepted = 0
    deltas = []
    t = float(schedule.t0)
    start_time = perf_counter()
    no_improve = 0
    for step in range(steps):
        move = move_operator.sample(current, rng)
        if not move_operator.is_legal(current, move):
            t = max(schedule.t_min, t * schedule.alpha)
            continue
        delta = float(move_operator.delta(instance, current, move))
        accept = delta <= 0 or rng.random() < np.exp(-delta / max(t, 1e-12))
        if accept:
            move_operator.apply(current, move)
            current.cost += delta
            accepted += 1
            if current.cost < best_cost:
                best = current.copy()
                best_cost = float(current.cost)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        deltas.append(delta)
        t = max(schedule.t_min, t * schedule.alpha)
        if t <= schedule.t_min:
            break
        if stall_steps > 0 and no_improve >= stall_steps:
            break
        if log_interval > 0 and (step + 1) % log_interval == 0:
            prefix = f"{log_prefix} " if log_prefix else ""
            print(
                f"{prefix}[sa] step={step + 1} best_improve={solution_start.cost - best_cost:.4f} "
                f"temp={t:.6f} accept_rate={accepted / max(1, step + 1):.4f}",
                flush=True,
            )
    elapsed = perf_counter() - start_time
    avg_delta = float(np.mean(deltas)) if deltas else 0.0
    accept_rate = float(accepted / max(1, steps))
    stats = {
        "accept_rate": accept_rate,
        "avg_delta": avg_delta,
        "best_improve": float(solution_start.cost - best_cost),
        "final_temp": t,
        "time": float(elapsed),
        "steps": float(len(deltas)),
    }
    return best, best_cost, stats
