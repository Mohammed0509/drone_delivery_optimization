"""
Simulated Annealing for the drone delivery problem.

Move set
--------
At each iteration we sample one neighbour from a mix of stochastic and
local-search moves:
    - swap two customers (anywhere in the solution)
    - move a customer to a different position / drone
    - reverse a segment within a route (random 2-opt for diversification)
    - inter-drone swap
    - best-improvement 2-opt on one route (occasional intensification)
    - shake (small composition of moves; used as restart kick)

The first four are cheap stochastic moves used most of the time; the
best-improvement 2-opt is applied with a small probability and provides
a structured intensification step (route-level local search), giving
the standard hybrid SA + local search behaviour.

Acceptance
----------
Standard Metropolis: ``exp(-delta / T)`` for worsening moves. The
fitness includes the same soft penalty as the GA so the search may
temporarily walk through infeasible regions.

Cooling
-------
Geometric: T_{i+1} = alpha * T_i. The initial temperature is calibrated
from a small batch of random moves so that ~50% of worsening moves are
accepted at the start.
"""

from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import List, Sequence

from src.models.graph_based_model import nearest_neighbor
from src.utils.energy import (
    Instance,
    distance_matrix,
    load_instance,
    solution_energy,
    solution_feasible,
)
from src.metaheuristics import operators as ops
from src.metaheuristics.genetic_algorithm import fitness as ga_fitness
from src.metaheuristics.repair import repair


# --------------------------------------------------------------------------
# Configuration / result
# --------------------------------------------------------------------------

@dataclass
class SAConfig:
    iterations: int = 5000
    initial_temperature: float | None = None     # None = auto calibrate
    final_temperature: float = 0.5
    alpha: float = 0.995
    reheat_after: int = 1000                     # iterations w/o improvement
    penalty_weight: float = 50.0
    two_opt_rate: float = 0.05                   # P(intensification per step)
    time_limit: float | None = None              # hard wall-clock cap (s)
    seed: int = 7


@dataclass
class SAResult:
    routes: List[List[int]]
    energy: float
    feasible: bool
    runtime: float
    history: List[float] = field(default_factory=list)
    iterations: int = 0
    accepted: int = 0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_NEIGHBOURS = (
    ops.swap_two,
    ops.move_customer,
    ops.reverse_segment,
    ops.inter_drone_swap,
)


def _calibrate_temperature(inst: Instance,
                           sol: Sequence[Sequence[int]],
                           dist: List[List[float]],
                           rng: random.Random,
                           penalty: float,
                           samples: int = 50) -> float:
    """
    Sample ``samples`` random moves from ``sol`` and pick T0 so that the
    average worsening move has acceptance probability ~0.5.
    """
    base = ga_fitness(inst, sol, dist, penalty)
    deltas: List[float] = []
    cur = [list(r) for r in sol]
    for _ in range(samples):
        op = rng.choice(_NEIGHBOURS)
        cand = repair(inst, op(cur, rng))
        f = ga_fitness(inst, cand, dist, penalty)
        if f > base:
            deltas.append(f - base)
    if not deltas:
        return 1.0
    avg = sum(deltas) / len(deltas)
    # P(accept) = exp(-avg / T) = 0.5  =>  T = avg / ln(2)
    return max(avg / math.log(2.0), 1e-3)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def solve(inst: Instance, cfg: SAConfig | None = None) -> SAResult:
    cfg = cfg or SAConfig()
    rng = random.Random(cfg.seed)
    dist = distance_matrix(inst)

    current = repair(inst, nearest_neighbor(inst))
    cur_fit = ga_fitness(inst, current, dist, cfg.penalty_weight)
    best = [list(r) for r in current]
    best_fit = cur_fit

    if cfg.initial_temperature is None:
        T = _calibrate_temperature(inst, current, dist, rng, cfg.penalty_weight)
    else:
        T = cfg.initial_temperature
    T_min = cfg.final_temperature

    history: List[float] = []
    accepted = 0
    last_improvement = 0
    t0 = time.perf_counter()
    deadline = (t0 + cfg.time_limit) if cfg.time_limit else None

    iters_done = 0
    for it in range(cfg.iterations):
        if deadline is not None and time.perf_counter() > deadline:
            break
        iters_done = it + 1
        # Most steps use a cheap stochastic neighbour; with probability
        # `two_opt_rate` we run a best-improvement 2-opt on one route as
        # an intensification step.
        if rng.random() < cfg.two_opt_rate:
            cand = ops.two_opt_route_best(inst, current, dist, rng)
        else:
            op = rng.choice(_NEIGHBOURS)
            cand = op(current, rng)
        cand = repair(inst, cand)
        cand_fit = ga_fitness(inst, cand, dist, cfg.penalty_weight)
        delta = cand_fit - cur_fit

        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            current = cand
            cur_fit = cand_fit
            accepted += 1
            if cur_fit < best_fit - 1e-9:
                best_fit = cur_fit
                best = [list(r) for r in current]
                last_improvement = it

        history.append(best_fit)

        # geometric cooling, with a floor
        T = max(T * cfg.alpha, T_min)

        # simple reheat: if we have not improved for a while, kick the
        # temperature back up and apply a perturbation
        if it - last_improvement >= cfg.reheat_after:
            T = max(T * 5.0, 1.0)
            current = repair(inst, ops.random_perturbation(best, rng, moves=4))
            cur_fit = ga_fitness(inst, current, dist, cfg.penalty_weight)
            last_improvement = it

    runtime = time.perf_counter() - t0
    feas, _ = solution_feasible(inst, best, dist)
    energy = solution_energy(inst, best, dist)
    return SAResult(
        routes=best,
        energy=energy,
        feasible=feas,
        runtime=runtime,
        history=history,
        iterations=iters_done,
        accepted=accepted,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Sequence[str]) -> None:
    if len(argv) < 2:
        print("usage: python -m src.metaheuristics.simulated_annealing "
              "<instance.json> [iterations]")
        sys.exit(1)
    inst = load_instance(argv[1])
    cfg = SAConfig()
    if len(argv) > 2:
        cfg.iterations = int(argv[2])
    res = solve(inst, cfg)
    rate = (res.accepted / res.iterations * 100.0
            if res.iterations else 0.0)
    print(f"[SA] best energy = {res.energy:.3f}   "
          f"feasible = {res.feasible}   "
          f"time = {res.runtime:.2f}s   "
          f"acceptance = {rate:.1f}%")
    for k, r in enumerate(res.routes):
        path = " -> ".join(str(c) for c in [0] + list(r) + [0])
        print(f"  drone {k}: {path}")


if __name__ == "__main__":
    main(sys.argv)
