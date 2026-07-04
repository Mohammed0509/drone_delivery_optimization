"""
Nearest-Neighbour solver.

A thin wrapper around the constructive heuristic already implemented in
``src.models.graph_based_model.nearest_neighbor`` that:

    1. runs the greedy construction,
    2. passes the result through the standard repair pipeline so the
       output is guaranteed feasible whenever a feasible solution exists,
    3. returns a typed ``NNResult`` analogous to ``GAResult`` / ``SAResult``
       so the experimental driver can treat it like any other method.

Why keep this separate?
-----------------------
The constructive heuristic itself lives in the graph model because that
is where the per-edge data structure naturally sits. The *solver* wrapper
(adding repair + result formatting) is a thin algorithmic concern, so we
keep it in its own module to mirror the layout of GA / SA / B&B.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import List, Sequence

from src.metaheuristics.repair import repair
from src.models.graph_based_model import nearest_neighbor
from src.utils.energy import (
    Instance,
    distance_matrix,
    load_instance,
    solution_energy,
    solution_feasible,
)


@dataclass
class NNResult:
    routes: List[List[int]]
    energy: float
    feasible: bool
    runtime: float
    history: List[float] = field(default_factory=list)  # always empty (no iterations)


def solve(inst: Instance) -> NNResult:
    """Run the nearest-neighbour construction + repair on ``inst``."""
    t0 = time.perf_counter()
    seed = nearest_neighbor(inst)
    routes = repair(inst, seed)
    runtime = time.perf_counter() - t0

    dist = distance_matrix(inst)
    feas, _ = solution_feasible(inst, routes, dist)
    energy = solution_energy(inst, routes, dist)
    return NNResult(
        routes=routes,
        energy=energy,
        feasible=feas,
        runtime=runtime,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Sequence[str]) -> None:
    if len(argv) < 2:
        print("usage: python -m src.heuristics.nearest_neighbor "
              "<instance.json>")
        sys.exit(1)
    inst = load_instance(argv[1])
    res = solve(inst)
    print(f"[NN] best energy = {res.energy:.3f}   "
          f"feasible = {res.feasible}   "
          f"time = {res.runtime:.4f}s")
    for k, r in enumerate(res.routes):
        path = " -> ".join(str(c) for c in [0] + list(r) + [0])
        print(f"  drone {k}: {path}")


if __name__ == "__main__":
    main(sys.argv)
