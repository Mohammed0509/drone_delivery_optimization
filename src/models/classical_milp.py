"""
Classical MILP formulation of the drone delivery problem.

We use a vehicle-flow formulation. Subtour elimination is handled implicitly
by the load-flow variables ``f[i,j,k]`` (no source of flow outside the
depot, so any subtour disconnected from the depot is infeasible).

Sets
----
    V        nodes, 0 = depot, 1..n = customers
    A        arcs (i, j) with i != j (excluding edges blocked by no-fly zones)
    K        drones

Decision variables
------------------
    x[i,j,k] in {0, 1}      arc (i, j) is used by drone k
    f[i,j,k] >= 0           payload carried on arc (i, j) by drone k

Objective
---------
    min  sum_{(i,j,k)} k_e * d(i,j) * ( W * x[i,j,k] + f[i,j,k] )

where ``W`` is the drone self-weight and ``k_e`` the energy factor.
The expression ``W * x + f`` is exactly (drone_weight + payload) when the
arc is used and 0 otherwise, so the linear objective matches the
non-linear energy formula evaluated along the route.

Constraints
-----------
    (1) each customer is entered exactly once
            sum_{j, k} x[j, i, k] = 1                 forall i in 1..n
    (2) flow conservation per drone at each node
            sum_j x[j, i, k] = sum_j x[i, j, k]       forall i, k
    (3) each drone leaves the depot at most once
            sum_{j != 0} x[0, j, k] <= 1              forall k
    (4) load balance at each customer
            sum_j f[j, i, k] - sum_j f[i, j, k]
              = demand[i] * sum_j x[j, i, k]          forall customer i, k
    (5) drone returns empty
            f[i, 0, k] = 0                            forall i, k
    (6) capacity link
            f[i, j, k] <= Q * x[i, j, k]              forall arc, k
    (7) battery
            sum_{(i,j)} k_e * d(i,j) * (W * x + f) <= B   forall k
    (8) no-fly
            x[i, j, k] = 0                            forall blocked (i, j), k
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pulp

from src.utils.energy import (
    Instance,
    distance_matrix,
    edge_blocked,
    load_instance,
    solution_energy,
    solution_feasible,
)


@dataclass
class MILPResult:
    routes: List[List[int]]
    energy: float
    status: str
    runtime: float
    objective_bound: float | None


# --------------------------------------------------------------------------
# Building the model
# --------------------------------------------------------------------------

def _build_model(inst: Instance,
                 dist: List[List[float]]
                 ) -> Tuple[pulp.LpProblem, dict, dict]:
    n = inst.n
    K = inst.num_drones
    nodes = list(range(n + 1))                  # 0 = depot
    customers = list(range(1, n + 1))
    arcs: List[Tuple[int, int]] = [
        (i, j) for i in nodes for j in nodes
        if i != j and not edge_blocked(inst, i, j)
    ]

    prob = pulp.LpProblem(f"drone_{inst.name}", pulp.LpMinimize)

    x = {(i, j, k): pulp.LpVariable(f"x_{i}_{j}_{k}", cat="Binary")
         for (i, j) in arcs for k in range(K)}
    f = {(i, j, k): pulp.LpVariable(f"f_{i}_{j}_{k}",
                                    lowBound=0,
                                    upBound=inst.payload_capacity)
         for (i, j) in arcs for k in range(K)}

    # objective ---------------------------------------------------------
    prob += pulp.lpSum(
        inst.energy_factor * dist[i][j] *
        (inst.drone_weight * x[(i, j, k)] + f[(i, j, k)])
        for (i, j) in arcs for k in range(K)
    )

    # (1) each customer entered once
    for i in customers:
        prob += pulp.lpSum(x[(j, i, k)]
                           for (j, ii) in arcs if ii == i
                           for k in range(K)) == 1, f"visit_{i}"

    # (2) per-drone flow conservation
    for k in range(K):
        for v in nodes:
            inflow = pulp.lpSum(x[(i, v, k)] for (i, vv) in arcs if vv == v)
            outflow = pulp.lpSum(x[(v, j, k)] for (vv, j) in arcs if vv == v)
            prob += inflow == outflow, f"cons_{v}_{k}"

    # (3) each drone leaves depot at most once
    for k in range(K):
        prob += pulp.lpSum(x[(0, j, k)] for (i, j) in arcs if i == 0) <= 1, \
            f"depot_out_{k}"

    # (4) load conservation at customers
    for k in range(K):
        for i in customers:
            in_load = pulp.lpSum(f[(j, i, k)] for (j, ii) in arcs if ii == i)
            out_load = pulp.lpSum(f[(i, j, k)] for (ii, j) in arcs if ii == i)
            visited = pulp.lpSum(x[(j, i, k)] for (j, ii) in arcs if ii == i)
            prob += in_load - out_load == inst.demand(i) * visited, \
                f"load_{i}_{k}"

    # (5) drone returns empty
    for k in range(K):
        for (i, j) in arcs:
            if j == 0:
                prob += f[(i, j, k)] == 0, f"empty_{i}_{k}"

    # (6) capacity link
    for (i, j) in arcs:
        for k in range(K):
            prob += f[(i, j, k)] <= inst.payload_capacity * x[(i, j, k)], \
                f"cap_{i}_{j}_{k}"

    # (7) battery
    for k in range(K):
        prob += pulp.lpSum(
            inst.energy_factor * dist[i][j] *
            (inst.drone_weight * x[(i, j, k)] + f[(i, j, k)])
            for (i, j) in arcs
        ) <= inst.battery_capacity, f"battery_{k}"

    return prob, x, f


# --------------------------------------------------------------------------
# Extracting routes from a solved model
# --------------------------------------------------------------------------

def _extract_routes(inst: Instance, x: dict) -> List[List[int]]:
    K = inst.num_drones
    routes: List[List[int]] = []
    for k in range(K):
        # build successor map for this drone
        succ: dict = {}
        for (i, j, kk), var in x.items():
            if kk != k:
                continue
            v = var.value()
            if v is not None and v > 0.5:
                succ[i] = j
        if 0 not in succ:
            routes.append([])
            continue
        route: List[int] = []
        cur = succ[0]
        steps = 0
        while cur != 0 and steps <= inst.n + 1:
            route.append(cur)
            cur = succ.get(cur, 0)
            steps += 1
        routes.append(route)
    return routes


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def solve(inst: Instance,
          time_limit: float = 60.0,
          verbose: bool = False) -> MILPResult:
    dist = distance_matrix(inst)
    prob, x, _f = _build_model(inst, dist)

    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0,
                               timeLimit=time_limit)

    t0 = time.perf_counter()
    prob.solve(solver)
    runtime = time.perf_counter() - t0

    status = pulp.LpStatus[prob.status]
    if prob.status not in (pulp.LpStatusOptimal, 1):
        # Even if not optimal, CBC may have an incumbent we can use
        if pulp.value(prob.objective) is None:
            return MILPResult([], float("inf"), status, runtime, None)

    routes = _extract_routes(inst, x)
    energy = solution_energy(inst, routes, dist)
    bound = None
    try:
        # PuLP exposes the LP bound via prob.bestBound on some backends
        bound = prob.solverModel.bestBound  # type: ignore[attr-defined]
    except Exception:
        bound = None
    return MILPResult(routes, energy, status, runtime, bound)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _print_result(inst: Instance, res: MILPResult) -> None:
    feas, why = solution_feasible(inst, res.routes)
    print(f"[MILP] status   = {res.status}")
    print(f"[MILP] energy   = {res.energy:.3f}")
    print(f"[MILP] runtime  = {res.runtime:.2f}s")
    print(f"[MILP] feasible = {feas} ({why})")
    for k, r in enumerate(res.routes):
        path = " -> ".join(str(c) for c in [0] + list(r) + [0])
        print(f"  drone {k}: {path}")


def main(argv: Sequence[str]) -> None:
    if len(argv) < 2:
        print("usage: python -m src.models.classical_milp <instance.json> "
              "[time_limit_seconds]")
        sys.exit(1)
    inst = load_instance(argv[1])
    tl = float(argv[2]) if len(argv) > 2 else 60.0
    res = solve(inst, time_limit=tl, verbose=False)
    _print_result(inst, res)


if __name__ == "__main__":
    main(sys.argv)
