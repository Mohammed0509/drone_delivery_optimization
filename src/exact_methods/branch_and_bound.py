"""
Branch and bound for drone delivery.

Strategy
--------
The energy along a route depends on the *order* of visits because the
payload changes after every drop-off. Branching directly on individual
arcs is therefore awkward: a partial route does not have a well-defined
energy until we know the full set of customers it serves.

We use a two-level decomposition that maps cleanly onto B&B:

    1. Branch on **assignment**: in a fixed order, decide which drone
       each customer belongs to. This produces a tree with at most K^n
       leaves but symmetry breaking and bounding cut a lot of it.
    2. At every leaf, the per-drone TSP becomes a small problem with
       known starting payload. We solve it exactly with the Held-Karp
       dynamic program.

Symmetry breaking
-----------------
Drones are identical so we forbid using drone ``k`` before all drones
``0..k-1`` have at least one customer. This collapses the K! relabelings
to a single representative.

Lower bound
-----------
For a partial assignment we use:

    LB = sum_k  HK(S_k)        # exact TSP energy of already-fixed sets
       + sum_{i unassigned}
            f * w_min(i) * drone_weight

where ``w_min(i) = min_{j != i} d(i, j)``. Each unassigned customer must
be entered by some arc, and the minimum-weight contribution is taken
with the empty-drone weight. The bound is therefore admissible.

Bounding is conservative and the implementation stays small enough to
remain readable, which matches the brief: educational, not industrial.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence, Tuple

from src.models.graph_based_model import nearest_neighbor
from src.utils.energy import (
    Instance,
    distance_matrix,
    load_instance,
    route_energy,
    solution_energy,
    solution_feasible,
)


@dataclass
class BBResult:
    routes: List[List[int]]
    energy: float
    runtime: float
    nodes_explored: int
    completed: bool        # False if the time limit was hit


# --------------------------------------------------------------------------
# Held-Karp TSP for a single drone
# --------------------------------------------------------------------------

def best_route_for_drone(inst: Instance,
                         customers: Sequence[int],
                         dist: List[List[float]]
                         ) -> Tuple[List[int], float]:
    """
    Optimal ordering of ``customers`` for a single drone (depot 0 at both
    ends). Returns (route, energy). Empty input gives ([], 0).

    Bitmask DP, O(2^m * m^2) with m = len(customers).
    """
    m = len(customers)
    if m == 0:
        return [], 0.0

    P = sum(inst.demand(c) for c in customers)
    W = inst.drone_weight
    f = inst.energy_factor

    idx = {c: t for t, c in enumerate(customers)}
    demands = [inst.demand(c) for c in customers]

    # f_dp[mask][last] = min energy to leave depot, visit exactly the
    # customers in `mask` (1-bit per index), ending at customer index
    # `last` (still on the customer, not yet returned).
    INF = math.inf
    size = 1 << m
    f_dp = [[INF] * m for _ in range(size)]

    for j in range(m):
        # depot -> customers[j], carrying full payload P
        f_dp[1 << j][j] = f * dist[0][customers[j]] * (W + P)

    for mask in range(size):
        for last in range(m):
            if not (mask >> last) & 1:
                continue
            cur = f_dp[mask][last]
            if cur == INF:
                continue
            # already-delivered demand = sum of demands in `mask`
            delivered = sum(demands[t] for t in range(m) if (mask >> t) & 1)
            payload_after = P - delivered      # weight carried right now
            for nxt in range(m):
                if (mask >> nxt) & 1:
                    continue
                add = f * dist[customers[last]][customers[nxt]] * (W + payload_after)
                v = cur + add
                nm = mask | (1 << nxt)
                if v < f_dp[nm][nxt]:
                    f_dp[nm][nxt] = v

    full = size - 1
    best = INF
    best_last = -1
    for last in range(m):
        # return leg with empty drone
        v = f_dp[full][last] + f * dist[customers[last]][0] * W
        if v < best:
            best = v
            best_last = last

    # reconstruct
    order: List[int] = []
    mask = full
    last = best_last
    while mask:
        order.append(customers[last])
        # find prev
        prev = -1
        prev_mask = mask ^ (1 << last)
        if prev_mask == 0:
            break
        delivered_prev = sum(demands[t] for t in range(m)
                             if (prev_mask >> t) & 1)
        payload_after_prev = P - delivered_prev
        for p in range(m):
            if not (prev_mask >> p) & 1:
                continue
            arc = f * dist[customers[p]][customers[last]] * (W + payload_after_prev)
            if abs(f_dp[prev_mask][p] + arc - f_dp[mask][last]) < 1e-6:
                prev = p
                break
        if prev < 0:           # numerical guard
            break
        mask = prev_mask
        last = prev
    order.reverse()

    # sanity check: if the recovered route is incomplete, fall back to
    # the natural enumeration order (rare numerical edge case)
    if len(order) != m:
        order = list(customers)
        best = route_energy(inst, order, dist)
    return order, best


# --------------------------------------------------------------------------
# Feasibility helpers used during branching
# --------------------------------------------------------------------------

def _drone_feasible(inst: Instance,
                    customers: Sequence[int],
                    dist: List[List[float]]) -> Tuple[bool, float]:
    """Check capacity then battery using the optimal routing."""
    if not customers:
        return True, 0.0
    if sum(inst.demand(c) for c in customers) > inst.payload_capacity + 1e-9:
        return False, math.inf
    _, e = best_route_for_drone(inst, customers, dist)
    if e > inst.battery_capacity + 1e-9:
        return False, math.inf
    return True, e


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def solve(inst: Instance, time_limit: float = 30.0) -> BBResult:
    n = inst.n
    K = inst.num_drones
    dist = distance_matrix(inst)

    # Initial upper bound from the greedy heuristic ----------------------
    seed = nearest_neighbor(inst)
    feas, _ = solution_feasible(inst, seed, dist)
    if feas:
        best_routes = [list(r) for r in seed]
        best_cost = solution_energy(inst, seed, dist)
    else:
        best_routes = None
        best_cost = math.inf

    # Customer processing order: descending demand, then ascending depot
    # distance. This puts "heavy" customers high in the tree which helps
    # pruning.
    order = sorted(
        range(1, n + 1),
        key=lambda c: (-inst.demand(c), dist[0][c]),
    )

    # Cheap admissible contribution per customer if it ends up unassigned
    w_min = [
        min(dist[i][j] for j in range(n + 1) if j != i)
        for i in range(n + 1)
    ]
    weight_factor = inst.energy_factor * inst.drone_weight
    remaining_lb = [
        sum(w_min[c] for c in order[k:]) * weight_factor
        for k in range(len(order) + 1)
    ]

    # State held during the search
    assignment: List[List[int]] = [[] for _ in range(K)]
    drone_costs = [0.0] * K
    nodes = [0]
    deadline = time.perf_counter() + time_limit
    timed_out = [False]

    # cached optimal-route cost for a tuple of customers (avoids redoing
    # the Held-Karp DP when the same set comes back)
    @lru_cache(maxsize=None)
    def opt_drone_cost(group: Tuple[int, ...]) -> float:
        if not group:
            return 0.0
        ok, e = _drone_feasible(inst, list(group), dist)
        return e if ok else math.inf

    def recurse(level: int, max_drone_used: int) -> None:
        nonlocal best_cost, best_routes
        if time.perf_counter() > deadline:
            timed_out[0] = True
            return
        nodes[0] += 1

        if level == len(order):
            # all customers placed; current drone_costs are exact
            total = sum(drone_costs)
            if total < best_cost - 1e-9:
                best_cost = total
                best_routes = []
                for k in range(K):
                    if assignment[k]:
                        route, _ = best_route_for_drone(inst, assignment[k], dist)
                    else:
                        route = []
                    best_routes.append(route)
            return

        cust = order[level]

        # symmetry breaking: cust may go to any drone k with k <= max_drone_used+1
        upper_drone = min(K - 1, max_drone_used + 1)
        for k in range(upper_drone + 1):
            assignment[k].append(cust)
            new_group = tuple(sorted(assignment[k]))
            new_cost_k = opt_drone_cost(new_group)
            if new_cost_k == math.inf:
                assignment[k].pop()
                continue

            old_cost_k = drone_costs[k]
            drone_costs[k] = new_cost_k

            partial = sum(drone_costs)
            lb = partial + remaining_lb[level + 1]
            if lb < best_cost - 1e-9:
                recurse(level + 1, max(max_drone_used, k))

            drone_costs[k] = old_cost_k
            assignment[k].pop()
            if timed_out[0]:
                return

    t0 = time.perf_counter()
    recurse(0, -1)
    runtime = time.perf_counter() - t0

    if best_routes is None:
        return BBResult([], math.inf, runtime, nodes[0], not timed_out[0])
    # pad to K drones for a uniform interface
    while len(best_routes) < K:
        best_routes.append([])
    return BBResult(best_routes, best_cost, runtime, nodes[0], not timed_out[0])


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Sequence[str]) -> None:
    if len(argv) < 2:
        print("usage: python -m src.exact_methods.branch_and_bound "
              "<instance.json> [time_limit_seconds]")
        sys.exit(1)
    inst = load_instance(argv[1])
    tl = float(argv[2]) if len(argv) > 2 else 30.0
    res = solve(inst, time_limit=tl)
    feas, why = solution_feasible(inst, res.routes)
    print(f"[B&B] energy   = {res.energy:.3f}")
    print(f"[B&B] runtime  = {res.runtime:.2f}s   "
          f"nodes = {res.nodes_explored}   "
          f"completed = {res.completed}")
    print(f"[B&B] feasible = {feas} ({why})")
    for k, r in enumerate(res.routes):
        path = " -> ".join(str(c) for c in [0] + list(r) + [0])
        print(f"  drone {k}: {path}")


if __name__ == "__main__":
    main(sys.argv)
