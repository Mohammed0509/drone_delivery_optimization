"""
Repair operators.

Crossover and mutation can produce solutions that violate one or more
constraints (capacity, battery, no-fly, missing or duplicated customer).
Rather than rejecting these solutions outright, we try to fix them with
small, deterministic moves. If repair fails, the caller can fall back
to a penalised fitness or simply discard the offspring.
"""

from __future__ import annotations

from typing import List, Sequence

from src.utils.energy import (
    Instance,
    distance_matrix,
    edge_blocked,
    route_energy,
    route_feasible,
    solution_feasible,
)


Solution = List[List[int]]


# --------------------------------------------------------------------------
# Customer cover (each customer exactly once)
# --------------------------------------------------------------------------

def fix_assignment(inst: Instance, sol: Solution) -> Solution:
    """
    Make sure every customer 1..n appears exactly once.

    Duplicates are removed (keeping the first occurrence). Missing
    customers are inserted into the lightest drone at the position
    that minimises the energy increase.
    """
    K = inst.num_drones
    out: Solution = [list(r) for r in sol]
    while len(out) < K:
        out.append([])

    seen: set = set()
    for k in range(K):
        cleaned: List[int] = []
        for c in out[k]:
            if c in seen or c < 1 or c > inst.n:
                continue
            seen.add(c)
            cleaned.append(c)
        out[k] = cleaned

    missing = [c for c in range(1, inst.n + 1) if c not in seen]
    if not missing:
        return out

    dist = distance_matrix(inst)
    for c in missing:
        # try cheapest insertion across all drones / positions
        best = None
        best_delta = float("inf")
        for k in range(K):
            r = out[k]
            for pos in range(len(r) + 1):
                trial = r[:pos] + [c] + r[pos:]
                load = sum(inst.demand(x) for x in trial)
                if load > inst.payload_capacity + 1e-9:
                    continue
                delta = route_energy(inst, trial, dist) \
                    - route_energy(inst, r, dist)
                if delta < best_delta:
                    best_delta = delta
                    best = (k, pos)
        if best is None:
            # capacity tight everywhere: drop on lightest drone, repair
            # later steps will handle the overflow
            k = min(range(K),
                    key=lambda kk: sum(inst.demand(x) for x in out[kk]))
            out[k].append(c)
        else:
            k, pos = best
            out[k].insert(pos, c)
    return out


# --------------------------------------------------------------------------
# Capacity
# --------------------------------------------------------------------------

def fix_overcapacity(inst: Instance, sol: Solution) -> Solution:
    """Move customers off drones that exceed payload capacity."""
    K = inst.num_drones
    out: Solution = [list(r) for r in sol]

    def load(k: int) -> float:
        return sum(inst.demand(x) for x in out[k])

    changed = True
    safety = 0
    while changed and safety < 5 * inst.n:
        changed = False
        safety += 1
        for k in range(K):
            while load(k) > inst.payload_capacity + 1e-9 and out[k]:
                # move the heaviest customer to the lightest other drone
                # that has room; if nothing fits, just dump it on the
                # globally lightest drone
                cust = max(out[k], key=lambda c: inst.demand(c))
                out[k].remove(cust)
                d = inst.demand(cust)
                candidates = [
                    j for j in range(K)
                    if j != k and load(j) + d <= inst.payload_capacity + 1e-9
                ]
                target = (min(candidates, key=lambda j: load(j))
                          if candidates
                          else min(range(K), key=lambda j: load(j) if j != k
                                   else float("inf")))
                out[target].append(cust)
                changed = True
    return out


# --------------------------------------------------------------------------
# Battery
# --------------------------------------------------------------------------

def fix_battery(inst: Instance, sol: Solution) -> Solution:
    """
    If a route exceeds the battery, remove its costliest customer and
    reinsert it elsewhere. Repeat until feasible or no progress.
    """
    out: Solution = [list(r) for r in sol]
    dist = distance_matrix(inst)
    K = inst.num_drones

    safety = 0
    while safety < 3 * inst.n:
        safety += 1
        violators = [
            k for k, r in enumerate(out)
            if r and route_energy(inst, r, dist) > inst.battery_capacity + 1e-9
        ]
        if not violators:
            return out

        progress = False
        for k in violators:
            r = out[k]
            base = route_energy(inst, r, dist)

            def saving(idx: int) -> float:
                trial = r[:idx] + r[idx + 1:]
                return base - route_energy(inst, trial, dist)

            idx = max(range(len(r)), key=saving)
            cust = r[idx]
            del r[idx]

            # cheapest feasible reinsertion on another drone
            best = None
            best_delta = float("inf")
            for j in range(K):
                if j == k:
                    continue
                rj = out[j]
                load_j = sum(inst.demand(x) for x in rj) + inst.demand(cust)
                if load_j > inst.payload_capacity + 1e-9:
                    continue
                for pos in range(len(rj) + 1):
                    trial = rj[:pos] + [cust] + rj[pos:]
                    if route_energy(inst, trial, dist) > \
                       inst.battery_capacity + 1e-9:
                        continue
                    delta = route_energy(inst, trial, dist) \
                        - route_energy(inst, rj, dist)
                    if delta < best_delta:
                        best_delta = delta
                        best = (j, pos)
            if best is None:
                # nowhere safe to put it back: leave on its own drone (the
                # solution remains infeasible, which the caller can handle)
                out[k].append(cust)
            else:
                j, pos = best
                out[j].insert(pos, cust)
                progress = True
        if not progress:
            return out
    return out


# --------------------------------------------------------------------------
# No-fly zones
# --------------------------------------------------------------------------

def _route_blocked(inst: Instance, r: Sequence[int]) -> bool:
    prev = 0
    for c in list(r) + [0]:
        if edge_blocked(inst, prev, c):
            return True
        prev = c
    return False


def _try_reorder_to_avoid_block(inst: Instance,
                                route: List[int],
                                dist: List[List[float]]) -> List[int] | None:
    """
    Attempt to fix a blocked route by reordering its customers. We try:
        - all 2-opt segment reversals
        - move-one-customer to every other position
    The first reordering that removes all blocked edges *and* keeps the
    route feasible (capacity / battery) is returned. None if nothing
    works.
    """
    if not _route_blocked(inst, route):
        return list(route)
    n = len(route)

    # 2-opt
    for i in range(n - 1):
        for j in range(i + 1, n):
            cand = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
            if _route_blocked(inst, cand):
                continue
            ok, _ = route_feasible(inst, cand, dist)
            if ok:
                return cand

    # single-customer reposition
    for i in range(n):
        c = route[i]
        rest = route[:i] + route[i + 1:]
        for j in range(len(rest) + 1):
            cand = rest[:j] + [c] + rest[j:]
            if _route_blocked(inst, cand):
                continue
            ok, _ = route_feasible(inst, cand, dist)
            if ok:
                return cand
    return None


def fix_no_fly(inst: Instance, sol: Solution) -> Solution:
    """
    Best-effort repair when a route uses a blocked edge.

    Strategy:
        1. Try to reorder the offending route so the blocked edge
           disappears (cheap, keeps the customer on the same drone).
        2. If reordering fails, move offending customers to other drones
           via cheapest feasible insertion.
    """
    if not inst.no_fly_zones:
        return sol
    out: Solution = [list(r) for r in sol]
    dist = distance_matrix(inst)
    K = inst.num_drones

    # Stage 1: in-route reordering ----------------------------------
    for k in range(K):
        if not _route_blocked(inst, out[k]):
            continue
        fixed = _try_reorder_to_avoid_block(inst, out[k], dist)
        if fixed is not None:
            out[k] = fixed

    # Stage 2: move offending customers between drones --------------
    for k in range(K):
        if not _route_blocked(inst, out[k]):
            continue
        moved = True
        while moved and _route_blocked(inst, out[k]):
            moved = False
            for idx, cust in enumerate(out[k]):
                left = out[k][idx - 1] if idx else 0
                right = out[k][idx + 1] if idx + 1 < len(out[k]) else 0
                if not (edge_blocked(inst, left, cust)
                        or edge_blocked(inst, cust, right)):
                    continue
                d = inst.demand(cust)
                best = None
                best_delta = float("inf")
                for j in range(K):
                    if j == k:
                        continue
                    rj = out[j]
                    if sum(inst.demand(x) for x in rj) + d \
                       > inst.payload_capacity + 1e-9:
                        continue
                    for pos in range(len(rj) + 1):
                        trial = rj[:pos] + [cust] + rj[pos:]
                        ok, _ = route_feasible(inst, trial, dist)
                        if not ok:
                            continue
                        delta = route_energy(inst, trial, dist) \
                            - route_energy(inst, rj, dist)
                        if delta < best_delta:
                            best_delta = delta
                            best = (j, pos)
                if best is not None:
                    j, pos = best
                    del out[k][idx]
                    out[j].insert(pos, cust)
                    moved = True
                    break
    return out


# --------------------------------------------------------------------------
# Top-level repair
# --------------------------------------------------------------------------

def repair(inst: Instance, sol: Solution) -> Solution:
    """
    Run all repair stages in order. Idempotent if the input is feasible.

    Fast-path: if the solution is already feasible we return a clean
    copy without invoking the expensive cheapest-insertion routines.
    Most offspring of a feasible parent stay feasible after a single
    swap or move, so this avoids a lot of pointless work.
    """
    feas, _ = solution_feasible(inst, sol)
    if feas:
        return [list(r) for r in sol]
    out = fix_assignment(inst, sol)
    out = fix_overcapacity(inst, out)
    out = fix_no_fly(inst, out)
    out = fix_battery(inst, out)
    return out
