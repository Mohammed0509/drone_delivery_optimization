"""
Operators shared by the GA and the SA.

Solution encoding
-----------------
A solution is a list of K routes, one per drone. A route is a list of
customer ids in the order they are visited. The depot (0) is implicit
at both ends. A drone may have an empty route (not used).

All operators in this file are *pure*: they take a solution and a
random generator and return a new solution without mutating the input.
"""

from __future__ import annotations

import copy
import random
from typing import List, Sequence, Tuple

from src.utils.energy import Instance, route_energy


Solution = List[List[int]]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def clone(sol: Solution) -> Solution:
    return [list(r) for r in sol]


def all_customers(sol: Solution) -> List[int]:
    out: List[int] = []
    for r in sol:
        out.extend(r)
    return out


def giant_tour(sol: Solution) -> List[int]:
    """Concatenate routes in order, dropping empties."""
    return [c for r in sol for c in r]


def split_giant_tour(inst: Instance,
                     tour: Sequence[int]) -> Solution:
    """
    Greedy split of a permutation of customers into K routes.
    Adds customers to the current route while capacity allows; then
    moves on to the next drone. Battery isn't checked here; the repair
    step fixes that if needed.
    """
    K = inst.num_drones
    routes: Solution = [[] for _ in range(K)]
    cap = inst.payload_capacity
    k = 0
    load = 0.0
    for c in tour:
        d = inst.demand(c)
        if k >= K:
            # all drones already at capacity; dump remaining customers on
            # the last drone and let repair untangle
            routes[K - 1].append(c)
            continue
        if load + d > cap:
            k += 1
            load = 0.0
            if k >= K:
                routes[K - 1].append(c)
                continue
        routes[k].append(c)
        load += d
    return routes


# --------------------------------------------------------------------------
# Crossover (GA)
# --------------------------------------------------------------------------

def order_crossover(p1: Solution, p2: Solution,
                    inst: Instance, rng: random.Random) -> Solution:
    """
    Order Crossover (OX) on the giant-tour view.

    1. Take the two parents as permutations of customers (giant tour).
    2. Pick a random slice [i, j] in parent 1 and copy it.
    3. Fill the remaining positions with parent 2's order, skipping
       customers already inherited.
    4. Re-split into K routes via ``split_giant_tour``.
    """
    t1 = giant_tour(p1)
    t2 = giant_tour(p2)
    n = len(t1)
    if n < 2 or t1 == t2:
        return clone(p1)

    i, j = sorted(rng.sample(range(n), 2))
    middle = t1[i:j + 1]
    middle_set = set(middle)
    fill = [c for c in t2 if c not in middle_set]

    child_tour: List[int] = [0] * n
    child_tour[i:j + 1] = middle
    pos = 0
    for k in range(n):
        if i <= k <= j:
            continue
        child_tour[k] = fill[pos]
        pos += 1
    return split_giant_tour(inst, child_tour)


def route_crossover(p1: Solution, p2: Solution,
                    inst: Instance, rng: random.Random) -> Solution:
    """
    Route-level crossover.

    Pick one route from p1 and keep it as-is in the child. Remove its
    customers from p2 and use the remaining order of p2 to fill the
    other drones via a greedy split.
    """
    non_empty = [k for k, r in enumerate(p1) if r]
    if not non_empty:
        return clone(p2)

    keep_idx = rng.choice(non_empty)
    kept_route = list(p1[keep_idx])
    kept_set = set(kept_route)

    remaining_tour = [c for c in giant_tour(p2) if c not in kept_set]
    other_routes = split_giant_tour(
        # build a temporary "virtual" instance with K-1 drones to split on
        # we just use the same instance and only fill the remaining slots
        inst, remaining_tour,
    )

    K = inst.num_drones
    child: Solution = [[] for _ in range(K)]
    child[keep_idx] = kept_route
    free_slots = [k for k in range(K) if k != keep_idx]
    # pour the K split routes into the free slots (drop overflow)
    for slot, src in zip(free_slots, other_routes):
        child[slot] = list(src)
    # if there were leftover customers (more split routes than free slots),
    # append them to the last drone; repair will spread them later
    leftover_routes = other_routes[len(free_slots):]
    if leftover_routes:
        last = free_slots[-1] if free_slots else keep_idx
        for r in leftover_routes:
            child[last].extend(r)
    return child


# --------------------------------------------------------------------------
# Mutation / neighbourhood moves (used by GA and SA)
# --------------------------------------------------------------------------

def swap_two(sol: Solution, rng: random.Random) -> Solution:
    """Swap two customers picked anywhere in the solution."""
    customers = [(k, idx) for k, r in enumerate(sol) for idx in range(len(r))]
    if len(customers) < 2:
        return clone(sol)
    a, b = rng.sample(customers, 2)
    new = clone(sol)
    new[a[0]][a[1]], new[b[0]][b[1]] = new[b[0]][b[1]], new[a[0]][a[1]]
    return new


def move_customer(sol: Solution, rng: random.Random) -> Solution:
    """
    Pick one customer and move it to a (possibly different) drone, at a
    random insertion position.
    """
    K = len(sol)
    src_candidates = [k for k, r in enumerate(sol) if r]
    if not src_candidates:
        return clone(sol)
    src = rng.choice(src_candidates)
    pos = rng.randrange(len(sol[src]))
    cust = sol[src][pos]

    new = clone(sol)
    del new[src][pos]
    dst = rng.randrange(K)
    insert_at = rng.randrange(len(new[dst]) + 1)
    new[dst].insert(insert_at, cust)
    return new


def reverse_segment(sol: Solution, rng: random.Random) -> Solution:
    """2-opt-style reversal of a segment within a single route."""
    candidates = [k for k, r in enumerate(sol) if len(r) >= 2]
    if not candidates:
        return clone(sol)
    k = rng.choice(candidates)
    r = sol[k]
    i, j = sorted(rng.sample(range(len(r)), 2))
    new = clone(sol)
    new[k][i:j + 1] = list(reversed(new[k][i:j + 1]))
    return new


def inter_drone_swap(sol: Solution, rng: random.Random) -> Solution:
    """
    Pick one customer in drone A and one in drone B (A != B) and swap
    them. Falls back to ``swap_two`` if only one drone is in use.
    """
    used = [k for k, r in enumerate(sol) if r]
    if len(used) < 2:
        return swap_two(sol, rng)
    ka, kb = rng.sample(used, 2)
    ia = rng.randrange(len(sol[ka]))
    ib = rng.randrange(len(sol[kb]))
    new = clone(sol)
    new[ka][ia], new[kb][ib] = new[kb][ib], new[ka][ia]
    return new


def random_perturbation(sol: Solution,
                        rng: random.Random,
                        moves: int = 3) -> Solution:
    """Compose several elementary moves: shake the solution."""
    new = clone(sol)
    ops = (swap_two, move_customer, reverse_segment, inter_drone_swap)
    for _ in range(moves):
        new = rng.choice(ops)(new, rng)
    return new


# --------------------------------------------------------------------------
# Best-improvement 2-opt on a single route (intensification move)
# --------------------------------------------------------------------------

def two_opt_route_best(inst: Instance,
                       sol: Solution,
                       dist: List[List[float]],
                       rng: random.Random) -> Solution:
    """
    Pick one non-trivial route and replace it by its best 2-opt neighbour.

    Scans every segment reversal r[i..j] (1 <= i < j <= len(r)-1) and keeps
    the one that gives the largest energy decrease. If no reversal helps,
    the solution is returned unchanged.

    This is a *deterministic* local-search move; using it occasionally
    inside SA mixes random walk diversification with structured
    intensification, similar to the standard hybrid SA + local search
    approach.
    """
    candidates = [k for k, r in enumerate(sol) if len(r) >= 3]
    if not candidates:
        return clone(sol)

    k = rng.choice(candidates)
    r = sol[k]
    base_e = route_energy(inst, r, dist)
    best_route = list(r)
    best_e = base_e

    n = len(r)
    for i in range(n - 1):
        for j in range(i + 1, n):
            cand = r[:i] + list(reversed(r[i:j + 1])) + r[j + 1:]
            e = route_energy(inst, cand, dist)
            if e < best_e - 1e-9:
                best_e = e
                best_route = cand

    if best_e >= base_e - 1e-9:
        return clone(sol)
    new = clone(sol)
    new[k] = best_route
    return new


# --------------------------------------------------------------------------
# Mutation choice used by the GA
# --------------------------------------------------------------------------

def mutate(sol: Solution, rng: random.Random,
           prob: float = 0.3) -> Solution:
    """Apply at most one mutation operator with probability ``prob``."""
    if rng.random() > prob:
        return clone(sol)
    op = rng.choices(
        [swap_two, move_customer, reverse_segment, inter_drone_swap],
        weights=[2, 3, 2, 2],
        k=1,
    )[0]
    return op(sol, rng)
