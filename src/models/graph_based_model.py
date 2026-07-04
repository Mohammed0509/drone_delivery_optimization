"""
Graph-based representation of the drone delivery problem.

Why a second model?
-------------------
The MILP describes the problem in algebraic form. It is convenient for an
exact solver but it does not directly expose the structure that
metaheuristics actually manipulate (routes, neighbouring nodes, ...).
The networkx model takes the opposite view:

    - the depot and customers are nodes of a directed graph
    - every feasible direct flight is an edge
    - the **weight** of an edge approximates its energy cost
    - a *route* is a simple path that starts and ends at the depot
    - a *solution* is a set of edge-disjoint depot-to-depot paths covering
      every customer once

The graph itself does not encode payload-dependent costs (those depend on
the order of visits, which is a property of a path, not of an edge). We
attach an *upper bound* energy weight per edge (taking the worst-case
payload) so that classical graph algorithms can be used as fast heuristics
or feasibility checks. The actual energy of a route is still computed with
``utils.energy.route_energy``.

This module mainly provides:

    - ``build_graph``        feasible flight graph with edge weights
    - ``route_to_path``      convert a route to a networkx path
    - ``check_solution``     connectivity / feasibility checks
    - ``nearest_neighbor``   greedy construction heuristic used to seed
                             the metaheuristics
"""

from __future__ import annotations

import sys
from typing import List, Sequence

import networkx as nx

from src.utils.energy import (
    Instance,
    distance_matrix,
    edge_blocked,
    load_instance,
    route_energy,
    solution_energy,
    solution_feasible,
)


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------

def build_graph(inst: Instance) -> nx.DiGraph:
    """
    Build the directed flight graph.

    Edge weight uses an upper-bound energy: the drone is assumed to carry
    its full payload capacity on every arc. This is pessimistic, but
    keeps the weights independent of the route order which is what
    classical graph algorithms expect.
    """
    g = nx.DiGraph()
    n = inst.n
    dist = distance_matrix(inst)

    g.add_node(0, kind="depot", pos=inst.coords(0))
    for c in inst.customers:
        g.add_node(c.id, kind="customer", pos=(c.x, c.y), demand=c.demand)

    upper_payload = inst.payload_capacity
    weight_factor = inst.energy_factor * (inst.drone_weight + upper_payload)

    for i in range(n + 1):
        for j in range(n + 1):
            if i == j:
                continue
            if edge_blocked(inst, i, j):
                continue
            g.add_edge(i, j,
                       distance=dist[i][j],
                       weight=dist[i][j] * weight_factor)
    return g


# --------------------------------------------------------------------------
# Conversions and checks
# --------------------------------------------------------------------------

def route_to_path(route: Sequence[int]) -> List[int]:
    """Prefix and suffix the depot to a sequence of customer ids."""
    return [0] + list(route) + [0]


def check_solution(inst: Instance,
                   routes: Sequence[Sequence[int]],
                   g: nx.DiGraph | None = None) -> dict:
    """
    Return a dict of feasibility flags using the graph view:
    every used edge must exist, every route must be a simple path, the
    customer cover must be a partition.
    """
    if g is None:
        g = build_graph(inst)

    edges_ok = True
    simple_ok = True
    for r in routes:
        path = route_to_path(r)
        seen = set()
        for u, v in zip(path[:-1], path[1:]):
            if not g.has_edge(u, v):
                edges_ok = False
            if v != 0 and v in seen:
                simple_ok = False
            seen.add(v)

    feas, why = solution_feasible(inst, routes)
    return {
        "edges_exist": edges_ok,
        "simple_paths": simple_ok,
        "energy_battery_capacity_ok": feas or why not in {"battery", "payload"},
        "fully_feasible": feas,
        "reason": why,
    }


# --------------------------------------------------------------------------
# Nearest-neighbour construction
# --------------------------------------------------------------------------

def nearest_neighbor(inst: Instance) -> List[List[int]]:
    """
    Greedy seed used by the metaheuristics. Customers are sorted by
    distance to the depot; we keep adding the closest unassigned customer
    to the current route while it stays feasible (capacity + battery),
    otherwise we open a new drone.

    If the construction needs more drones than available, we start
    over-filling the last drone and rely on the repair operators to fix
    the resulting infeasibility.
    """
    dist = distance_matrix(inst)
    unassigned = set(range(1, inst.n + 1))
    routes: List[List[int]] = []

    while unassigned and len(routes) < inst.num_drones:
        route: List[int] = []
        load = 0.0
        current = 0
        while True:
            cands = [c for c in unassigned
                     if load + inst.demand(c) <= inst.payload_capacity]
            if not cands:
                break
            c = min(cands, key=lambda j: dist[current][j])
            tentative = route + [c]
            # cheap battery check using the upper bound; the real check
            # happens later in the energy model
            if route_energy(inst, tentative, dist) > inst.battery_capacity:
                break
            route.append(c)
            load += inst.demand(c)
            unassigned.remove(c)
            current = c
        routes.append(route)

    # if some customers remain (rare but possible on tight instances),
    # dump them on the least loaded route; the repair operator will move
    # them around afterwards
    if unassigned:
        if not routes:
            routes.append([])
        order = sorted(range(len(routes)),
                       key=lambda k: sum(inst.demand(c) for c in routes[k]))
        for c in list(unassigned):
            routes[order[0]].append(c)
        unassigned.clear()

    while len(routes) < inst.num_drones:
        routes.append([])
    return routes


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Sequence[str]) -> None:
    if len(argv) < 2:
        print("usage: python -m src.models.graph_based_model <instance.json>")
        sys.exit(1)
    inst = load_instance(argv[1])
    g = build_graph(inst)
    print(f"graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    seed = nearest_neighbor(inst)
    energy = solution_energy(inst, seed)
    feas, why = solution_feasible(inst, seed)
    print(f"nearest-neighbour energy = {energy:.2f}  feasible = {feas} ({why})")
    for k, r in enumerate(seed):
        print(f"  drone {k}: {[0] + list(r) + [0]}")
    print("checks:", check_solution(inst, seed, g))


if __name__ == "__main__":
    main(sys.argv)
