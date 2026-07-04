"""
Small metric helpers used by experiments and reporting.
"""

from __future__ import annotations

from typing import Sequence

from .energy import (
    Instance,
    distance_matrix,
    route_energy,
    solution_energy,
    solution_feasible,
)


def total_distance(inst: Instance, routes: Sequence[Sequence[int]]) -> float:
    """Plain travelled distance, ignoring payload (sanity check)."""
    d = distance_matrix(inst)
    total = 0.0
    for r in routes:
        prev = 0
        for c in r:
            total += d[prev][c]
            prev = c
        total += d[prev][0]
    return total


def used_drones(routes: Sequence[Sequence[int]]) -> int:
    return sum(1 for r in routes if r)


def gap(value: float, reference: float) -> float:
    """Relative gap (%) of ``value`` vs ``reference``. Reference must be > 0."""
    if reference <= 0:
        return 0.0
    return 100.0 * (value - reference) / reference


def summarize(inst: Instance, routes: Sequence[Sequence[int]]) -> dict:
    """Bundle the most common metrics into a dict."""
    d = distance_matrix(inst)
    feas, why = solution_feasible(inst, routes, d)
    return {
        "energy": solution_energy(inst, routes, d),
        "distance": total_distance(inst, routes),
        "drones_used": used_drones(routes),
        "feasible": feas,
        "infeasibility": "ok" if feas else why,
        "per_route_energy": [round(route_energy(inst, r, d), 3) for r in routes],
    }
