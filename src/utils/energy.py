"""
Energy model and instance container for the drone delivery problem.

We use a simple but realistic energy model:

    E(i -> j, w) = k * d(i, j) * (W_drone + w)

where:
    d(i, j)   euclidean distance between i and j
    W_drone   self-weight of the drone (kg)
    w         payload currently carried on edge (i, j) (kg)
    k         scaling constant (energy per unit distance per unit weight)

A drone leaves the depot carrying the sum of demands of its assigned
customers. After each delivery the carried weight decreases by that
customer's demand. The cost of an arc therefore depends on where in the
route it is taken, which is what makes the problem more than just a TSP.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------

@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float


@dataclass
class NoFlyZone:
    """Axis-aligned rectangular forbidden region [xmin, xmax] x [ymin, ymax]."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class Instance:
    name: str
    depot: Tuple[float, float]
    customers: List[Customer]
    num_drones: int
    payload_capacity: float        # max kg per drone
    battery_capacity: float        # max energy per drone
    drone_weight: float            # self-weight of a drone (kg)
    energy_factor: float = 1.0     # k in the formula
    no_fly_zones: List[NoFlyZone] = field(default_factory=list)

    # convenience -------------------------------------------------------
    @property
    def n(self) -> int:
        """Number of customers (depot excluded)."""
        return len(self.customers)

    def coords(self, node: int) -> Tuple[float, float]:
        """Coordinates of a node id; 0 is the depot, 1..n are customers."""
        if node == 0:
            return self.depot
        return (self.customers[node - 1].x, self.customers[node - 1].y)

    def demand(self, node: int) -> float:
        return 0.0 if node == 0 else self.customers[node - 1].demand


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def euclid(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def distance_matrix(inst: Instance) -> List[List[float]]:
    """Symmetric distance matrix of size (n+1) x (n+1) with depot at index 0."""
    n = inst.n
    nodes = [inst.coords(i) for i in range(n + 1)]
    d = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            v = euclid(nodes[i], nodes[j])
            d[i][j] = v
            d[j][i] = v
    return d


# --------------------------------------------------------------------------
# No-fly zone handling
# --------------------------------------------------------------------------

def _segment_intersects_rect(p: Tuple[float, float],
                             q: Tuple[float, float],
                             zone: NoFlyZone) -> bool:
    """Cohen-Sutherland-style test: does segment pq cross the rectangle?"""
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def code(x: float, y: float) -> int:
        c = INSIDE
        if x < zone.xmin:
            c |= LEFT
        elif x > zone.xmax:
            c |= RIGHT
        if y < zone.ymin:
            c |= BOTTOM
        elif y > zone.ymax:
            c |= TOP
        return c

    x1, y1 = p
    x2, y2 = q
    c1 = code(x1, y1)
    c2 = code(x2, y2)
    while True:
        if not (c1 | c2):           # both inside -> intersect
            return True
        if c1 & c2:                 # both share an outside region -> miss
            return False
        # clip the outside endpoint
        co = c1 or c2
        if co & TOP:
            x = x1 + (x2 - x1) * (zone.ymax - y1) / (y2 - y1)
            y = zone.ymax
        elif co & BOTTOM:
            x = x1 + (x2 - x1) * (zone.ymin - y1) / (y2 - y1)
            y = zone.ymin
        elif co & RIGHT:
            y = y1 + (y2 - y1) * (zone.xmax - x1) / (x2 - x1)
            x = zone.xmax
        elif co & LEFT:
            y = y1 + (y2 - y1) * (zone.xmin - x1) / (x2 - x1)
            x = zone.xmin
        else:
            return False
        if co == c1:
            x1, y1 = x, y
            c1 = code(x1, y1)
        else:
            x2, y2 = x, y
            c2 = code(x2, y2)


def edge_blocked(inst: Instance, i: int, j: int) -> bool:
    """True if the straight segment (i, j) crosses any no-fly zone."""
    if not inst.no_fly_zones:
        return False
    p = inst.coords(i)
    q = inst.coords(j)
    return any(_segment_intersects_rect(p, q, z) for z in inst.no_fly_zones)


# --------------------------------------------------------------------------
# Energy of a single route and a full solution
# --------------------------------------------------------------------------

def route_energy(inst: Instance, route: Sequence[int],
                 dist: List[List[float]] | None = None) -> float:
    """
    Energy of a route ``[c1, c2, ..., ck]`` (depot 0 implicit at both ends).

    Empty route -> 0.0. The drone starts with the sum of demands and drops
    each customer's demand after visiting them.
    """
    if not route:
        return 0.0
    if dist is None:
        dist = distance_matrix(inst)

    payload = sum(inst.demand(c) for c in route)
    energy = 0.0
    prev = 0
    for c in route:
        # leg from prev to c: drone carries `payload` kg
        energy += inst.energy_factor * dist[prev][c] * (inst.drone_weight + payload)
        payload -= inst.demand(c)        # delivered
        prev = c
    # return leg, drone is empty
    energy += inst.energy_factor * dist[prev][0] * inst.drone_weight
    return energy


def solution_energy(inst: Instance,
                    routes: Sequence[Sequence[int]],
                    dist: List[List[float]] | None = None) -> float:
    if dist is None:
        dist = distance_matrix(inst)
    return sum(route_energy(inst, r, dist) for r in routes)


# --------------------------------------------------------------------------
# Feasibility
# --------------------------------------------------------------------------

def route_feasible(inst: Instance, route: Sequence[int],
                   dist: List[List[float]] | None = None
                   ) -> Tuple[bool, str]:
    """Return (ok, reason). ``reason`` is 'ok' when feasible."""
    if not route:
        return True, "ok"
    if dist is None:
        dist = distance_matrix(inst)

    load = sum(inst.demand(c) for c in route)
    if load > inst.payload_capacity + 1e-9:
        return False, "payload"

    # battery
    if route_energy(inst, route, dist) > inst.battery_capacity + 1e-9:
        return False, "battery"

    # no-fly
    prev = 0
    for c in list(route) + [0]:
        if edge_blocked(inst, prev, c):
            return False, "no_fly"
        prev = c
    return True, "ok"


def solution_feasible(inst: Instance,
                      routes: Sequence[Sequence[int]],
                      dist: List[List[float]] | None = None
                      ) -> Tuple[bool, str]:
    """Check that every customer is served exactly once and each route is ok."""
    if dist is None:
        dist = distance_matrix(inst)

    served: List[int] = []
    for r in routes:
        served.extend(r)
    if sorted(served) != list(range(1, inst.n + 1)):
        return False, "assignment"

    if len(routes) > inst.num_drones:
        return False, "too_many_drones"

    for r in routes:
        ok, why = route_feasible(inst, r, dist)
        if not ok:
            return False, why
    return True, "ok"


# --------------------------------------------------------------------------
# JSON I/O
# --------------------------------------------------------------------------

def load_instance(path: str) -> Instance:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    customers = [Customer(**c) for c in raw["customers"]]
    nfz = [NoFlyZone(**z) for z in raw.get("no_fly_zones", [])]
    return Instance(
        name=raw["name"],
        depot=tuple(raw["depot"]),
        customers=customers,
        num_drones=raw["num_drones"],
        payload_capacity=raw["payload_capacity"],
        battery_capacity=raw["battery_capacity"],
        drone_weight=raw["drone_weight"],
        energy_factor=raw.get("energy_factor", 1.0),
        no_fly_zones=nfz,
    )


def save_instance(inst: Instance, path: str) -> None:
    payload = {
        "name": inst.name,
        "depot": list(inst.depot),
        "customers": [c.__dict__ for c in inst.customers],
        "num_drones": inst.num_drones,
        "payload_capacity": inst.payload_capacity,
        "battery_capacity": inst.battery_capacity,
        "drone_weight": inst.drone_weight,
        "energy_factor": inst.energy_factor,
        "no_fly_zones": [z.__dict__ for z in inst.no_fly_zones],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
