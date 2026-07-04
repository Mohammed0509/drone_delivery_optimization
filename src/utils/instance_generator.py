"""
Random instance generator.

We sample customer coordinates uniformly inside a square map, draw demands
from a small integer range, and pick drone parameters that are usually
binding (i.e. battery and payload do matter, otherwise the problem becomes
a pure assignment problem).

Run as a script to fill ``data/generated_instances`` and copy a curated
subset to ``data/benchmarks``.
"""

from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from typing import List

from .energy import Customer, Instance, NoFlyZone, save_instance


# --------------------------------------------------------------------------
# Generator configuration
# --------------------------------------------------------------------------

@dataclass
class GenConfig:
    name: str
    n_customers: int
    n_drones: int
    map_size: float = 100.0
    demand_low: int = 1
    demand_high: int = 5
    payload_capacity: float = 12.0
    battery_capacity: float = 2500.0
    drone_weight: float = 2.0
    energy_factor: float = 1.0
    n_no_fly_zones: int = 0
    seed: int = 0


# Curated set covering the three size classes asked for in the brief.
# Sizes chosen so that the exact methods can still terminate on the
# small ones in a reasonable time. Capacities are picked so that the
# expected total demand sits comfortably below K * Q (no trivial
# infeasibility) while still being a binding constraint in practice.
DEFAULT_CONFIGS: List[GenConfig] = [
    # small ----------------------------------------------------------
    GenConfig("small_01", n_customers=6,  n_drones=2,
              payload_capacity=12.0, battery_capacity=2500.0, seed=1),
    GenConfig("small_02", n_customers=7,  n_drones=2,
              payload_capacity=14.0, battery_capacity=2800.0, seed=2),
    GenConfig("small_03", n_customers=8,  n_drones=3,
              payload_capacity=12.0, battery_capacity=2500.0, seed=3),
    GenConfig("small_04", n_customers=8,  n_drones=3,
              payload_capacity=12.0, battery_capacity=2500.0, seed=4,
              n_no_fly_zones=1),
    # medium ---------------------------------------------------------
    GenConfig("medium_01", n_customers=12, n_drones=3,
              payload_capacity=16.0, battery_capacity=3500.0, seed=11),
    GenConfig("medium_02", n_customers=14, n_drones=3,
              payload_capacity=18.0, battery_capacity=4000.0, seed=12),
    GenConfig("medium_03", n_customers=15, n_drones=4,
              payload_capacity=16.0, battery_capacity=3500.0, seed=13,
              n_no_fly_zones=1),
    # large ----------------------------------------------------------
    GenConfig("large_01", n_customers=20, n_drones=4,
              payload_capacity=20.0, battery_capacity=4500.0, seed=21),
    GenConfig("large_02", n_customers=25, n_drones=5,
              payload_capacity=20.0, battery_capacity=4500.0, seed=22),
    GenConfig("large_03", n_customers=30, n_drones=5,
              payload_capacity=22.0, battery_capacity=5000.0, seed=23,
              n_no_fly_zones=2),
]


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def _sample_no_fly_zones(rng: random.Random,
                         k: int,
                         map_size: float,
                         depot: tuple) -> List[NoFlyZone]:
    """Place ``k`` non-overlapping rectangles that do not contain the depot."""
    zones: List[NoFlyZone] = []
    attempts = 0
    while len(zones) < k and attempts < 200:
        attempts += 1
        w = rng.uniform(map_size * 0.10, map_size * 0.20)
        h = rng.uniform(map_size * 0.10, map_size * 0.20)
        x0 = rng.uniform(0, map_size - w)
        y0 = rng.uniform(0, map_size - h)
        cand = NoFlyZone(x0, y0, x0 + w, y0 + h)
        # depot must be reachable
        if cand.xmin <= depot[0] <= cand.xmax and \
           cand.ymin <= depot[1] <= cand.ymax:
            continue
        zones.append(cand)
    return zones


def _inside_any_zone(x: float, y: float,
                     zones: List[NoFlyZone],
                     margin: float = 0.0) -> bool:
    for z in zones:
        if (z.xmin - margin <= x <= z.xmax + margin and
                z.ymin - margin <= y <= z.ymax + margin):
            return True
    return False


def generate_instance(cfg: GenConfig) -> Instance:
    rng = random.Random(cfg.seed)
    depot = (cfg.map_size / 2.0, cfg.map_size / 2.0)

    # Sample no-fly zones first so we can reject customers that fall
    # inside them. A customer placed inside a zone is unreachable and
    # makes the instance trivially infeasible.
    nfz = _sample_no_fly_zones(rng, cfg.n_no_fly_zones, cfg.map_size, depot)

    customers: List[Customer] = []
    for cid in range(1, cfg.n_customers + 1):
        for _ in range(200):
            x = rng.uniform(0, cfg.map_size)
            y = rng.uniform(0, cfg.map_size)
            if not _inside_any_zone(x, y, nfz, margin=0.5):
                break
        d = float(rng.randint(cfg.demand_low, cfg.demand_high))
        customers.append(Customer(id=cid, x=round(x, 2), y=round(y, 2), demand=d))

    return Instance(
        name=cfg.name,
        depot=depot,
        customers=customers,
        num_drones=cfg.n_drones,
        payload_capacity=cfg.payload_capacity,
        battery_capacity=cfg.battery_capacity,
        drone_weight=cfg.drone_weight,
        energy_factor=cfg.energy_factor,
        no_fly_zones=nfz,
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def build_all(out_dir: str, bench_dir: str,
              configs: List[GenConfig] | None = None) -> List[str]:
    """Generate every configured instance, return their JSON paths."""
    if configs is None:
        configs = DEFAULT_CONFIGS

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(bench_dir, exist_ok=True)

    paths: List[str] = []
    for cfg in configs:
        inst = generate_instance(cfg)
        out = os.path.join(out_dir, f"{cfg.name}.json")
        save_instance(inst, out)
        paths.append(out)
        # mirror to benchmarks so the experimental pipeline always has
        # the same instances even if the generator parameters change later
        shutil.copyfile(out, os.path.join(bench_dir, f"{cfg.name}.json"))

    return paths


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    root = os.path.abspath(os.path.join(_here(), "..", ".."))
    out_dir = os.path.join(root, "data", "generated_instances")
    bench_dir = os.path.join(root, "data", "benchmarks")
    paths = build_all(out_dir, bench_dir)
    print(f"Generated {len(paths)} instances in {out_dir}")
    for p in paths:
        print(" -", os.path.relpath(p, root))


if __name__ == "__main__":
    main()
