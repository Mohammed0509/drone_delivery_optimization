"""
Genetic Algorithm for the drone delivery problem.

Encoding
--------
Direct: a chromosome is a list of K routes. The crossover and mutation
operators (see ``operators.py``) work on this representation directly.

Fitness
-------
Energy + lambda * infeasibility_penalty. The penalty scales with the
amount of capacity / battery overrun so even infeasible solutions can
be ranked. A repair pass is run on every offspring before evaluation
(it does not always succeed but usually pushes the solution back to a
feasible region).

Loop
----
    1. initial population (greedy seed + random shuffles)
    2. tournament selection (size 3)
    3. crossover (route-XO or order-XO, picked randomly)
    4. mutation
    5. repair
    6. elitism: top E individuals always survive
    7. stop on max generations or stagnation
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from src.models.graph_based_model import nearest_neighbor
from src.utils.energy import (
    Instance,
    distance_matrix,
    edge_blocked,
    load_instance,
    route_energy,
    solution_energy,
    solution_feasible,
)
from src.metaheuristics import operators as ops
from src.metaheuristics.repair import repair


# --------------------------------------------------------------------------
# Configuration and result containers
# --------------------------------------------------------------------------

@dataclass
class GAConfig:
    population_size: int = 60
    generations: int = 200
    tournament_size: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.3
    elite: int = 2
    stagnation_limit: int = 50      # stop after this many gens without improvement
    penalty_weight: float = 50.0
    seed: int = 42


@dataclass
class GAResult:
    routes: List[List[int]]
    energy: float
    feasible: bool
    runtime: float
    history: List[float] = field(default_factory=list)
    generations: int = 0


# --------------------------------------------------------------------------
# Fitness with a soft infeasibility penalty
# --------------------------------------------------------------------------

def _violation(inst: Instance,
               sol: Sequence[Sequence[int]],
               dist: List[List[float]]) -> float:
    """Aggregated overrun across all constraints. 0 if feasible."""
    pen = 0.0
    K = inst.num_drones
    if len(sol) > K:
        pen += sum(len(r) for r in sol[K:])
    for r in sol[:K]:
        if not r:
            continue
        load = sum(inst.demand(c) for c in r)
        if load > inst.payload_capacity:
            pen += load - inst.payload_capacity
        e = route_energy(inst, r, dist)
        if e > inst.battery_capacity:
            pen += (e - inst.battery_capacity) / max(inst.battery_capacity, 1.0)
        prev = 0
        for c in list(r) + [0]:
            if edge_blocked(inst, prev, c):
                # no-fly violations are *hard* to detect with a soft
                # penalty if the constant is small, so we make them
                # dominant w.r.t. energy
                pen += 100.0
            prev = c
    # missing or duplicated customers
    flat = [c for r in sol for c in r]
    if sorted(flat) != list(range(1, inst.n + 1)):
        pen += abs(len(flat) - inst.n)
    return pen


def fitness(inst: Instance,
            sol: Sequence[Sequence[int]],
            dist: List[List[float]],
            penalty: float) -> float:
    return solution_energy(inst, sol, dist) + penalty * _violation(inst, sol, dist)


# --------------------------------------------------------------------------
# Population initialisation
# --------------------------------------------------------------------------

def _shuffled_seed(inst: Instance, rng: random.Random) -> List[List[int]]:
    """Random permutation of customers split greedily into K routes."""
    perm = list(range(1, inst.n + 1))
    rng.shuffle(perm)
    return ops.split_giant_tour(inst, perm)


def _initial_population(inst: Instance,
                        size: int,
                        rng: random.Random) -> List[List[List[int]]]:
    pop: List[List[List[int]]] = []
    pop.append(repair(inst, nearest_neighbor(inst)))
    while len(pop) < size:
        pop.append(repair(inst, _shuffled_seed(inst, rng)))
    return pop


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def _tournament(rng: random.Random,
                pop: Sequence[List[List[int]]],
                fits: Sequence[float],
                k: int) -> List[List[int]]:
    candidates = rng.sample(range(len(pop)), k)
    best = min(candidates, key=lambda i: fits[i])
    return [list(r) for r in pop[best]]


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def solve(inst: Instance, cfg: GAConfig | None = None) -> GAResult:
    cfg = cfg or GAConfig()
    rng = random.Random(cfg.seed)
    dist = distance_matrix(inst)

    population = _initial_population(inst, cfg.population_size, rng)
    fits = [fitness(inst, s, dist, cfg.penalty_weight) for s in population]

    best_idx = min(range(len(population)), key=lambda i: fits[i])
    best_sol = [list(r) for r in population[best_idx]]
    best_fit = fits[best_idx]

    history: List[float] = []
    last_improvement = 0
    t0 = time.perf_counter()

    for gen in range(cfg.generations):
        # elitism
        order = sorted(range(len(population)), key=lambda i: fits[i])
        new_pop: List[List[List[int]]] = [
            [list(r) for r in population[order[i]]] for i in range(cfg.elite)
        ]

        while len(new_pop) < cfg.population_size:
            p1 = _tournament(rng, population, fits, cfg.tournament_size)
            p2 = _tournament(rng, population, fits, cfg.tournament_size)
            if rng.random() < cfg.crossover_rate:
                cx = rng.choice((ops.order_crossover, ops.route_crossover))
                child = cx(p1, p2, inst, rng)
            else:
                child = [list(r) for r in p1]
            child = ops.mutate(child, rng, prob=cfg.mutation_rate)
            child = repair(inst, child)
            new_pop.append(child)

        population = new_pop
        fits = [fitness(inst, s, dist, cfg.penalty_weight) for s in population]

        cur_idx = min(range(len(population)), key=lambda i: fits[i])
        if fits[cur_idx] < best_fit - 1e-9:
            best_fit = fits[cur_idx]
            best_sol = [list(r) for r in population[cur_idx]]
            last_improvement = gen

        history.append(min(fits))

        if gen - last_improvement >= cfg.stagnation_limit:
            break

    runtime = time.perf_counter() - t0
    feas, _ = solution_feasible(inst, best_sol, dist)
    energy = solution_energy(inst, best_sol, dist)
    return GAResult(
        routes=best_sol,
        energy=energy,
        feasible=feas,
        runtime=runtime,
        history=history,
        generations=len(history),
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Sequence[str]) -> None:
    if len(argv) < 2:
        print("usage: python -m src.metaheuristics.genetic_algorithm "
              "<instance.json> [generations]")
        sys.exit(1)
    inst = load_instance(argv[1])
    cfg = GAConfig()
    if len(argv) > 2:
        cfg.generations = int(argv[2])
    res = solve(inst, cfg)
    print(f"[GA] best energy = {res.energy:.3f}   "
          f"feasible = {res.feasible}   "
          f"time = {res.runtime:.2f}s   "
          f"generations = {res.generations}")
    for k, r in enumerate(res.routes):
        path = " -> ".join(str(c) for c in [0] + list(r) + [0])
        print(f"  drone {k}: {path}")


if __name__ == "__main__":
    main(sys.argv)
