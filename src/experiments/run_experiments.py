"""
Experimental driver.

Runs all solvers on every benchmark instance and writes a single CSV
with one row per (instance, method) pair plus a few per-method log
files. The metaheuristics are run several times with different seeds so
we can report mean and standard deviation.

Method line-up:
    NN       Nearest-neighbour constructive baseline (deterministic).
    MILP     Pulp/CBC mixed-integer LP, time-limited to ``MILP_TIME_LIMIT``.
    B&B      Hand-rolled branch and bound, capped at ``EXACT_MAX_N_BB``
             customers.
    OR-Tools Google CP-SAT routing engine with GLS metaheuristic.
    GA       Direct-encoding genetic algorithm with repair.
    SA       Simulated annealing with mixed stochastic + 2-opt moves.

Time limits keep the script reasonable on a laptop. Output files:
    results/tables/results.csv
    results/tables/summary.csv
    results/logs/<method>_<instance>.json   (best routes + history)
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

from src.exact_methods.branch_and_bound import solve as bb_solve
from src.heuristics.nearest_neighbor import solve as nn_solve
from src.metaheuristics.genetic_algorithm import GAConfig, solve as ga_solve
from src.metaheuristics.simulated_annealing import SAConfig, solve as sa_solve
from src.models.classical_milp import solve as milp_solve
from src.utils.energy import Instance, load_instance
from src.utils.metrics import summarize


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BENCH_DIR = os.path.join("data", "benchmarks")
TABLE_DIR = os.path.join("results", "tables")
LOG_DIR = os.path.join("results", "logs")

MILP_TIME_LIMIT = 60.0
BB_TIME_LIMIT = 30.0
# OR-Tools removed from the project; no ORTOOLS_TIME_LIMIT needed
META_RUNS = 3                 # repetitions per metaheuristic per instance
META_SEEDS = (1, 7, 42)

# Skip the heavy exact methods on instances larger than this number of
# customers (still solvable in principle, but not interesting for the
# report given laptop budgets).
EXACT_MAX_N_MILP = 15
EXACT_MAX_N_BB = 10


@dataclass
class RunRecord:
    instance: str
    n_customers: int
    n_drones: int
    method: str
    seed: Optional[int]
    energy: float
    distance: float
    drones_used: int
    feasible: bool
    infeasibility: str
    runtime: float
    extra: str = ""


# --------------------------------------------------------------------------
# Per-method runners
# --------------------------------------------------------------------------

def _ga_cfg_for(inst: Instance, seed: int) -> GAConfig:
    n = inst.n
    if n <= 8:
        gens, pop = 200, 60
    elif n <= 15:
        gens, pop = 200, 60
    else:
        gens, pop = 150, 60
    return GAConfig(generations=gens, population_size=pop, seed=seed)


def _sa_cfg_for(inst: Instance, seed: int) -> SAConfig:
    n = inst.n
    if n <= 8:
        iters, tlim = 5000, 20.0
    elif n <= 15:
        iters, tlim = 6000, 30.0
    else:
        iters, tlim = 6000, 40.0
    return SAConfig(iterations=iters, time_limit=tlim, seed=seed)


def run_nn(inst: Instance) -> List[RunRecord]:
    """Constructive baseline: NN + repair (deterministic, single run)."""
    res = nn_solve(inst)
    s = summarize(inst, res.routes)
    _dump_log("NN", inst.name, None, res.routes, [], s, res.runtime,
              extra={})
    return [RunRecord(
        instance=inst.name, n_customers=inst.n, n_drones=inst.num_drones,
        method="NN", seed=None,
        energy=s["energy"], distance=s["distance"],
        drones_used=s["drones_used"],
        feasible=s["feasible"], infeasibility=s["infeasibility"],
        runtime=res.runtime, extra="",
    )]





def run_milp(inst: Instance) -> List[RunRecord]:
    if inst.n > EXACT_MAX_N_MILP:
        return [RunRecord(
            instance=inst.name, n_customers=inst.n, n_drones=inst.num_drones,
            method="MILP", seed=None, energy=float("nan"), distance=float("nan"),
            drones_used=0, feasible=False, infeasibility="skipped",
            runtime=0.0, extra="instance too large for the time budget",
        )]
    res = milp_solve(inst, time_limit=MILP_TIME_LIMIT)
    s = summarize(inst, res.routes)
    _dump_log("MILP", inst.name, None, res.routes, [], s, res.runtime,
              extra={"status": res.status})
    return [RunRecord(
        instance=inst.name, n_customers=inst.n, n_drones=inst.num_drones,
        method="MILP", seed=None,
        energy=s["energy"], distance=s["distance"],
        drones_used=s["drones_used"],
        feasible=s["feasible"], infeasibility=s["infeasibility"],
        runtime=res.runtime, extra=res.status,
    )]


def run_bb(inst: Instance) -> List[RunRecord]:
    if inst.n > EXACT_MAX_N_BB:
        return [RunRecord(
            instance=inst.name, n_customers=inst.n, n_drones=inst.num_drones,
            method="B&B", seed=None, energy=float("nan"), distance=float("nan"),
            drones_used=0, feasible=False, infeasibility="skipped",
            runtime=0.0, extra="instance too large for the time budget",
        )]
    res = bb_solve(inst, time_limit=BB_TIME_LIMIT)
    s = summarize(inst, res.routes)
    _dump_log("BB", inst.name, None, res.routes, [], s, res.runtime,
              extra={"completed": res.completed, "nodes": res.nodes_explored})
    extra = f"nodes={res.nodes_explored} completed={res.completed}"
    return [RunRecord(
        instance=inst.name, n_customers=inst.n, n_drones=inst.num_drones,
        method="B&B", seed=None,
        energy=s["energy"], distance=s["distance"],
        drones_used=s["drones_used"],
        feasible=s["feasible"], infeasibility=s["infeasibility"],
        runtime=res.runtime, extra=extra,
    )]


def run_ga(inst: Instance) -> List[RunRecord]:
    out: List[RunRecord] = []
    for seed in META_SEEDS:
        cfg = _ga_cfg_for(inst, seed)
        res = ga_solve(inst, cfg)
        s = summarize(inst, res.routes)
        _dump_log("GA", inst.name, seed, res.routes, res.history, s, res.runtime,
                  extra={"generations": res.generations})
        out.append(RunRecord(
            instance=inst.name, n_customers=inst.n, n_drones=inst.num_drones,
            method="GA", seed=seed,
            energy=s["energy"], distance=s["distance"],
            drones_used=s["drones_used"],
            feasible=s["feasible"], infeasibility=s["infeasibility"],
            runtime=res.runtime,
            extra=f"generations={res.generations}",
        ))
    return out


def run_sa(inst: Instance) -> List[RunRecord]:
    out: List[RunRecord] = []
    for seed in META_SEEDS:
        cfg = _sa_cfg_for(inst, seed)
        res = sa_solve(inst, cfg)
        s = summarize(inst, res.routes)
        _dump_log("SA", inst.name, seed, res.routes, res.history, s, res.runtime,
                  extra={"iterations": res.iterations, "accepted": res.accepted})
        out.append(RunRecord(
            instance=inst.name, n_customers=inst.n, n_drones=inst.num_drones,
            method="SA", seed=seed,
            energy=s["energy"], distance=s["distance"],
            drones_used=s["drones_used"],
            feasible=s["feasible"], infeasibility=s["infeasibility"],
            runtime=res.runtime,
            extra=f"iters={res.iterations}",
        ))
    return out


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def _dump_log(method: str, instance: str, seed: Optional[int],
              routes, history, summary, runtime: float, extra: dict) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    suffix = f"_seed{seed}" if seed is not None else ""
    path = os.path.join(LOG_DIR, f"{method}_{instance}{suffix}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "method": method,
            "instance": instance,
            "seed": seed,
            "routes": [list(r) for r in routes],
            "history": list(history),
            "summary": summary,
            "runtime": runtime,
            **extra,
        }, f, indent=2)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def list_instances(folder: str = BENCH_DIR) -> List[str]:
    """Return paths grouped small -> medium -> large so partial runs of
    the experiment still cover the whole size spectrum if interrupted."""
    order = {"small": 0, "medium": 1, "large": 2}
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    files.sort(key=lambda f: (order.get(f.split("_")[0], 9), f))
    return [os.path.join(folder, f) for f in files]


def run_all() -> List[RunRecord]:
    records: List[RunRecord] = []
    paths = list_instances()
    for path in paths:
        inst = load_instance(path)
        print(f"--- {inst.name}  (n={inst.n}, K={inst.num_drones}) ---",
              flush=True)

        for runner, name in [(run_nn, "NN"),
                     (run_milp, "MILP"),
                     (run_bb, "B&B"),
                     (run_ga, "GA"),
                     (run_sa, "SA")]:
            t0 = time.perf_counter()
            recs = runner(inst)
            dt = time.perf_counter() - t0
            for r in recs:
                tag = f"seed={r.seed}" if r.seed is not None else "exact"
                feas = "ok" if r.feasible else f"infeas:{r.infeasibility}"
                print(f"  {name:5s} {tag:10s} "
                      f"energy={r.energy:10.2f}  "
                      f"time={r.runtime:6.2f}s  {feas}",
                      flush=True)
            print(f"  -> {name} block total {dt:.2f}s", flush=True)
            records.extend(recs)
    return records


# --------------------------------------------------------------------------
# CSV writers
# --------------------------------------------------------------------------

def write_results(records: Sequence[RunRecord]) -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)

    df = pd.DataFrame([asdict(r) for r in records])
    df.to_csv(os.path.join(TABLE_DIR, "results.csv"),
              index=False, float_format="%.3f")

    # summary: mean energy / time / feasibility per (instance, method)
    rows = []
    for (instance, method), sub in df.groupby(["instance", "method"]):
        feas_mask = sub["feasible"] & sub["energy"].notna()
        feas_sub = sub[feas_mask]
        if len(feas_sub) > 0:
            mean_e = feas_sub["energy"].mean()
            std_e = (feas_sub["energy"].std()
                     if len(feas_sub) > 1 else 0.0)
            best_e = feas_sub["energy"].min()
        else:
            mean_e, std_e, best_e = float("nan"), float("nan"), float("nan")
        mean_t = sub["runtime"].mean()
        rows.append({
            "instance": instance,
            "method": method,
            "n_customers": int(sub["n_customers"].iloc[0]),
            "n_drones": int(sub["n_drones"].iloc[0]),
            "runs": len(sub),
            "feasible_runs": int(feas_mask.sum()),
            "best_energy": best_e,
            "mean_energy": mean_e,
            "std_energy": std_e,
            "mean_runtime": mean_t,
        })
    sdf = pd.DataFrame(rows)

    # nice ordering: instance group then method
    method_order = {"NN": 0, "MILP": 1, "B&B": 2, "GA": 3, "SA": 4}
    sdf["m_order"] = sdf["method"].map(method_order).fillna(99)
    sdf = sdf.sort_values(["instance", "m_order"]).drop(columns=["m_order"])
    sdf.to_csv(os.path.join(TABLE_DIR, "summary.csv"),
               index=False, float_format="%.3f")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    records = run_all()
    write_results(records)
    print(f"\nWrote results to {TABLE_DIR}")


if __name__ == "__main__":
    main()
