"""
Read the CSV produced by ``run_experiments`` and generate the figures
used in the report:

    - bar chart: best energy per method, per instance
    - bar chart: mean runtime per method, per instance
    - convergence curves: GA and SA on a representative instance
    - route plots: best feasible solution found, per method, per
      instance (a few are picked for the report)

All figures are written under ``results/plots/`` and the gap-to-best
table under ``results/tables/gap_table.csv``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import pandas as pd

from src.utils.energy import load_instance
from src.utils.visualization import (
    plot_combined_summary,
    plot_comparison_bars,
    plot_convergence,
    plot_runtime_scatter,
    plot_solution,
)


TABLE_DIR = os.path.join("results", "tables")
PLOT_DIR = os.path.join("results", "plots")
LOG_DIR = os.path.join("results", "logs")
BENCH_DIR = os.path.join("data", "benchmarks")


# --------------------------------------------------------------------------
# Energy / time bar charts
# --------------------------------------------------------------------------

METHODS = ["NN", "MILP", "B&B", "GA", "SA"]


def make_bar_charts(summary: pd.DataFrame) -> None:
    instances = sorted(summary["instance"].unique(),
                       key=lambda x: (
                           {"small": 0, "medium": 1, "large": 2}.get(
                               x.split("_")[0], 9), x))
    methods = METHODS

    energy: Dict[str, List[float]] = {m: [] for m in methods}
    times: Dict[str, List[float]] = {m: [] for m in methods}
    for inst in instances:
        sub = summary[summary["instance"] == inst]
        for m in methods:
            row = sub[sub["method"] == m]
            if row.empty:
                energy[m].append(None)
                times[m].append(None)
                continue
            r = row.iloc[0]
            e = r["best_energy"] if r["feasible_runs"] > 0 else None
            energy[m].append(None if pd.isna(e) else float(e))
            times[m].append(float(r["mean_runtime"]))

    plot_comparison_bars(
        instances, energy,
        metric="best feasible energy",
        out_path=os.path.join(PLOT_DIR, "energy_by_method.png"),
    )
    plot_comparison_bars(
        instances, times,
        metric="mean runtime (s)",
        out_path=os.path.join(PLOT_DIR, "runtime_by_method.png"),
    )


# --------------------------------------------------------------------------
# Runtime vs instance size scatter
# --------------------------------------------------------------------------

def make_runtime_scatter(summary: pd.DataFrame) -> None:
    methods = METHODS
    sizes: List[int] = []
    times: Dict[str, List[float]] = {m: [] for m in methods}

    for inst, sub in summary.groupby("instance"):
        sizes.append(int(sub["n_customers"].iloc[0]))
        for m in methods:
            row = sub[sub["method"] == m]
            if row.empty:
                times[m].append(float("nan"))
            else:
                times[m].append(float(row.iloc[0]["mean_runtime"]))

    # paired sort by size for nicer reading
    order = sorted(range(len(sizes)), key=lambda i: sizes[i])
    sizes = [sizes[i] for i in order]
    for m in methods:
        times[m] = [times[m][i] for i in order]

    plot_runtime_scatter(
        sizes, times,
        out_path=os.path.join(PLOT_DIR, "runtime_vs_size.png"),
    )


# --------------------------------------------------------------------------
# Convergence curves (one figure per instance for the report)
# --------------------------------------------------------------------------

def _load_history(method: str, instance: str, seed: int) -> List[float]:
    path = os.path.join(LOG_DIR, f"{method}_{instance}_seed{seed}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("history", []))


def make_convergence_plots(target_instances: List[str],
                            seed: int = 7) -> None:
    for inst_name in target_instances:
        ga_hist = _load_history("GA", inst_name, seed)
        sa_hist = _load_history("SA", inst_name, seed)
        if not (ga_hist or sa_hist):
            continue
        plot_convergence(
            {"GA": ga_hist, "SA": sa_hist},
            title=f"convergence on {inst_name}",
            out_path=os.path.join(PLOT_DIR,
                                  f"convergence_{inst_name}.png"),
        )


# --------------------------------------------------------------------------
# Route plots for the best feasible solution per (instance, method)
# --------------------------------------------------------------------------

def _best_log_for(method: str, instance: str) -> dict | None:
    """Return the JSON log of the best feasible run for the given pair."""
    candidates: List[dict] = []
    for f in os.listdir(LOG_DIR):
        if not f.startswith(f"{method}_{instance}"):
            continue
        with open(os.path.join(LOG_DIR, f), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data["summary"]["feasible"]:
            candidates.append(data)
    if not candidates:
        return None
    return min(candidates, key=lambda d: d["summary"]["energy"])


def make_route_plots(target_instances: List[str]) -> None:
    methods = METHODS
    for inst_name in target_instances:
        path = os.path.join(BENCH_DIR, f"{inst_name}.json")
        if not os.path.exists(path):
            continue
        inst = load_instance(path)
        for m in methods:
            log = _best_log_for(m, inst_name)
            if log is None:
                continue
            s = log["summary"]
            plot_solution(
                inst, log["routes"],
                title=f"{m} on {inst_name}  (E = {s['energy']:.1f})",
                out_path=os.path.join(PLOT_DIR,
                                      f"routes_{inst_name}_{m}.png"),
                summary={
                    "energy": s["energy"],
                    "drones_used": s["drones_used"],
                    "feasible": s["feasible"],
                    "runtime": log.get("runtime", 0.0),
                },
            )


# --------------------------------------------------------------------------
# Gap-to-best table
# --------------------------------------------------------------------------

def make_gap_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for inst, sub in summary.groupby("instance"):
        feas = sub[sub["feasible_runs"] > 0]
        if feas.empty:
            best = float("nan")
        else:
            best = float(feas["best_energy"].min())
        for _, r in sub.iterrows():
            e = r["best_energy"]
            gap = (100.0 * (e - best) / best
                   if best > 0 and not pd.isna(e) else float("nan"))
            rows.append({
                "instance": inst,
                "method": r["method"],
                "best_energy": e,
                "reference_best": best,
                "gap_percent": gap,
                "mean_runtime": r["mean_runtime"],
                "feasible_runs": int(r["feasible_runs"]),
                "runs": int(r["runs"]),
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLE_DIR, "gap_table.csv"),
               index=False, float_format="%.3f")
    return out


# --------------------------------------------------------------------------
# Combined 2x2 dashboard
# --------------------------------------------------------------------------

def make_combined_dashboard(summary: pd.DataFrame,
                             gap_df: pd.DataFrame) -> None:
    """One PNG with energy / runtime / gap / runtime-vs-size for the report."""
    instances = sorted(summary["instance"].unique(),
                       key=lambda x: (
                           {"small": 0, "medium": 1, "large": 2}.get(
                               x.split("_")[0], 9), x))

    energy: Dict[str, List[float]] = {m: [] for m in METHODS}
    times: Dict[str, List[float]] = {m: [] for m in METHODS}
    gap: Dict[str, List[float]] = {m: [] for m in METHODS}
    sizes: List[int] = []

    gap_pivot = gap_df.pivot(index="instance",
                             columns="method",
                             values="gap_percent")

    for inst in instances:
        sub = summary[summary["instance"] == inst]
        sizes.append(int(sub["n_customers"].iloc[0]))
        for m in METHODS:
            row = sub[sub["method"] == m]
            if row.empty:
                energy[m].append(None)
                times[m].append(None)
                gap[m].append(None)
                continue
            r = row.iloc[0]
            e = r["best_energy"] if r["feasible_runs"] > 0 else None
            energy[m].append(None if pd.isna(e) else float(e))
            times[m].append(float(r["mean_runtime"]))
            g_val = (gap_pivot.loc[inst, m]
                     if m in gap_pivot.columns and inst in gap_pivot.index
                     else float("nan"))
            gap[m].append(None if pd.isna(g_val) else float(g_val))

    plot_combined_summary(
        instances, sizes, energy, times, gap,
        out_path=os.path.join(PLOT_DIR, "combined_summary.png"),
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main() -> None:
    summary_path = os.path.join(TABLE_DIR, "summary.csv")
    if not os.path.exists(summary_path):
        raise SystemExit(f"missing {summary_path}; run run_experiments first")
    summary = pd.read_csv(summary_path)

    # Print the loaded summary table so users see per-instance/method stats
    print("\nLoaded summary (per-instance, per-method):")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(summary.round(3).to_string(index=False))

    os.makedirs(PLOT_DIR, exist_ok=True)
    make_bar_charts(summary)
    make_runtime_scatter(summary)

    # representative instances for convergence and route plots
    targets = ["small_03", "medium_01", "medium_03", "large_01", "large_03"]
    make_convergence_plots(targets)
    make_route_plots(targets)

    gap = make_gap_table(summary)
    make_combined_dashboard(summary, gap)

    # show a compact text summary of the gap table
    print("Gap to best (%) per (instance, method):")
    pivot = gap.pivot(index="instance",
                      columns="method",
                      values="gap_percent")
    print(pivot.round(2).to_string())


if __name__ == "__main__":
    main()
