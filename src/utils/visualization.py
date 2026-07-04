"""
Plotting helpers (matplotlib only).

The visual style is intentionally plain: small markers, thin lines,
black-and-white friendly colour cycle, no fancy backgrounds. The goal
is readable figures that survive being printed in a report.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .energy import Instance


# A small, distinct, print-safe colour cycle for routes.
_ROUTE_COLOURS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd",
    "#ff7f0e", "#8c564b", "#17becf", "#bcbd22",
]


# --------------------------------------------------------------------------
# Routes on the map
# --------------------------------------------------------------------------

def plot_solution(inst: Instance,
                  routes: Sequence[Sequence[int]],
                  title: str = "",
                  out_path: str | None = None,
                  show: bool = False,
                  summary: dict | None = None) -> None:
    """
    Draw depot, customers, no-fly zones and the routes.

    Parameters
    ----------
    summary : dict | None
        Optional dict with any of the keys ``energy``, ``runtime``,
        ``feasible``, ``drones_used``. If provided, a small text box is
        rendered in the lower-right corner with these values.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # no-fly zones first so they sit behind everything else
    for z in inst.no_fly_zones:
        rect = mpatches.Rectangle(
            (z.xmin, z.ymin), z.xmax - z.xmin, z.ymax - z.ymin,
            linewidth=1.0, edgecolor="#888888",
            facecolor="#bdbdbd", alpha=0.35,
            hatch="//",
        )
        ax.add_patch(rect)

    # customers
    xs = [c.x for c in inst.customers]
    ys = [c.y for c in inst.customers]
    ax.scatter(xs, ys, s=35, c="black", zorder=3)
    for c in inst.customers:
        ax.annotate(str(c.id), (c.x, c.y),
                    textcoords="offset points",
                    xytext=(4, 4), fontsize=8)

    # depot
    ax.scatter([inst.depot[0]], [inst.depot[1]],
               marker="s", s=80, c="black", zorder=4, label="depot")

    # routes
    for k, r in enumerate(routes):
        if not r:
            continue
        col = _ROUTE_COLOURS[k % len(_ROUTE_COLOURS)]
        path = [0] + list(r) + [0]
        px = [inst.coords(node)[0] for node in path]
        py = [inst.coords(node)[1] for node in path]
        ax.plot(px, py, "-", color=col, linewidth=1.4,
                label=f"drone {k} ({len(r)} stops)", zorder=2)

    ax.set_aspect("equal")
    ax.set_title(title or f"routes for {inst.name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    # optional summary box (energy / time / feasibility)
    if summary:
        lines = []
        if "energy" in summary:
            lines.append(f"energy: {summary['energy']:.1f}")
        if "drones_used" in summary:
            lines.append(f"drones used: {summary['drones_used']}")
        if "feasible" in summary:
            lines.append(f"feasible: {summary['feasible']}")
        if "runtime" in summary:
            lines.append(f"time: {summary['runtime']:.2f}s")
        if lines:
            ax.text(
                0.98, 0.02, "\n".join(lines),
                transform=ax.transAxes,
                fontsize=8, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#fffbe6",
                          edgecolor="#999999",
                          alpha=0.9),
            )

    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------
# Convergence curves
# --------------------------------------------------------------------------

def plot_convergence(histories: Dict[str, Sequence[float]],
                     title: str = "convergence",
                     out_path: str | None = None,
                     show: bool = False) -> None:
    """
    ``histories`` is a mapping algorithm-name -> per-iteration objective
    values (best so far). Each curve is drawn with the same x-axis even
    if the lengths differ.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, hist in histories.items():
        if not hist:
            continue
        # smooth the GA's noisy "best of generation" by taking a running min
        running = np.minimum.accumulate(np.asarray(hist, dtype=float))
        ax.plot(running, label=label, linewidth=1.2)
    ax.set_xlabel("iteration / generation")
    ax.set_ylabel("best objective so far")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------
# Comparative bar chart across instances and methods
# --------------------------------------------------------------------------

def plot_comparison_bars(instances: Sequence[str],
                         results: Dict[str, Sequence[float]],
                         metric: str = "energy",
                         out_path: str | None = None,
                         show: bool = False) -> None:
    """
    ``results`` maps algorithm-name -> list of values, in the same order
    as ``instances``. Missing values can be passed as ``None`` and will
    be skipped.
    """
    methods = list(results.keys())
    n = len(instances)
    x = np.arange(n)
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(max(7, 0.7 * n), 4.5))
    for i, m in enumerate(methods):
        vals = [v if v is not None else np.nan for v in results[m]]
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=m,
               color=_ROUTE_COLOURS[i % len(_ROUTE_COLOURS)],
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across methods")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------
# Runtime scatter (size vs time)
# --------------------------------------------------------------------------

def plot_runtime_scatter(sizes: Sequence[int],
                         times: Dict[str, Sequence[float]],
                         out_path: str | None = None,
                         show: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (label, ts) in enumerate(times.items()):
        ax.scatter(sizes, ts, label=label,
                   color=_ROUTE_COLOURS[i % len(_ROUTE_COLOURS)],
                   edgecolor="black", linewidth=0.5, s=40)
    ax.set_xlabel("number of customers")
    ax.set_ylabel("runtime (s)")
    ax.set_yscale("log")
    ax.set_title("runtime vs instance size")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------
# Combined 2x2 dashboard for the report
# --------------------------------------------------------------------------

def plot_combined_summary(instances: Sequence[str],
                          sizes: Sequence[int],
                          energy: Dict[str, Sequence[float]],
                          times: Dict[str, Sequence[float]],
                          gap: Dict[str, Sequence[float]],
                          out_path: str | None = None,
                          show: bool = False) -> None:
    """
    Single 2x2 dashboard combining the four most useful comparison plots:

        top-left      best feasible energy per (instance, method)
        top-right     mean runtime per (instance, method)
        bottom-left   gap-to-best in percent per (instance, method)
        bottom-right  runtime vs instance size, log y

    ``energy``, ``times`` and ``gap`` are method-name -> per-instance
    sequences aligned with ``instances`` and ``sizes``. Missing values
    can be passed as ``None`` or ``nan``.
    """
    methods = list(energy.keys())
    n = len(instances)
    x = np.arange(n)
    width = 0.8 / max(len(methods), 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("solver comparison", fontsize=13, fontweight="bold")

    def _bar(ax, data: Dict[str, Sequence[float]], ylabel: str, title: str):
        for i, m in enumerate(methods):
            vals = [np.nan if v is None else v for v in data[m]]
            offset = (i - (len(methods) - 1) / 2) * width
            ax.bar(x + offset, vals, width=width, label=m,
                   color=_ROUTE_COLOURS[i % len(_ROUTE_COLOURS)],
                   edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(instances, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    _bar(axes[0, 0], energy,
         ylabel="energy (Wh)",
         title="best feasible energy")
    _bar(axes[0, 1], times,
         ylabel="seconds",
         title="mean runtime")
    _bar(axes[1, 0], gap,
         ylabel="gap (%)",
         title="gap to best feasible solution")

    # bottom-right: runtime vs size scatter, log y
    sax = axes[1, 1]
    order = sorted(range(len(sizes)), key=lambda i: sizes[i])
    sorted_sizes = [sizes[i] for i in order]
    for i, m in enumerate(methods):
        ts = [times[m][k] for k in order]
        ts = [np.nan if v is None else v for v in ts]
        sax.plot(sorted_sizes, ts, "-o", label=m,
                 color=_ROUTE_COLOURS[i % len(_ROUTE_COLOURS)],
                 markersize=4, linewidth=1.0)
    sax.set_yscale("log")
    sax.set_xlabel("number of customers")
    sax.set_ylabel("runtime (s)")
    sax.set_title("runtime vs instance size", fontsize=10)
    sax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

    # one global legend to the right of the figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right",
               bbox_to_anchor=(1.0, 0.5),
               fontsize=9, frameon=True)
    plt.tight_layout(rect=(0.0, 0.0, 0.93, 0.96))

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
