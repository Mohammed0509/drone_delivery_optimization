# Drone-Based Delivery Optimization

University project for the *Combinatorial Optimization and Metaheuristics* course.

We model a small drone delivery system: one depot, several drones, a set of
customers with demands, and the goal of minimizing total energy consumption.
Five solvers are compared on the same instances:

- **NN** — nearest-neighbour constructive baseline (greedy + repair)
- **MILP** — classical mixed-integer LP (PuLP / CBC)
- **B&B** — hand-rolled branch-and-bound exact method (small instances only)
- **GA** — direct-encoding genetic algorithm with repair operators
- **SA** — simulated annealing, mixed stochastic moves + best-improvement 2-opt

A graph-based model (networkx) is also provided as an alternative way of
representing routes and validating feasibility.

## Problem in one sentence

Assign each customer to exactly one drone and order the visits so that every
drone leaves the depot, serves its assigned customers, comes back, respects
its payload and battery, and the total energy is as small as possible.

## Installation

Tested with Python 3.11.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / Mac
source .venv/bin/activate

pip install -r requirements.txt
```

## Project structure

```
drone_delivery_optimization/
├── data/
│   ├── generated_instances/   # JSON files produced by the generator
│   └── benchmarks/            # frozen instances used for the report
├── src/
│   ├── models/                # MILP and graph-based formulations
│   ├── exact_methods/         # branch and bound
│   ├── heuristics/            # NN baseline
│   ├── metaheuristics/        # GA, SA, operators, repair
│   ├── utils/                 # energy model, generator, plots, metrics
│   └── experiments/           # batch runs and comparison
├── results/                   # plots, tables, logs (filled at run time)
├── report/                    # report and presentation outlines
├── requirements.txt
└── README.md
```

## Usage

All commands are run from the project root.

### 1. Generate instances

```bash
python -m src.utils.instance_generator
```

This produces JSON files in `data/generated_instances/` and copies a few of
them into `data/benchmarks/` so the experiments are reproducible.

### 2. Solve a single instance

Quick sanity check on one small instance:

```bash
python -m src.heuristics.nearest_neighbor    data/benchmarks/small_01.json
python -m src.models.classical_milp          data/benchmarks/small_01.json
python -m src.exact_methods.branch_and_bound data/benchmarks/small_01.json
python -m src.metaheuristics.genetic_algorithm    data/benchmarks/small_01.json
python -m src.metaheuristics.simulated_annealing  data/benchmarks/small_01.json
```

Each script prints the best solution it found and its energy. NN is a fast
warm baseline; the metaheuristics typically match or improve on it.

### 3. Run the full experimental pipeline

```bash
python -m src.experiments.run_experiments
python -m src.experiments.compare_results
```

The first command runs every solver on every benchmark instance and writes
CSV logs into `results/tables/` and `results/logs/`. The second command
produces comparison plots in `results/plots/`.

## Sample output

Running the SA on `small_01.json` (6 customers, 2 drones):

```
[SA] best energy = 1328.980   feasible = True   time = 1.23s   acceptance = 8.0%
  drone 0: 0 -> 2 -> 1 -> 5 -> 0
  drone 1: 0 -> 3 -> 6 -> 4 -> 0
```

Routes are printed as customer IDs; `0` is the depot.

## Benchmark results (10 instances, 3 seeds for the metaheuristics)

Best feasible energy per `(instance, method)`. NN, MILP and B&B are
deterministic and run once; GA and SA are run with 3 seeds and the best
of three is reported. Empty cells are methods we deliberately skipped on
that size class (MILP and B&B do not scale to 20+ customers under our
time limits).

| instance   | n  | K |   NN    |  MILP   |   B&B   |   GA    |   SA    |
|------------|----|---|---------|---------|---------|---------|---------|
| small_01   |  6 | 2 | 1623.99 | 1328.98 | 1328.98 | 1328.98 | 1328.98 |
| small_02   |  7 | 2 | 2129.00 | 1745.31 | 1745.31 | 1745.31 | 1745.31 |
| small_03   |  8 | 3 | 2463.42 | 1930.89 | 1930.89 | 1930.89 | 1930.89 |
| small_04   |  8 | 3 | 1933.13 | 1613.70 | 1613.70 | 1613.70 | 1613.70 |
| medium_01  | 12 | 3 | 2862.77 | 1984.02 |   --    | 1965.73 | 1965.73 |
| medium_02  | 14 | 3 | 3714.18 | 4116.69 |   --    | 3214.62 | 3214.62 |
| medium_03* | 15 | 4 | 4610.36 | 2922.20 |   --    | 2732.18 | 2644.66 |
| large_01   | 20 | 4 | 4708.14 |   --    |   --    | 4193.03 | 3944.79 |
| large_02   | 25 | 5 | 7054.50 |   --    |   --    | 5047.36 | 4634.77 |
| large_03*  | 30 | 5 | 8183.51 |   --    |   --    | 7001.08 | 6912.99 |

`*` instance contains no-fly zones.

Headline numbers (`results/tables/gap_table.csv` has the full gap-to-best
matrix):

- On every small instance the four "serious" methods (MILP, B&B, GA, SA)
  agree on the same optimum, which is a good sanity check. B&B finds it in
  about 0.01 s, the MILP between 5 and 13 s.
- On medium instances B&B is skipped (would not finish in time). The MILP
  is cut off at 60 s and is sometimes very far off — on `medium_02` it
  comes out at 4116.69 while GA/SA find 3214.62 (~28 % gap).
- On large instances both exact methods are skipped. SA wins all three,
  with GA within 1–9 %.
- NN is a fast warm-start (≤0.02 s) but always 15–75 % above the best
  energy, which is exactly the role we expect of a greedy baseline.
- See `results/plots/combined_summary.png` for the 2×2 dashboard used in
  the report (best energy by method, mean runtime, gap-to-best, runtime
  vs. instance size).

## Notes

- The MILP uses CBC, which is bundled with PuLP. No extra solver install needed.
- Branch and Bound is configured for instances with up to ~10 customers; the
  experimental pipeline skips it beyond that.
- No-fly zones are axis-aligned rectangles. The generator samples them
  first and then rejects any customer that lands inside a zone, so
  every instance is feasible by construction.
- SA mixes stochastic moves (swap, move, reverse, inter-drone swap) with a
  periodic best-improvement 2-opt sweep on each route, which substantially
  cleans up the late-stage solutions.
- Random seeds are fixed in the generator and in each metaheuristic so runs
  are reproducible.

## Reproducing the report figures

```bash
python -m src.utils.instance_generator      # writes 10 instances
python -m src.experiments.run_experiments   # ~12-15 minutes
python -m src.experiments.compare_results   # writes plots and gap table
```

The whole pipeline is single-threaded and finishes in around 12-15 minutes
on a 2024 laptop CPU. MILP and B&B have their own time limits in
`src/experiments/run_experiments.py` (60 / 30 seconds respectively).
Tweak them if you want a shorter or longer run.
