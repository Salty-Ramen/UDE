# ============================================================================
# sweep.jl  (experiments/UDE-on-noisy-synthetic-data/Fixed-budget-sweep/)
#
# Fit-pass DRIVER: fixed mouse BUDGET reallocated across timepoints. Only the
# grid + output dir live here; the grid-agnostic runner (run_cell /
# run_fit_sweep / sweep_status) is in ../sweep-runner.jl.
#
# Run — interactive (does NOT run on include):
#   include("sweep.jl"); run_fit_sweep(CONFIGS, SWEEP_DIR)
#   sweep_status(CONFIGS, SWEEP_DIR)
#   run_fit_sweep(CONFIGS, SWEEP_DIR; force = true)
#
# Run — GNU parallel across PROCESSES (process-safe, NOT thread-safe). 8 shards:
#   seq 8 | parallel -j8 julia --project experiments/UDE-on-noisy-synthetic-data/Fixed-budget-sweep/sweep.jl {} 8
# One shard alone:
#   julia --project experiments/UDE-on-noisy-synthetic-data/Fixed-budget-sweep/sweep.jl 3 8
# ============================================================================

using DrWatson
@quickactivate "UDE"
include(joinpath(@__DIR__, "..", "sweep-runner.jl"))   # run_fit_sweep, sweep_status, run_cell, …

# ── Grid (edit here) ─────────────────────────────────────────────────────────
const BUDGET      = 120
const ALLOCATIONS = [(40, 3), (20, 6), (10, 12), (5, 24), (4, 30)]   # (timepoints, mice); T*m == BUDGET
const NOISE       = [0.0, 0.05, 0.10, 0.15, 0.20]
const SEEDS       = 1:5

for (T, m) in ALLOCATIONS
    @assert T * m == BUDGET "allocation ($T,$m) doesn't spend budget $BUDGET"
end

# (T,m) coupled (T*m==BUDGET) ⇒ manual product, not dict_list.
const CONFIGS = vec([Dict("timepoints" => T, "mice_per_timepoint" => m,
                          "noise_frac" => ν, "seed" => s)
                     for (T, m) in ALLOCATIONS, ν in NOISE, s in SEEDS])

const SWEEP_DIR = projectdir("experiments", "UDE-on-noisy-synthetic-data",
                             "Fixed-budget-sweep", "Results", "sweep")

@info "fit-pass plan" n_cells = length(CONFIGS) dir = SWEEP_DIR

# ── Script entry: `julia sweep.jl [shard nshards]`. Interactive include does
#    NOT trigger this (PROGRAM_FILE differs from this file).
if abspath(PROGRAM_FILE) == @__FILE__
    k, N = length(ARGS) >= 2 ? (parse(Int, ARGS[1]), parse(Int, ARGS[2])) : (1, 1)
    run_fit_sweep(CONFIGS, SWEEP_DIR; shard = k, nshards = N)
end
