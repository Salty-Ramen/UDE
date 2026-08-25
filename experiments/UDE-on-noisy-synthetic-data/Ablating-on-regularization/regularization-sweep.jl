# ============================================================================
# reg-sweep.jl  (experiments/UDE-on-noisy-synthetic-data/Ablating-on-regularization/)
#
# Fit-pass DRIVER for the regularization question: at each (T, m, noise, seed)
# of the replicates grid, how does L² weight-decay λ move rel-L2-vs-truth?
# λ=0 is NOT in this grid — it is the reused Ablating-on-replicates cache
# (control). Grid-agnostic runner: ../sweep-runner.jl.
#
# Run — interactive:
#   include("reg-sweep.jl"); run_fit_sweep(CONFIGS, SWEEP_DIR)
#   sweep_status(CONFIGS, SWEEP_DIR)
#   # pilot before committing the full grid — filter, don't edit:
#   pilot = filter(c -> c["timepoints"]==20 && c["seed"]<=2 && c["lambda"]==1e-2, CONFIGS)
#   run_fit_sweep(pilot, SWEEP_DIR)
#
# Run — GNU parallel, 8 shards:
#   seq 8 | parallel -j8 julia --project experiments/UDE-on-noisy-synthetic-data/Ablating-on-regularization/reg-sweep.jl {} 8
# ============================================================================

using DrWatson
@quickactivate "UDE"
include(joinpath(@__DIR__, "..", "sweep-runner.jl"))

# ── Base axes MUST match the frozen replicates control (so λ=0 lines up with
#    that cache). Verify length(CONFIGS_replicates) == #cached files first. ────
const ALLOCATIONS = [(20, 3:6), (10, 3:6), (5, 3:6)]   # copied from replicate-sweep.jl
const NOISE       = [0.0, 0.05, 0.10, 0.15, 0.20]
const SEEDS       = 1:5
const LAMBDAS     = [1e-4, 1e-3, 1e-2, 1e-1]           # λ=0 excluded ⇒ reused control

const CONFIGS = [Dict("timepoints" => T, "mice_per_timepoint" => m,
                      "noise_frac" => ν, "seed" => s, "lambda" => λ)
                 for (T, mrange) in ALLOCATIONS for m in mrange
                 for ν in NOISE for s in SEEDS for λ in LAMBDAS]

const SWEEP_DIR = projectdir("experiments", "UDE-on-noisy-synthetic-data",
                             "Ablating-on-regularization", "Results", "sweep")

@info "reg fit-pass plan" n_cells = length(CONFIGS) dir = SWEEP_DIR

if abspath(PROGRAM_FILE) == @__FILE__
    k, N = length(ARGS) >= 2 ? (parse(Int, ARGS[1]), parse(Int, ARGS[2])) : (1, 1)
    run_fit_sweep(CONFIGS, SWEEP_DIR; shard = k, nshards = N)
end
