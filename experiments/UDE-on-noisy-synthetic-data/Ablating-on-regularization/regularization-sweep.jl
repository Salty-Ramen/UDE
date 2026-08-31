# ============================================================================
# reg-sweep.jl  (experiments/UDE-on-noisy-synthetic-data/Ablating-on-regularization/)
#
# Fit-pass DRIVER for the regularization question: at each (T, m, noise, seed),
# how do the three penalty weights move rel-L2-vs-truth?
#   lambda_w    L² weight decay on all of θ          (parameter-space shrinkage)
#   lambda_jac  mean (∂g_i/∂x_j · xstd_j / s_i)²     (state-space flatness)
#   lambda_curv mean (∂²g_i/∂x_j² · xstd_j² / s_i)²  (state-space affineness)
# All three are swept independently, and (0,0,0) is IN this grid — the control is
# no longer the borrowed Ablating-on-replicates cache, because the λ key rename
# means no pre-rename cell file lines up anyway. Grid-agnostic runner:
# ../sweep-runner.jl.
#
# Run — interactive:
#   include("reg-sweep.jl"); run_fit_sweep(CONFIGS, SWEEP_DIR)
#   sweep_status(CONFIGS, SWEEP_DIR)
#   # pilot before committing the full grid — filter, don't edit:
#   pilot = filter(c -> c["seed"] == 1 && c["noise_frac"] == 0.1, CONFIGS)
#   run_fit_sweep(pilot, SWEEP_DIR)
#
# Run — GNU parallel, 8 shards:
#   seq 8 | parallel -j8 julia --project experiments/UDE-on-noisy-synthetic-data/Ablating-on-regularization/reg-sweep.jl {} 8
# ============================================================================

using DrWatson
@quickactivate "UDE"
include(joinpath(@__DIR__, "..", "sweep-runner.jl"))

# This is for the PRIOR sweep attempts:
# const ALLOCATIONS = [(20, 6)]   # copied from replicate-sweep.jl
# const NOISE       = [0.0, 0.05, 0.10, 0.15, 0.20]
# const SEEDS       = 1:5
# const LAMBDAS     = [1e-4, 1e-3, 1e-2, 1e-1]           # λ=0 excluded ⇒ reused control


# ── Axes. Cell count = ∏ of all of these, so the two λ shape axes are the
#    factorial and everything else is trimmed to keep the first pass ~144 cells.
const ALLOCATIONS = [(20, 6)]
const NOISE       = [0.0, 0.10, 0.20]
const SEEDS       = 1:3
# lambda_w on one level to start: for the SR goal P2/P3 shrink g in STATE space,
# which is the same job weight decay was doing in parameter space, so w is the
# axis most likely to be redundant. Re-add [0.0, 1e-3, 1e-1] once the jac×curv
# plane is read — the plotter picks up new axes automatically.
const LAM_W       = [0.0]
const LAM_JAC     = [0.0, 1e-3, 1e-2, 1e-1]
const LAM_CURV    = [0.0, 1e-3, 1e-2, 1e-1]

const CONFIGS = [Dict("timepoints" => T, "mice_per_timepoint" => m,
                      "noise_frac" => ν, "seed" => s,
                      "lambda_w" => λw, "lambda_jac" => λj, "lambda_curv" => λc)
                 for (T, mrange) in ALLOCATIONS for m in mrange
                 for ν in NOISE for s in SEEDS
                 for λw in LAM_W for λj in LAM_JAC for λc in LAM_CURV]

const SWEEP_DIR = projectdir("experiments", "UDE-on-noisy-synthetic-data",
                             "Ablating-on-regularization", "Results", "sweep")

@info "reg fit-pass plan" n_cells = length(CONFIGS) dir = SWEEP_DIR

if abspath(PROGRAM_FILE) == @__FILE__
    k, N = length(ARGS) >= 2 ? (parse(Int, ARGS[1]), parse(Int, ARGS[2])) : (1, 1)
    run_fit_sweep(CONFIGS, SWEEP_DIR; shard = k, nshards = N)
end
