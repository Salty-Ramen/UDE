# ============================================================================
# sweep.jl  (experiments/UDE-on-noisy-synthetic-data/Ablating-on-replicates/)
#
# Fit-pass DRIVER for the replicate question: at a FIXED temporal resolution T,
# does m=3 land within ~10% rel-L2 of the largest m tested? Total budget is NOT
# held fixed — that IS the question. Grid-agnostic runner: ../sweep-runner.jl.
#
# Run — interactive:
#   include("sweep.jl"); run_fit_sweep(CONFIGS, SWEEP_DIR)
#   sweep_status(CONFIGS, SWEEP_DIR)
#   # pilot without editing the grid — pass a filtered subset:
#   pilot = filter(c -> c["timepoints"]==20 && c["noise_frac"] in (0.1,0.2) && c["seed"]<=3, CONFIGS)
#   run_fit_sweep(pilot, SWEEP_DIR)         # 6 cells; rest resume later (isfile-skip)
#
# Run — GNU parallel, 8 shards:
#   seq 8 | parallel -j8 julia --project experiments/UDE-on-noisy-synthetic-data/Ablating-on-replicates/sweep.jl {} 8
# ============================================================================

using DrWatson
@quickactivate "UDE"
include(joinpath(@__DIR__, "..", "sweep-runner.jl"))

# ── Grid (edit here). (T, m-range): each T expands over its OWN replicate range,
#    so this is a DEPENDENT comprehension (mrange depends on the outer T) — not
#    dict_list, whose axes are independent. ────────────────────────────────────
const ALLOCATIONS = [(20, 3:6), (10, 3:6), (5, 3:6)]   # (timepoints, mice range)
const NOISE       = [0.0, 0.05, 0.10, 0.15, 0.20]
const SEEDS       = 1:5

const CONFIGS = [Dict("timepoints" => T, "mice_per_timepoint" => m,
                      "noise_frac" => ν, "seed" => s)
                 for (T, mrange) in ALLOCATIONS for m in mrange
                 for ν in NOISE for s in SEEDS]                    # ⇒ 11×5×5 = 275

const SWEEP_DIR = projectdir("experiments", "UDE-on-noisy-synthetic-data",
                             "Ablating-on-replicates", "Results", "sweep")

@info "fit-pass plan" n_cells = length(CONFIGS) dir = SWEEP_DIR

if abspath(PROGRAM_FILE) == @__FILE__
    k, N = length(ARGS) >= 2 ? (parse(Int, ARGS[1]), parse(Int, ARGS[2])) : (1, 1)
    run_fit_sweep(CONFIGS, SWEEP_DIR; shard = k, nshards = N)
end
