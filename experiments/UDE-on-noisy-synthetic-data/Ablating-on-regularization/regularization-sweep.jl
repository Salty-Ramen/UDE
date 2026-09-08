# ============================================================================
# reg-sweep.jl  (experiments/UDE-on-noisy-synthetic-data/Ablating-on-regularization/)
#
# PILOT: does the discrepancy stop and the time-roughness penalty do the same
# job, and are they additive? Two mechanisms, crossed:
#   stop_kappa  0 = off, 1 = halt phases 2-3 when the DATA term reaches the
#               replicate-spread noise floor (Morozov). Not a tunable knob here:
#               the attainable misfit floors ~11% below τ, narrower than τ's own
#               ~8% sampling error, so κ is a switch.
#   lambda_dtt  second difference of g/s IN TIME along the fitted trajectory.
#               Anchored at λ★ = τ / P3_true (the penalty's value at the ORACLE
#               g) — the λ above which the true solution is penalised harder
#               than a noise-level misfit. Grid spans λ★ by decades.
#   lambda_dt   first difference in time. Pinned at 0: its maximum force lands
#               on the Hill switch (steep ∂g/∂V × fast V̇), i.e. on the sharpest
#               TRUE feature. Second order discriminates wiggle from signal
#               ~(timescale ratio)² better.
#   lambda_w    L² weight decay on θ. Pinned at 0 for this pilot.
# Grid-agnostic runner: ../sweep-runner.jl.
#
# λ★ anchor — run ONCE before launching, paste the result into LAM_DTT:
#   let dd = generate_data(timepoints=20, mice_per_timepoint=6, noise_frac=0.10, seed=1)
#       xs = Float32.(vec(max.(std(dd.Y_train; dims=2), 1f-6)))
#       s  = Float32[1, ode_params.d_ifn*xs[2], ode_params.d_m*xs[3]]
#       G  = Float32.(true_g(solve_true(Float64.(T_PEN)))) ./ s
#       d2 = (G[:,3:end] .- 2f0 .* G[:,2:end-1] .+ G[:,1:end-2]) ./ H_TAU^2
#       P3 = Statistics.mean(abs2, d2); τ = noise_floor(dd)
#       @info "λ anchor" P3 τ λ_star = τ / P3
#   end
#
# Run — interactive:
#   include("reg-sweep.jl"); run_fit_sweep(CONFIGS, SWEEP_DIR)
#   sweep_status(CONFIGS, SWEEP_DIR)
#   # pilot the pilot — filter, don't edit:
#   run_fit_sweep(filter(c -> c["seed"] == 1, CONFIGS), SWEEP_DIR)
#
# Run — GNU parallel, 8 shards:
#   seq 8 | parallel -j8 julia --project experiments/UDE-on-noisy-synthetic-data/Ablating-on-regularization/reg-sweep.jl {} 8
# ============================================================================

using DrWatson
@quickactivate "UDE"
include(joinpath(@__DIR__, "..", "sweep-runner.jl"))

const λ★ = 2.7341519f-5          # ← REPLACE with τ/P3_true from the anchor block above

# ── Axes. κ × λ_dtt is the factorial; everything else trimmed to 48 cells.
#    ν=0 dropped: τ=0 there, so the stop is inert and half the arms would be
#    duplicates. The clean-data control lives in the earlier sweeps.
const ALLOCATIONS = [(20, 6)]
const NOISE       = [0.10, 0.20]
const SEEDS       = 1:3
const LAM_W       = [0.0, 0.1, 0.01, 0.001]
const LAM_DT      = [0.0]
const LAM_DTT     = [0.0, 0.1 * λ★, λ★, 10*λ★]
const STOP_KAPPA  = [0.0, 1.0]
const OUT_RESCALE = [true, false]        # stage 2; flip to [true,false] for the scaler arm

# Every switch that changes what a cell MEANS is a config key, so it lands in
# savename and in the payload. output_rescale and stop_kappa are in here for
# exactly that reason — a kwarg-only switch collides on filename with its own
# other setting.
const CONFIGS = [Dict("timepoints" => T, "mice_per_timepoint" => m,
                      "noise_frac" => ν, "seed" => s,
                      "lambda_w" => λw, "lambda_dt" => λd, "lambda_dtt" => λc,
                      "stop_kappa" => κ, "output_rescale" => orc)
                 for (T, mrange) in ALLOCATIONS for m in mrange
                 for ν in NOISE for s in SEEDS
                 for λw in LAM_W for λd in LAM_DT for λc in LAM_DTT
                 for κ in STOP_KAPPA for orc in OUT_RESCALE]

const SWEEP_DIR = projectdir("experiments", "UDE-on-noisy-synthetic-data",
                             "Ablating-on-regularization", "Results", "sweep-stop-vs-dtt")

@info "reg fit-pass plan" n_cells = length(CONFIGS) λ_star = λ★ dir = SWEEP_DIR

if abspath(PROGRAM_FILE) == @__FILE__
    k, N = length(ARGS) >= 2 ? (parse(Int, ARGS[1]), parse(Int, ARGS[2])) : (1, 1)
    run_fit_sweep(CONFIGS, SWEEP_DIR; shard = k, nshards = N)
end
