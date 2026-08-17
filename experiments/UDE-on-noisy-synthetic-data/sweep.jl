# ============================================================================
# sweeper.jl — step 4: the FIT PASS of the ablation.
#
# Loops fit_and_eval over the grid and caches a numeric-only payload per cell to
# JLD2 (via DrWatson). Symbolic recovery is a SEPARATE downstream pass (step 5,
# recover_and_score.jl) over the cached curves — no refits — which is why the
# payload stores X_sr/g_sr/f_sr rather than the live contract/network.
#
# Design (all settled earlier):
#   • Interpretation A: fixed mouse BUDGET, reallocated across timepoints.
#   • Payload is plain numbers only — NO closures. The contract and g_NN close
#     over θ★, the ODEProblem, and a data-statistics lambda; JLD2 can mis-store
#     closures and fail cryptically on load. We persist θ★ as a raw vector; its
#     ComponentArray axes and the xmean/xstd normalization are regenerable from
#     (config, seed) via generate_data, so they're not stored (same reasoning as
#     skipping data CSVs — deterministic ⇒ regenerable).
#   • Resumable: a finished cell is skipped on the next run. A failed cell is
#     cached as an error payload so it doesn't wedge an unattended run; delete
#     its file to retry.
#
# Run — interactive (does NOT run on include):
#   include("sweeper.jl"); run_fit_sweep()      # whole grid in this process
#   sweep_status()                              # cached / remaining
#   run_fit_sweep(force=true)                   # ignore cache, recompute
#
# Run — GNU parallel across PROCESSES (this stack is process-safe, NOT
# thread-safe: the ODE adjoint + reverse-mode AD carry mutable state, so never
# @threads the fits). 8 shards on 8 jobs:
#   seq 8 | parallel -j8 julia --project sweeper.jl {} 8
# One shard alone:  julia --project sweeper.jl 3 8   # shard 3 of 8
# Shards are disjoint (CONFIGS[k:N:end]); the file cache + isfile-skip make it
# resumable — just re-run the same command after an interrupt. Pick -j by RAM,
# not just cores: each process holds a full SciML stack resident (~1–3 GB).
# ============================================================================

using DrWatson                     # savename, wsave, wload (JLD2 under the hood)
using Statistics, Pkg, LinearAlgebra
include(joinpath(@__DIR__, "UDE-fit.jl"))   # fit_and_eval, generate_data, true_g, Y0, ComponentArrays

# One BLAS thread per process. N heavy processes each spinning up BLAS threads
# would oversubscribe cores; each fit is ~one core of work anyway (tiny NN,
# serial ODE solve), so cores ≈ process concurrency.
BLAS.set_num_threads(1)

# ── Grid (edit here) ─────────────────────────────────────────────────────────
const BUDGET      = 120
const ALLOCATIONS = [(40, 3), (20, 6), (10, 12), (5, 24), (4, 30)]   # (timepoints, mice); T*m == BUDGET
const NOISE       = [0.0, 0.05, 0.10, 0.15, 0.20]
const SEEDS       = 1

for (T, m) in ALLOCATIONS
    @assert T * m == BUDGET "allocation ($T,$m) doesn't spend budget $BUDGET"
end

const SWEEP_DIR = joinpath(dirname(Pkg.project().path),
                           "experiments", "UDE-on-noisy-synthetic-data", "Results", "sweep")
mkpath(SWEEP_DIR)

const TG = collect(range(0f0, 8f0; length = 200))   # curve grid for SR + plots

# (T,m) are coupled (T*m==BUDGET), so a manual product — not dict_list, which
# assumes independent axes and would emit budget-violating combinations.
const CONFIGS = vec([Dict("timepoints" => T, "mice_per_timepoint" => m,
                          "noise_frac" => ν, "seed" => s)
                     for (T, m) in ALLOCATIONS, ν in NOISE, s in SEEDS])

cellname(c) = savename(c, "jld2")   # deterministic filename from the config dict
cellpath(c) = joinpath(SWEEP_DIR, cellname(c))

@info "fit-pass plan" n_cells = length(CONFIGS) dir = SWEEP_DIR

# ── One cell: generate → fit → sample curves. Numbers only. ──────────────────
function run_cell(c)
    data = generate_data(timepoints        = c["timepoints"],
                         mice_per_timepoint = c["mice_per_timepoint"],
                         noise_frac         = c["noise_frac"],
                         seed               = c["seed"])
    r = fit_and_eval(data; seed = c["seed"])
    X_sr = Float32.(r.contract.predict_state_raw(TG))   # 3×200 predicted states
    g_sr = Float32.(r.contract.predict_g_raw(TG))       # 3×200 learned g
    f_sr = Float32.(true_g(Float64.(X_sr)))             # 3×200 true g on those states
    Dict{String,Any}(
        c...,                                                       # config (provenance)
        "budget"     => c["timepoints"] * c["mice_per_timepoint"],
        "rel_l2_V"   => r.rel_l2_state[1],   "rel_l2_IFN"  => r.rel_l2_state[2],   "rel_l2_M"  => r.rel_l2_state[3],
        "rel_l2_gV"  => r.rel_l2_missing[1], "rel_l2_gIFN" => r.rel_l2_missing[2], "rel_l2_gM" => r.rel_l2_missing[3],
        "n_params"   => r.n_params,          "final_loss"  => r.final_loss,
        "theta"      => Float32.(ComponentArrays.getdata(r.θ)),     # raw; axes regenerable from (config,seed)
        "X_sr" => X_sr, "g_sr" => g_sr, "f_sr" => f_sr, "tg" => TG,
        "error" => "",
    )
end

# Cached marker for a failed cell — keeps columns consistent for collect_results.
function error_payload(c, err)
    Dict{String,Any}(
        c..., "budget" => c["timepoints"] * c["mice_per_timepoint"],
        "rel_l2_V" => NaN32, "rel_l2_IFN" => NaN32, "rel_l2_M" => NaN32,
        "rel_l2_gV" => NaN32, "rel_l2_gIFN" => NaN32, "rel_l2_gM" => NaN32,
        "n_params" => -1, "final_loss" => NaN32,
        "theta" => Float32[],
        "X_sr" => zeros(Float32, 3, 0), "g_sr" => zeros(Float32, 3, 0),
        "f_sr" => zeros(Float32, 3, 0), "tg" => TG,
        "error" => sprint(showerror, err),
    )
end

function run_fit_sweep(; force::Bool = false, shard::Int = 1, nshards::Int = 1)
    @assert 1 <= shard <= nshards "shard must be in 1:nshards"
    cells = CONFIGS[shard:nshards:end]        # disjoint cover across shards
    n = length(cells)
    @info "shard" shard nshards cells = n of = length(CONFIGS)
    for (i, c) in enumerate(cells)
        path = cellpath(c)
        if !force && isfile(path)
            @info "skip (cached)" shard i n file = basename(path); continue
        end
        @info "fit" shard i n T = c["timepoints"] m = c["mice_per_timepoint"] noise = c["noise_frac"] seed = c["seed"]
        local payload
        t = @elapsed payload = try
            run_cell(c)
        catch err
            @warn "cell failed" T = c["timepoints"] noise = c["noise_frac"] seed = c["seed"] exception = err
            error_payload(c, err)
        end
        # write-then-rename: a process killed mid-write leaves the .tmp, never a
        # half-written cell file. mv is atomic on the same filesystem.
        tmp = tempname(SWEEP_DIR) * ".jld2"
        wsave(tmp, payload)
        mv(tmp, path; force = true)
        @info "done" shard i n minutes = round(t / 60; digits = 1) rel_l2_state = (payload["rel_l2_V"], payload["rel_l2_IFN"], payload["rel_l2_M"]) file = basename(path)
    end
    @info "shard complete" shard nshards dir = SWEEP_DIR
end

function sweep_status()
    done = count(c -> isfile(cellpath(c)), CONFIGS)
    @info "sweep status" cached = done remaining = length(CONFIGS) - done total = length(CONFIGS) dir = SWEEP_DIR
    return done
end

# ── Script entry: `julia sweeper.jl [shard nshards]` runs that shard and exits.
#    Interactive `include("sweeper.jl")` does NOT trigger this (PROGRAM_FILE
#    differs from this file), so the REPL workflow above is unchanged.
if abspath(PROGRAM_FILE) == @__FILE__
    k, N = length(ARGS) >= 2 ? (parse(Int, ARGS[1]), parse(Int, ARGS[2])) : (1, 1)
    run_fit_sweep(; shard = k, nshards = N)
end
