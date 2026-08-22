# ============================================================================
# sweep-runner.jl — grid-agnostic FIT-PASS runner, shared across ablations.
# Lives at experiments/UDE-on-noisy-synthetic-data/. Each ablation's driver
# (Fixed-budget-sweep/sweep.jl, Ablating-on-replicates/sweep.jl, …) defines its
# own CONFIGS + SWEEP_DIR and calls run_fit_sweep(CONFIGS, SWEEP_DIR).
#
# Extracted verbatim from the original sweep.jl; the ONLY behavioural change is
# that CONFIGS and SWEEP_DIR are now function ARGUMENTS, not module globals —
# so the same runner serves any grid, and you can pilot by passing a filtered
# subset of CONFIGS. Payload schema is unchanged (collect_results-compatible).
#
# Assumes the DRIVER has already run `@quickactivate "UDE"` before including
# this file (projectdir/Pkg path resolution below depends on the active project).
# ============================================================================

using DrWatson                     # savename, wsave (JLD2 under the hood)
using LinearAlgebra                # BLAS
include(joinpath(@__DIR__, "UDE-fit.jl"))   # fit_and_eval, generate_data, true_g, ComponentArrays

# One BLAS thread per process — N heavy processes each spinning BLAS threads
# would oversubscribe cores; each fit is ~one core (tiny NN, serial ODE solve).
BLAS.set_num_threads(1)

const TG = collect(range(0f0, 8f0; length = 200))   # curve grid for SR + plots

cellname(c)            = savename(c, "jld2")          # deterministic filename
cellpath(sweep_dir, c) = joinpath(sweep_dir, cellname(c))

# ── One cell: generate → fit → sample curves. Numbers only (no closures). ────
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

function run_fit_sweep(configs, sweep_dir; force::Bool = false,
                       shard::Int = 1, nshards::Int = 1)
    @assert 1 <= shard <= nshards "shard must be in 1:nshards"
    mkpath(sweep_dir)
    cells = configs[shard:nshards:end]        # disjoint cover across shards
    n = length(cells)
    @info "shard" shard nshards cells = n of = length(configs) dir = sweep_dir
    for (i, c) in enumerate(cells)
        path = cellpath(sweep_dir, c)
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
        tmp = tempname(sweep_dir) * ".jld2"
        wsave(tmp, payload)
        mv(tmp, path; force = true)
        @info "done" shard i n minutes = round(t / 60; digits = 1) rel_l2_state = (payload["rel_l2_V"], payload["rel_l2_IFN"], payload["rel_l2_M"]) file = basename(path)
    end
    @info "shard complete" shard nshards dir = sweep_dir
end

function sweep_status(configs, sweep_dir)
    done = count(c -> isfile(cellpath(sweep_dir, c)), configs)
    @info "sweep status" cached = done remaining = length(configs) - done total = length(configs) dir = sweep_dir
    return done
end
