# ============================================================================
# fit_and_eval.jl — step 2 of the ablation: UDE fit + deterministic scoring as
# one callable the sweep can loop over. Reuses fit_ude (Src/UDE.jl), evaluate
# (Src/EvalAndRecover.jl), and generate_data (synthetic-data-gen.jl).
#
# Differences from UDE-fit.jl, on purpose:
#   • Input normalization (X_MEAN/X_STD) is recomputed from THIS data's Y_train
#     — the noisy, possibly sparse samples you'd actually have — instead of being
#     baked once from a clean Y_dense. Identical on the clean default data, so it
#     reproduces UDE-fit's diagnostics there; honest once noise/sparsity kick in.
#   • No symbolic recovery here (that's step 3). Returns the two deterministic
#     evaluate metrics plus the trained contract/θ so step 3 can run SR on it.
#
# NOTE: GreyParams / grey_rhs / ode_params are re-declared here; they also live
# in UDE-fit.jl. Run THIS sweep pipeline OR the UDE-fit/plotting demo in a given
# session, not both — GreyParams is a struct and Julia errors on struct
# redefinition. To dedupe later, move these three into a shared grey-model.jl and
# include it from both files.
# ============================================================================

using Lux, NNlib, ComponentArrays
using OrdinaryDiffEq, SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL, Zygote
using Random, Statistics
using Pkg
# using ReverseDiff   # uncomment if ReverseDiffVJP(true) complains it isn't loaded

const SRC = joinpath(dirname(Pkg.project().path), "Src")
include(joinpath(SRC, "UDE.jl"))
include(joinpath(SRC, "EvalAndRecover.jl"))
include(joinpath(@__DIR__, "synthetic-data-gen.jl"))   # generate_data, Y0, TRUE_PARAMS, true_g

const N_STATES = 3
const N_G      = 3

# ── Grey box: config-E structure; params FIXED at true (optimistic baseline) ──
struct GreyParams{T<:AbstractFloat}
    k::T; K::T; d_v::T; d_ifn::T; d_m::T
end

function grey_rhs(y, g, p::GreyParams)
    V, IFN, M = y
    logistic(x, r, cc) = r .* x .* (1 .- x ./ cc)
    dV   = logistic(V, p.k, p.K) .- V .* g[1, :] .- p.d_v   .* V
    dIFN = g[2, :]               .- p.d_ifn .* IFN
    dM   = g[3, :]               .- p.d_m   .* M
    permutedims(hcat(dV, dIFN, dM))
end

const ode_params = GreyParams(Float32(TRUE_PARAMS.k),   Float32(TRUE_PARAMS.K),
                              Float32(TRUE_PARAMS.d_v), Float32(TRUE_PARAMS.d_ifn),
                              Float32(TRUE_PARAMS.d_m))

# Robinson-cohort/-female's proven adjoint (not fit_ude's default QuadratureAdjoint,
# which QuadGK-NaNs when a BFGS probe drives the solve to the Float32 dt-floor).
const SENSEALG = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))

_silent(state, l) = false   # no per-iter printing during a sweep

"""
    fit_and_eval(data; seed=5) -> NamedTuple

Run the config-E schedule (Adam 1e-2 → Adam 1e-3 → BFGS, warm-started) on `data`,
then score the trained model against the CLEAN ground truth in `data`.

Input normalization is computed from `data.Y_train`, so nothing about the clean
diagnostics grid leaks into the model. `seed` seeds the network init only (the
data-noise seed lives in generate_data).

Returns:
  rel_l2_state   :: Vector{Float32}   per-state fit error vs clean truth (V,IFN,M)
  rel_l2_missing :: Vector{Float32}   per-term error of g vs clean f_true
  n_params       :: Int
  final_loss     :: Float32           training loss at θ★
  contract       :: NamedTuple        trained model (for step-3 symbolic recovery)
  θ              :: trained parameters
"""
function fit_and_eval(data; seed::Int = 5, λ::Real = 0)
    xmean = Float32.(vec(mean(data.Y_train; dims = 2)))
    xstd  = Float32.(vec(max.(std(data.Y_train; dims = 2), 1f-6)))

    # Channel scales from KNOWN grey params: additive channels g2,g3 ~ decay·state-scale;
    # g1 is a per-capita rate ~ O(1), NOT a state magnitude — do NOT use xstd there.
    s = Float32[1,                                   # g1: rate on V, O(1)
                ode_params.d_ifn * xstd[2],          # g2 ~ d_ifn · IFN-scale
                ode_params.d_m   * xstd[3]]          # g3 ~ d_m   · M-scale

    # g_builder = () -> Lux.Chain(
    #     Lux.WrappedFunction(x -> (x .- xmean) ./ xstd),
    #     Lux.Dense(N_STATES, 16, tanh),
    #     Lux.Dense(16, 16, tanh),
    #     Lux.Dense(16, N_G, softplus),
    #     Lux.WrappedFunction(y -> y .* s),            # fixed output rescale
    # )

    g_builder = () -> Lux.Chain(
        Lux.WrappedFunction(x -> (x .- xmean) ./ xstd),
        Lux.Dense(N_STATES, 16, tanh),
        Lux.Dense(16, 16, tanh),
        Lux.Dense(16, N_G, softplus),
    )

    reg = λ == 0 ? (_ -> 0f0) :
    θ -> λ * (sum(abs2, θ.layer_2) + sum(abs2, θ.layer_3))  

    r1 = fit_ude(data, grey_rhs, ode_params, Y0, g_builder;
                 seed = seed, opt = OptimizationOptimisers.Adam(1f-2),
                 maxiters = 1000, sensealg = SENSEALG,
                 callback = _silent,
                 reg = reg)
    r2 = fit_ude(data, grey_rhs, ode_params, Y0, g_builder;
                 seed = seed, opt = OptimizationOptimisers.Adam(1f-3),
                 maxiters = 3000, θ_init = r1.θ, sensealg = SENSEALG,
                 callback = _silent,
                 reg = reg)
    r3 = fit_ude(data, grey_rhs, ode_params, Y0, g_builder;
                 seed = seed, opt = OptimizationOptimJL.BFGS(initial_stepnorm = 1f-2),
                 # BFGS can autoterminate; maxiters is just for an upper bound
                 maxiters = 1000, θ_init = r2.θ, sensealg = SENSEALG,
                 callback = _silent,
                 reg = reg)

    ev = evaluate(r3.contract, data)
    return (rel_l2_state   = ev.rel_l2_state,
            rel_l2_missing = ev.rel_l2_missing,
            n_params       = ev.n_params,
            final_loss     = Float32(r3.loss(r3.θ, nothing)),
            contract       = r3.contract,
            θ              = r3.θ,
            retcode        = r3.retcode)
end
