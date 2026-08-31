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

# ── Shape penalties on g as a function of state ──────────────────────────────
# `reg(θ)` is added to the loss OUTSIDE `predict`, so it is differentiated by
# Optimization.AutoZygote() (the outer reverse pass), not by `sensealg`. A
# generic AD for ∂g/∂x would therefore have to run nested inside Zygote:
# Zygote-over-ForwardDiff for the jacobian (ForwardDiff.jacobian seeds in place,
# which Zygote can't differentiate) and forward-over-forward for the curvature,
# which Lux's nested-AD helpers don't cover at all. So the input derivatives are
# written out here as a forward tangent pass in plain array arithmetic — Zygote
# sees only matmul and broadcast. Price: the Dense stack is hardcoded, and that
# assumption is asserted against g_builder in fit_and_eval below.
#
# Inputs are the NORMALISED states Z = (X .- xmean) ./ xstd, which is why the
# xstd_j / xstd_j² factors of the two penalties never appear: layer_1 IS that
# affine map, so ∂g_i/∂x_j · xstd_j = ∂g_i/∂z_j exactly.
"""
    g_tangent(θ, Z, Ż) -> (g, ġ, g̈)

Value plus first and second directional derivative of the g-network, at the
columns of `Z` (normalised states) along the matching columns of `Ż`:
`ġ[:, c] = ∂g/∂z · Ż[:, c]`, `g̈[:, c]` the second derivative in that same
direction. `Ż` is constant, so there is no z̈ term. Names: `d`/`dd` prefixes are
the first/second-order tangents (ȧ, ä) of each pre-activation.
"""
function g_tangent(θ, Z, Ż)
    W1, b1 = θ.layer_2.weight, θ.layer_2.bias
    W2, b2 = θ.layer_3.weight, θ.layer_3.bias
    W3, b3 = θ.layer_4.weight, θ.layer_4.bias

    a1   = W1 * Z .+ b1
    da1  = W1 * Ż                                  # dda1 = 0 (Ż constant)
    t1   = tanh.(a1);  p1 = 1f0 .- t1.^2           # tanh′
    dh1  = p1 .* da1
    ddh1 = (-2f0 .* t1 .* p1) .* da1.^2            # tanh″ · da1²

    a2   = W2 * t1 .+ b2
    da2  = W2 * dh1
    dda2 = W2 * ddh1
    t2   = tanh.(a2);  p2 = 1f0 .- t2.^2
    dh2  = p2 .* da2
    ddh2 = (-2f0 .* t2 .* p2) .* da2.^2 .+ p2 .* dda2

    a3   = W3 * t2 .+ b3
    da3  = W3 * dh2
    dda3 = W3 * ddh2
    q    = sigmoid.(a3 .- 3f0)                     # φ′, for φ = softplus(· − 3)
    return (softplus.(a3 .- 3f0),
            q .* da3,
            (q .* (1f0 .- q)) .* da3.^2 .+ q .* dda3)
end

"""
    g_shape_penalties(θ, Z, Ż, s) -> (P2, P3)

P2 = mean over (channel i, direction j, point b) of (∂g_i/∂z_j / s_i)²
P3 = same for (∂²g_i/∂z_j² / s_i)²

`s` is the channel-scale vector, so one λ means the same thing in all three
channels. `reg` weights the two with λ.jac / λ.curv.
"""
function g_shape_penalties(θ, Z, Ż, s)
    _, dg, ddg = g_tangent(θ, Z, Ż)
    return (Statistics.mean(abs2, dg ./ s), Statistics.mean(abs2, ddg ./ s))
end

"""
    fit_and_eval(data; seed=5) -> NamedTuple

Run the config-E schedule (Adam 1e-2 → Adam 1e-3 → BFGS, warm-started) on `data`,
then score the trained model against the CLEAN ground truth in `data`.

Input normalization is computed from `data.Y_train`, so nothing about the clean
diagnostics grid leaks into the model. `seed` seeds the network init only (the
data-noise seed lives in generate_data).

`λ = (w, jac, curv)` weights L² weight decay on all of θ, and the jacobian /
curvature penalties on g as a function of state (see `g_shape_penalties`).

Returns:
  rel_l2_state   :: Vector{Float32}   per-state fit error vs clean truth (V,IFN,M)
  rel_l2_missing :: Vector{Float32}   per-term error of g vs clean f_true
  n_params       :: Int
  final_loss     :: Float32           training loss at θ★
  contract       :: NamedTuple        trained model (for step-3 symbolic recovery)
  θ              :: trained parameters
"""
function fit_and_eval(data; seed::Int = 5, λ = (w = 0f0, jac = 0f0, curv = 0f0))
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

    # Output layer zero-initialised, so g ≡ 0 at θ0:
    # every seed starts from the pure mechanistic model and g grows only as the
    # data demands. (Glorot on hidden layers, zeros on final weights + biases —
    # Philipps et al., npj Syst Biol Appl 11:101, 2025.)

    g_builder = () -> Lux.Chain(
        Lux.WrappedFunction(x -> (x .- xmean) ./ xstd),
        Lux.Dense(N_STATES, 16, tanh),
        Lux.Dense(16, 16, tanh),
        Lux.Dense(16, N_G, x -> softplus(x - 3f0);   # softplus(0) ≈ log(2)
                  init_weight = Lux.zeros32, init_bias = Lux.zeros32),
    )

    # ── Collocation cloud for the shape penalties. Built ONCE here: reg runs on
    #    every BFGS line-search probe. ────────────────────────────────────────
    # Evaluated on a cloud filling the BOX the training states span, not on the
    # fitted trajectory. The data loss constrains g only along the 1-D curve the
    # solution traces through state space; off that curve the loss is flat, and
    # that flatness is exactly where the seed-to-seed spread in ĝ lives, so it is
    # the thing worth penalising. The box contains the curve and we do not
    # exclude it — excluding would need the θ-dependent trajectory, i.e. a
    # penalty whose support moves during the fit. A fixed grid (no RNG, never
    # resampled) keeps reg a deterministic function of θ, which the BFGS line
    # search requires.
    lo = vec(minimum(data.Y_train; dims = 2))
    hi = vec(maximum(data.Y_train; dims = 2))
    ax = [range(max(l - 0.1f0 * (h - l), 0f0), h + 0.1f0 * (h - l); length = 5)
          for (l, h) in zip(lo, hi)]                       # 10% wider than the data
    X_col = reduce(hcat, vec([Float32[a, b, c] for a in ax[1], b in ax[2], c in ax[3]]))
    M     = size(X_col, 2)                                 # 5³ = 125 points
    # One copy of the cloud per differentiation direction: column block j is the
    # whole cloud with direction e_j, so both penalties are a single pass.
    Xb = repeat(X_col, 1, N_STATES)
    Zb = (Xb .- xmean) ./ xstd
    Ż = Float32[i == (c - 1) ÷ M + 1 for i in 1:N_STATES, c in 1:N_STATES*M]

    # The architecture g_tangent writes out by hand, asserted rather than
    # assumed: its value branch must reproduce g_builder's network. Randomised
    # final layer, because at the production zero-init g is constant and a wrong
    # hidden activation would slip straight through.
    let net = g_builder(), rng = MersenneTwister(0)
        ps, st = Lux.setup(rng, net)
        ps.layer_4.weight .= randn(rng, Float32, size(ps.layer_4.weight))
        θc = ComponentArray(ps)
        @assert first(g_tangent(θc, Zb, Ż)) ≈ first(net(Xb, θc, st)) "g_tangent ≠ g_builder network"
    end

    # Penalise ALL of θ, mean-normalised: same footing as the mean-squared data
    # term, and no dependence on layer indexing.
    wd(θ) = λ.w * sum(abs2, θ) / length(θ)
    # Field access, not `all(iszero, λ)`: a caller still passing a scalar λ errors
    # here instead of silently running unregularised.
    reg = if λ.w == 0 && λ.jac == 0 && λ.curv == 0
        _ -> 0f0
    elseif λ.jac == 0 && λ.curv == 0
        wd
    else
        θ -> let (p2, p3) = g_shape_penalties(θ, Zb, Ż, s)
                 wd(θ) + λ.jac * p2 + λ.curv * p3
             end
    end
    
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
