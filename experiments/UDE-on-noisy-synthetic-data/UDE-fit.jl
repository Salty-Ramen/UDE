# ============================================================================
# fit_and_eval.jl — step 2 of the ablation: UDE fit + deterministic scoring as
# one callable the sweep can loop over. Reuses fit_ude (Src/UDE.jl), evaluate
# (Src/EvalAndRecover.jl), and generate_data (synthetic-data-gen.jl).
#
# Differences from the Src/UDE-fit demo script this was derived from, on purpose:
#   • Input normalization (X_MEAN/X_STD) is recomputed from THIS data's Y_train
#     — the noisy, possibly sparse samples you'd actually have — instead of being
#     baked once from a clean Y_dense. Identical on the clean default data, so it
#     reproduces UDE-fit's diagnostics there; honest once noise/sparsity kick in.
#   • No symbolic recovery here (that's step 3). Returns the two deterministic
#     evaluate metrics plus the trained contract/θ so step 3 can run SR on it.

# NOTE: GreyParams / grey_rhs / ode_params are re-declared here; they also live
# in the standalone UDE-fit/plotting demo. Run THIS sweep pipeline OR that demo
# in a given session, not both — GreyParams is a struct and Julia errors on
# struct redefinition. To dedupe later, move these three into a shared
# grey-model.jl and include it from both files.
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

# Roughness-penalty time grid. 50 points over the 8-day span ≈ 0.16 d spacing:
# fine enough to resolve overfit wiggle (Nyquist on 20 timepoints is ~0.8 d),
# coarse enough that the 1/h² amplification of solver noise (~2.4e3) stays well
# below the true signal.

const T_PEN = collect(range(T_SPAN[1], T_SPAN[2]; length = 50))
const H_TAU = 1f0 / (length(T_PEN) - 1)        # step, time normalised to [0,1]

"""
    noise_floor(data) -> Float32

The value fit_ude's data term is expected to reach when the prediction equals
the TRUE states — the level below which further descent is fitting noise.
Estimated from replicate spread: with m mice at one timepoint, the sample
variance across mice IS the noise magnitude there, so no oracle and no knowledge
of `noise_frac` is needed and the same number is computable on a real cohort.
Uses the same per-state σ as fit_ude's data term, so the two are directly
comparable. Returns 0 both when no timepoint has ≥2 mice (cannot estimate) and
when noise_frac = 0 (replicates are bit-identical, so there is genuinely no
floor). The two are currently indistinguishable in the output.
"""
function noise_floor(data)
    σ2 = abs2.(state_scale(data.Y_train))
    t  = vec(data.t_train)
    per_t = [Statistics.mean(vec(var(data.Y_train[:, cols]; dims = 2)) ./ σ2)
             for cols in (findall(==(tv), t) for tv in unique(t))
             if length(cols) >= 2]
    isempty(per_t) ? 0f0 : Float32(Statistics.mean(per_t))
end

"""
    err_weights(data, model) -> (w, floor)

Observation weights w = 1/σ (0 = masked) paired with the value fit_ude's data
term reaches at the TRUE states, ON THE SAME SCALE as w. The pair must travel
together: scaling w by c scales the data term by c², so w from one model with a
floor from another makes the discrepancy stop fire in the wrong place silently.

"spread" — w_i = 1/state_scale(Y_train)[i], one weight per state (fit_ude's
    historical default; floor from `noise_floor`, which divides replicate
    variance by that same state_scale). Matched empirically: it assumes nothing
    about the SHAPE of the noise, only measures its size in these units.

"prop"   — matches the generator's y = x(1 + ν·ε): σ_ij = ν̂_i · ȳ_i(t_k), one
    pooled CV per state times the REPLICATE MEAN at that observation's timepoint.
    Not the prediction — ŷ in the denominator is asymmetric and blows up as
    ŷ→0. floor = 1 by construction, since each term is then (error ÷ its own σ)²,
    which averages to 1 when the model is right.

    ν not estimable (m = 1, or noise_frac = 0 so replicates are bit-identical)
    ⇒ ν̂_i = 1 and floor = 0. Not a special case: a constant factor on the loss
    does not move the minimiser, and since the generator uses one noise_frac for
    all states the ν̂_i are near-equal, so dropping them fits identically. floor
    = 0 disables the stop, the only consumer of the absolute scale.
"""
function err_weights(data, model::AbstractString)
    if model == "spread"
        return (1f0 ./ state_scale(data.Y_train), noise_floor(data))
    end
    model == "prop" ||
        error("unknown err_model $(repr(model)) — want \"spread\" or \"prop\"")

    Y  = data.Y_train
    t  = vec(data.t_train)
    n  = size(Y, 1)
    cols = [findall(==(tv), t) for tv in unique(t)]      # replicate groups
    T    = length(cols)

    # Replicate mean per (state, timepoint), and the pooled CV² — averaged
    # EQUALLY over timepoints, not signal-weighted, so timepoints where a state
    # is small still count.

    ȳ   = zeros(Float32, n, T)
    acc = zeros(Float64, n)
    cnt = zeros(Int, n)
    for k in 1:T, i in 1:n
        c = cols[k]
        ȳ[i, k] = Statistics.mean(Y[i, c])
        if length(c) >= 2 && ȳ[i, k] != 0 && !allequal(@view Y[i, c])
            acc[i] += Statistics.var(Y[i, c]) / ȳ[i, k]^2
            cnt[i] += 1
        end
    end
    ν̂  = Float32[cnt[i] > 0 ? sqrt(acc[i] / cnt[i]) : 1f0 for i in 1:n]
    fl = all(>(0), cnt) ? 1f0 : 0f0

    W = zeros(Float32, n, size(Y, 2))
    for k in 1:T, j in cols[k], i in 1:n
        σ = ν̂[i] * ȳ[i, k]
        W[i, j] = σ > 0 ? 1f0 / σ : 0f0   # ȳ=0 ⇒ mask; dividing first gives Inf*0
    end
    return (W, fl)
end


"""
    g_time_penalties(g_net, g_st, θ, X_pen, s) -> (P2, P3)

Roughness of g **in time along the fitted trajectory**: mean squared first and
second central difference of g on the uniform `T_PEN` grid, time normalised to
[0,1]. `X_pen` is the predicted states at those times. Dividing g by the channel
scale `s` first puts all three channels on one footing, so one λ means one thing.
"""
function g_time_penalties(g_net, g_st, θ, X_pen, s)
    G  = first(g_net(X_pen, θ, g_st)) ./ s
    d1 = (G[:, 3:end] .- G[:, 1:end-2]) ./ (2f0 * H_TAU)
    d2 = (G[:, 3:end] .- 2f0 .* G[:, 2:end-1] .+ G[:, 1:end-2]) ./ H_TAU^2
    (Statistics.mean(abs2, d1), Statistics.mean(abs2, d2))
end



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
    fit_and_eval(data; seed::Int = 5, λ = (w = 0f0, dt = 0f0, dtt = 0f0),
                      output_rescale::Bool = true, stop_kappa = 0f0,
                      err_model::AbstractString = "spread")


Run the config-E schedule (Adam 1e-2 → Adam 1e-3 → BFGS, warm-started) on `data`,
then score the trained model against the CLEAN ground truth in `data`.

Input normalization is computed from `data.Y_train`, so nothing about the clean
diagnostics grid leaks into the model. `seed` seeds the network init only (the
data-noise seed lives in generate_data) — note the sweep currently passes the
same integer to both unless "init_seed" is set in the config.

`λ = (w, dt, dtt)` weights L² weight decay on all of θ, and the first- and
second-difference roughness of g IN TIME along the fitted trajectory (see
`g_time_penalties`) — not in state space.

`output_rescale` appends the fixed `.* s` output layer (stage 2, default); false
reproduces the pre-rescale regime (stage 1).

Returns:
  rel_l2_state   :: Vector{Float32}   per-state fit error vs clean truth (V,IFN,M)
  rel_l2_missing :: Vector{Float32}   per-term error of g vs clean f_true
  n_params       :: Int
  final_loss     :: Float32           training loss at θ★ (data term + reg)
  data_loss      :: Float32           data term ALONE at θ★ (comparable across λ)
  contract       :: NamedTuple        trained model (for step-3 symbolic recovery)
  θ              :: trained parameters
  retcode        :: BFGS phase return code ("Success" / "MaxIters" / …)
  bfgs_iters, bfgs_fevals :: Int      phase-3 accepted iterations / f evaluations
"""
function fit_and_eval(data; seed::Int = 5, λ = (w = 0f0, dt = 0f0, dtt = 0f0),
                      output_rescale::Bool = true, stop_kappa = 0f0,
                      err_model::AbstractString = "spread")

    xmean = Float32.(vec(mean(data.Y_train; dims = 2)))
    xstd  = Float32.(vec(max.(std(data.Y_train; dims = 2), 1f-6)))

    # Channel scales from KNOWN grey params: additive channels g2,g3 ~ decay·state-scale;
    # g1 is a per-capita rate ~ O(1)
    s = Float32[1,                                   # g1: rate on V, O(1)
                ode_params.d_ifn * xstd[2],          # g2 ~ d_ifn · IFN-scale
                ode_params.d_m   * xstd[3]]          # g3 ~ d_m   · M-scale

    # Output rescale restored. Without it the trunk (tanh-bounded, O(1)) has to
    # reach g2 ≈ 136 through softplus, forcing final-layer weights ~8.5 from a
    # ZERO init behind a sigmoid(−3) ≈ 0.05 gradient gate. With `.* s` the same
    # target is a pre-rescale value ≈ 2.9, i.e. weights ~0.4 — reachable. It also
    # makes λ_w scale-fair: the data term divides by σ_state and the roughness
    # penalty divides by s, so the raw network output was the ONE unnormalised
    # quantity, and sum(abs2,θ) was measuring output magnitude, not shape.
    # NB g is no longer identically 0 at θ0: it is softplus(−3)·s ≈ 0.05·s,
    # which is a few percent of each channel's true peak.

    # output_rescale = false reproduces the pre-rescale regime (stage 1): layer_4
    # weights carry the raw channel magnitude, so scalar λ_w hits channels in
    # order of output size. true (default) is the scale-fair version (stage 2).
    layers = (Lux.WrappedFunction(x -> (x .- xmean) ./ xstd),
              Lux.Dense(N_STATES, 16, tanh),
              Lux.Dense(16, 16, tanh),
              Lux.Dense(16, N_G, x -> softplus(x - 3f0);
                        init_weight = Lux.zeros32, init_bias = Lux.zeros32))
    g_builder = output_rescale ?
        () -> Lux.Chain(layers..., Lux.WrappedFunction(y -> y .* s)) :
        () -> Lux.Chain(layers...)
    
    # lo = vec(minimum(data.Y_train; dims = 2))
    # hi = vec(maximum(data.Y_train; dims = 2))
    # ax = [range(max(l - 0.1f0 * (h - l), 0f0), h + 0.1f0 * (h - l); length = 5)
    #       for (l, h) in zip(lo, hi)]                       # 10% wider than the data
    # X_col = reduce(hcat, vec([Float32[a, b, c] for a in ax[1], b in ax[2], c in ax[3]]))
    # M     = size(X_col, 2)                                 # 5³ = 125 points
    # One copy of the cloud per differentiation direction: column block j is the
    # whole cloud with direction e_j, so both penalties are a single pass.
    # Xb = repeat(X_col, 1, N_STATES)
    # Zb = (Xb .- xmean) ./ xstd
    # Ż = Float32[i == (c - 1) ÷ M + 1 for i in 1:N_STATES, c in 1:N_STATES*M]
    
    wd(θ) = λ.w * sum(abs2, θ) / length(θ)
    use_shape = !(λ.dt == 0 && λ.dtt == 0)
    reg = if λ.w == 0 && !use_shape
        (θ, _) -> 0f0
    elseif !use_shape
        (θ, _) -> wd(θ)
    else
        g_net = g_builder()
        g_st  = last(Lux.setup(MersenneTwister(0), g_net))   # st only; θ comes from the optimizer
        (θ, X_pen) -> let (p2, p3) = g_time_penalties(g_net, g_st, θ, X_pen, s)
            wd(θ) + λ.dt * p2 + λ.dtt * p3
        end
    end

    t_pen = use_shape ? T_PEN : nothing

    w_obs, τ = err_weights(data, err_model)
    stop = Float32(stop_kappa) * τ    # 0 when κ=0, ν=0, or m=1 ⇒ stop disabled

    
    r1 = fit_ude(data, grey_rhs, ode_params, Y0, g_builder;
                 seed = seed, opt = OptimizationOptimisers.Adam(1f-2),
                 maxiters = 1000, sensealg = SENSEALG,
                 callback = _silent,
                 reg = reg, t_pen = t_pen,
                 w_obs = w_obs)
    r2 = fit_ude(data, grey_rhs, ode_params, Y0, g_builder;
                 seed = seed, opt = OptimizationOptimisers.Adam(1f-3),
                 maxiters = 3000, θ_init = r1.θ, sensealg = SENSEALG,
                 callback = _silent, reg = reg, t_pen = t_pen,
                 stop_below = stop,
                 w_obs = w_obs)

    # BFGS past the floor is exactly the descent the discrepancy principle skips.
    r3 = r2.stopped ? nothing :
         fit_ude(data, grey_rhs, ode_params, Y0, g_builder;
                 seed = seed, opt = OptimizationOptimJL.BFGS(initial_stepnorm = 1f-2),
                 maxiters = 1000, θ_init = r2.θ, sensealg = SENSEALG,
                 callback = _silent, reg = reg, t_pen = t_pen,
                 stop_below = stop,
                 w_obs = w_obs)
    rf = r3 === nothing ? r2 : r3

    ev = evaluate(rf.contract, data)
    return (rel_l2_state   = ev.rel_l2_state,
            rel_l2_missing = ev.rel_l2_missing,
            n_params       = ev.n_params,
            final_loss     = Float32(rf.loss(rf.θ, nothing)),
            data_loss      = Float32(rf.data_loss(rf.θ)),
            noise_floor    = τ,
            stopped        = rf.stopped,
            stop_phase     = rf.stopped ? (r3 === nothing ? 2 : 3) : 0,
            stop_iter      = rf.stop_iter,
            contract       = rf.contract,
            θ              = rf.θ,
            retcode        = rf.retcode,
            bfgs_iters     = r3 === nothing ? 0 : r3.stats.iterations,
            bfgs_fevals    = r3 === nothing ? 0 : r3.stats.fevals)
end
