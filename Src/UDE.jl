#=-------------------------------------------------------------------------------
UDE.jl

Universal Differential Equation fitter, as a single plain function returning the
same output contract as AI-Aristotle (consumed by eval_and_recover.jl).

The difference from AI-Aristotle: there is NO state network. The state comes
from integrating the ODE, and the only trainable object is the missing-term
network, which takes the STATE as input (not time):

    dy/dt = architecture(y, g_NN(y), ode_params),   integrated from y0 over t_span.

`architecture` is the SAME grey-box RHS both methods share. AI-Aristotle calls
it with g = g_net(t); UDE calls it with g = g_NN(y). The one wrinkle is that
`architecture` is written batched (n_states × B), while the solver calls the RHS
pointwise with a single state vector — so the RHS shims one column in and `vec`s
the result out. `architecture` itself is untouched.

Contract produced from the trained params θ★:
    predict_state_raw(t_grid) = solve-and-sample          (IC exact: integrates from y0)
    predict_g_raw(t_grid)     = g_NN(predict_state_raw(t)) (already raw)
    n_params                  = length(θ★)                 (g-network only)

Fixed choices / notes:
  - ODE params are FIXED (captured in the RHS closure, not trained). Fitting them
    later (the fixed/prior/fitted ablation) means wrapping θ as
    ComponentArray(gMLP = …, ode = …) and reading ode from p inside the RHS;
    none of the rest changes.
  - Positivity is NOT enforced. On the virus/IFN/M system V=0 is invariant under
    the true flow (dV = V·(…)), so the exact solution cannot cross zero, but the
    numerical one can undershoot to ~-1e-8 as V decays, and `abstol` permits it.
    Nothing in the RHS or the loss breaks there; the exposure is downstream, where
    a non-integer power of a state throws (see the V-clamp in true_g). If a future
    system needs hard positivity, prefer log-state integration or `isoutofdomain`
    over an in-RHS `max`/`clamp` — both keep the comparison out of the
    differentiated RHS, so `ReverseDiffVJP(true)` stays correct.
  - `sensealg` defaults to QuadratureAdjoint(ReverseDiffVJP(true)). The `true`
    (compiled tape) is valid only because the RHS is branch-free: the grey-box
    arithmetic plus tanh and softplus, all smooth, no value- or time-dependent
    control flow. Adding a clamp, ReLU, or dosing switch invalidates the tape —
    drop the `true` or switch to ZygoteVJP/EnzymeVJP. Note the experiments
    override this default: UDE-fit.jl passes InterpolatingAdjoint(ReverseDiffVJP
    (true)) because QuadratureAdjoint's QuadGK returns NaN when a BFGS probe
    drives the solve to the Float32 dt-floor.
-------------------------------------------------------------------------------=#

using Lux
using ComponentArrays
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization, OptimizationOptimisers
using Zygote
using Statistics
using Random

# If `QuadratureAdjoint(ReverseDiffVJP(true))` errors that ReverseDiff is not
# loaded, add `using ReverseDiff` in the experiment.

"""
    ude_print_callback(state, l) -> false

Minimal default callback; prints every 200 iterations.
"""
function ude_print_callback(state, l)
    state.iter % 200 == 0 && println("iter = ", state.iter, "  loss = ", l)
    return false
end

# The per-state scale the data term divides by. Shared with noise_floor so the
# floor and the misfit are on one scale — they are comparable only if identical.
state_scale(Y) = Float32.(vec(max.(std(Y; dims = 2), 1f-6)))

"""
    fit_ude(data, architecture, ode_params, y0, g_builder;
                 seed::Int        = 1,
                 opt              = OptimizationOptimisers.Adam(1f-2),
                 maxiters::Int    = 1000,
                 callback         = ude_print_callback,
                 solver           = AutoTsit5(Rosenbrock23()),
                 sensealg         = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)),
                 abstol::Float32  = 1f-6,
                 reltol::Float32  = 1f-6,
                 stop_below::Float32 = 0f0,
                 stop_every::Int  = 25,
                 reg              = (_, X_pen) -> 0f0,
                 t_pen            = nothing,
                 θ_init           = nothing)

Fit a UDE on `data` (needs `t_train` 1×Ntrain, `Y_train` n_states×Ntrain,
`t_span` length-2). `g_builder()` constructs the missing-term Lux network
(input = n_states, output = n_g). `ode_params` is whatever `architecture`
expects and is held fixed.

Returns:
  contract :: (predict_state_raw, predict_g_raw, n_params)   — for eval_and_recover
  θ        :: trained parameters (ComponentArray, g-network only)
  predict  :: (θ, t_grid) -> raw states      (for warm-start / inspection)
  loss     :: (θ, _) -> Float32

Warm-start a second phase by passing the returned `θ` back as `θ_init`.
"""
function fit_ude(data, architecture, ode_params, y0, g_builder;
                 seed::Int        = 1,
                 opt              = OptimizationOptimisers.Adam(1f-2),
                 maxiters::Int    = 1000,
                 callback         = ude_print_callback,
                 solver           = AutoTsit5(Rosenbrock23()),
                 sensealg         = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)),
                 abstol::Float32  = 1f-6,
                 reltol::Float32  = 1f-6,
                 stop_below::Float32 = 0f0,
                 stop_every::Int  = 25,
                 reg              = (_, X_pen) -> 0f0,
                 t_pen            = nothing,
                 θ_init           = nothing)

    rng = MersenneTwister(seed)
    g_NN = g_builder()
    ps_g, st_g = Lux.setup(rng, g_NN)
    θ0 = θ_init === nothing ? ComponentArray(ps_g) : θ_init

    # Per-state scale for the data loss (so a large-magnitude state can't dominate).
    σ_state = state_scale(data.Y_train)

    tspan = (Float32(data.t_span[1]), Float32(data.t_span[2]))

    # Batched architecture ↔ pointwise solver shim: reshape u → (n×1), vec back.
    function ude_rhs!(du, u, p, t)
        umat = reshape(u, :, 1)
        gmat = first(g_NN(umat, p, st_g))               # (n_g × 1)
        du .= vec(architecture(umat, gmat, ode_params)) # (n_states,)
        return nothing
    end

    prob = ODEProblem(ude_rhs!, y0, tspan, θ0)
    # Unique solve times + a map from each observation column to its solve column,
    # so replicates (repeated timepoints) all contribute to the loss while the
    # adjoint only ever sees unique saveat times.
    t_obs        = vec(data.t_train)
    t_unique     = sort(unique(t_obs))
    # Penalty times are unioned into saveat, so ONE solve serves both the data
    # term and the roughness penalty. t_pen is uniform, so the columns pulled out
    # by col_of_pen are uniformly spaced in time (t_all itself is not).
    t_pen_v      = t_pen === nothing ? Float32[] : Float32.(vec(t_pen))
    t_all        = sort(unique(vcat(t_unique, t_pen_v)))
    col_of_obs   = [searchsortedfirst(t_all, t) for t in t_obs]
    col_of_pen   = [searchsortedfirst(t_all, t) for t in t_pen_v]
    t_all_row    = permutedims(t_all)

    predict(θ, t_grid) = Array(solve(remake(prob; p = θ), solver;
                                     saveat = vec(t_grid),
                                     abstol = abstol, reltol = reltol,
                                     sensealg = sensealg))

    # ONE definition of the data term. `loss` reuses the solve it already needs
    # for the penalty; `data_loss` is standalone, for the discrepancy check and
    # for reporting (the objective is NOT comparable across λ — it carries reg).
    _data_term(sol) = Statistics.mean(abs2,
                          (sol[:, col_of_obs] .- data.Y_train) ./ σ_state)

    function loss(θ, _)
        sol = predict(θ, t_all_row)
        return _data_term(sol) + reg(θ, sol[:, col_of_pen])
    end

    data_loss(θ) = _data_term(predict(θ, t_all_row))

    # Discrepancy stop (Morozov): halt when the DATA term alone reaches the noise
    # floor. state.u is the accepted iterate — the callback fires per accepted
    # iteration, not per line-search probe — so this never stops on a rejected
    # point. Costs one forward solve per check. stop_below = 0 disables.
    # NB a callback halt returns retcode Failure; `stopped` is the real signal.
    stop_iter = Ref(0)
    function _cb(state, l)
        callback(state, l) && return true
        if stop_below > 0 && state.iter % stop_every == 0 &&
           data_loss(state.u) <= stop_below
            stop_iter[] = state.iter
            return true
        end
        return false
    end

    # Fail at θ0 rather than several thousand iterations into a NaN. Catches an
    # empty penalty grid (t_pen omitted while reg penalises), NaNs in Y_train,
    # and a warm-start θ that no longer solves.
    @assert isfinite(loss(θ0, nothing)) "loss at θ0 is not finite. heck t_pen vs reg, and data for NaNs"

    optf     = Optimization.OptimizationFunction(loss, Optimization.AutoZygote())
    prob_opt = Optimization.OptimizationProblem(optf, θ0)
    res      = Optimization.solve(prob_opt, opt; maxiters = maxiters, callback = _cb)
    θ★      = res.u

    predict_state_raw(t_grid) = predict(θ★, t_grid)
    predict_g_raw(t_grid)     = first(g_NN(predict_state_raw(t_grid), θ★, st_g))
    n_params                  = length(ComponentArrays.getdata(θ★))

    return (contract = (predict_state_raw = predict_state_raw,
                        predict_g_raw     = predict_g_raw,
                        n_params          = n_params),
            θ       = θ★,
            predict = predict,
            loss    = loss,
            retcode = res.retcode,
            stats   = res.stats,
            stopped   = stop_iter[] > 0,
            stop_iter = stop_iter[],
            data_loss = data_loss)
end
