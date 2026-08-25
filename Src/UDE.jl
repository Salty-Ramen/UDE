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
  - Positivity is NOT enforced for PK (G is structurally non-negative; small-init
    g keeps B sane). If a future system needs it, prefer log-state integration
    or `isoutofdomain` over an in-RHS `max`/`clamp` — both keep the comparison
    out of the differentiated RHS, so `ReverseDiffVJP(true)` stays correct.
  - `sensealg` defaults to QuadratureAdjoint(ReverseDiffVJP(true)): fast compiled
    tape, valid because the RHS is branch-free (pure arithmetic + tanh). If you
    add value- or time-dependent branching to the RHS (clamps, ReLU, dosing
    switches), drop the `true` or switch to ZygoteVJP/EnzymeVJP.
  - Single optimizer per call; pass `θ_init` to warm-start (e.g. Adam → LBFGS),
    composing in the experiment rather than baking a multi-phase wrapper.
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

"""
    fit_ude(data, architecture, ode_params, y0, g_builder; kwargs...) -> NamedTuple

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
                 λ::Real          = 0,
                 θ_init           = nothing)

    rng = MersenneTwister(seed)
    g_NN = g_builder()
    ps_g, st_g = Lux.setup(rng, g_NN)
    θ0 = θ_init === nothing ? ComponentArray(ps_g) : θ_init

    # Per-state scale for the data loss (so a large-magnitude state can't dominate).
    σ_state = Float32.(vec(max.(std(data.Y_train; dims = 2), 1f-6)))

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
    col_of_obs   = [searchsortedfirst(t_unique, t) for t in t_obs]   # length Nobs
    t_unique_row = permutedims(t_unique)

    predict(θ, t_grid) = Array(solve(remake(prob; p = θ), solver;
                                     saveat = vec(t_grid),
                                     abstol = abstol, reltol = reltol,
                                     sensealg = sensealg))

    function loss(θ, _)
        sol = predict(θ, t_unique_row)          # n_states × n_unique
        Ŷ   = sol[:, col_of_obs]                # n_states × Nobs  (scattered back)
        Statistics.mean(abs2, (Ŷ .- data.Y_train) ./ σ_state)
        λ == 0 ? data_term : data_term + λ * sum(abs2, θ)
    end
    

    optf     = Optimization.OptimizationFunction(loss, Optimization.AutoZygote())
    prob_opt = Optimization.OptimizationProblem(optf, θ0)
    res      = Optimization.solve(prob_opt, opt; maxiters = maxiters, callback = callback)
    θ★      = res.u

    predict_state_raw(t_grid) = predict(θ★, t_grid)
    predict_g_raw(t_grid)     = first(g_NN(predict_state_raw(t_grid), θ★, st_g))
    n_params                  = length(ComponentArrays.getdata(θ★))

    return (contract = (predict_state_raw = predict_state_raw,
                        predict_g_raw     = predict_g_raw,
                        n_params          = n_params),
            θ       = θ★,
            predict = predict,
            loss    = loss)
end
