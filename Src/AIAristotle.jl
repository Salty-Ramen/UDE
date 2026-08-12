#=-------------------------------------------------------------------------------
AI-Aristotle.jl

The AI-Aristotle missing-part method, generalized out of
No-API-pharmacokinetics-repro.jl. The fitter internals are copied verbatim;
the only change is that the three PK globals are now inputs:

    pk_grey_rhs      ->  architecture   (a function (y_raw, g, ode_params) -> dy/dt_raw)
    PK_TRUE_PARAMS   ->  ode_params     (whatever object `architecture` expects)
    PK_U0            ->  y0             (raw initial state, length n_states)

Nothing here knows about pharmacokinetics. PK becomes "the experiment that
passes `pk_grey_rhs` as the architecture", which is the intended use of the
shared RHS convention.

Fixed architecture choices (unchanged from the reference):
  - Hardwired IC:   ŷ(t) = y₀ + t · MLP(t, θ)
  - ForwardDiff AD: per-column derivative through a StatefulLuxLayer
  - Z-score state:  z = (y - μ) ./ σ
  - Fixed ODE params: captured in the RHS closure, not optimized
  - Fixed loss weights: W_DATA, W_ODE, W_L1G, W_L1S constants
  - Plain MSE residuals in z-space

Three training stages (compose them in the experiment script, as the reference
run block does — there is no train()/fit() wrapper):
  1. Supervised (Adam): fit state MLP to observations (data loss only).
  2. Joint (Adam):      data + ODE residual + g-net L1; both MLPs train.
  3. g-only (Adam):     freeze state MLP, fine-tune g-net against ODE residual.

Output contract (consumed by eval_and_recover.jl):
  `as_contract(setup, ps)` -> (predict_state_raw, predict_g_raw, n_params).

ABLATION (flagged for later — fixed vs prior vs fitted ODE params):
  Today `ode_params` is fixed: it enters the RHS via `setup.ode_params` and is
  never trained. The planned study compares (a) fixed params, (b) a prior on
  params, (c) jointly fitting params. Each is a different *loss*, written in the
  experiment per the "loss is user space" rule: fitting would move params into
  the trainable `ps` and add a term to the stage-2/3 loss. The seam is that
  `ode_params` flows in through one argument, so none of the fitter internals
  change — only what the experiment hands to `build_setup` / its loss.

The StatefulLuxLayer wrapper on the state MLP is the hook Lux's nested-AD rule
uses so ForwardDiff inside Zygote does not silently drop the inner gradient.
See: https://lux.csail.mit.edu/stable/manual/nested_autodiff
-------------------------------------------------------------------------------=#

using Lux
using ComponentArrays
using ForwardDiff
using Zygote
using Optimization, OptimizationOptimisers
using Statistics
using Random
using Printf
using JLD2

#=-------------------------------------------------------------------------------
§1. Z-score helpers

  scale(y, μ, σ)    :  y_raw → z
  unscale(z, μ, σ)  :  z → y_raw
  scale_rhs(f, σ)   :  dy/dt → dz/dt = (dy/dt) / σ   (μ cancels)
-------------------------------------------------------------------------------=#

scale(y, μ, σ)      = (y .- μ) ./ σ
unscale(z, μ, σ)    = σ .* z .+ μ
scale_rhs(f_raw, σ) = f_raw ./ σ

#=-------------------------------------------------------------------------------
§2. Predictors (method-internal; not shared across methods)
-------------------------------------------------------------------------------=#

function make_predictors(state_mlp, st_state, g_mlp, st_g, y0_z)
    smlp = Lux.StatefulLuxLayer{true}(state_mlp, nothing, st_state)
    sg   = Lux.StatefulLuxLayer{true}(g_mlp,     nothing, st_g)

    predict_state(ps, t_grid) = y0_z .+ t_grid .* smlp(t_grid, ps.StateMLP)

    function predict_deriv(ps, t_grid)
        B = size(t_grid, 2)
        cols = map(1:B) do j
            tj = t_grid[1, j]
            ForwardDiff.derivative(
                t -> vec(predict_state(ps, reshape([t], 1, 1))),
                tj,
            )
        end
        return reduce(hcat, cols)
    end

    predict_g(ps, t_grid) = sg(t_grid, ps.gMLP)

    return predict_state, predict_deriv, predict_g
end

#=-------------------------------------------------------------------------------
§3. Losses

Plain MSE in z-space for both data and ODE residuals. The z-transform of the
state already puts the residual on an O(1) scale, so there is no secondary
scaling.
-------------------------------------------------------------------------------=#

function loss_data(ps, predict_state, t_train, Z_train)
    ẑ = predict_state(ps, t_train)
    return mean(abs2, ẑ .- Z_train)
end

# `architecture(y_raw, g, ode_params)` is the user's grey-box RHS in raw coords.
function loss_ode(ps, predict_state, predict_deriv, predict_g,
                  t_dense, μ_state, σ_state, architecture, ode_params)
    ẑ     = predict_state(ps, t_dense)
    dẑdt  = predict_deriv(ps, t_dense)
    g_arr = predict_g(ps, t_dense)

    y_raw = unscale(ẑ, μ_state, σ_state)
    f_raw = architecture(y_raw, g_arr, ode_params)
    f_z   = scale_rhs(f_raw, σ_state)

    return mean(abs2, dẑdt .- f_z)
end

l1_g(ps)     = (w = ComponentArrays.getdata(ps.gMLP);     sum(abs, w) / length(w))
l1_state(ps) = (w = ComponentArrays.getdata(ps.StateMLP); sum(abs, w) / length(w))

# Fixed loss weights (the 1/ϵ² weights of the library, written directly).
const W_DATA = 1f0
const W_ODE  = 1f0
const W_L1G  = 0f0
const W_L1S  = 1f0

#=-------------------------------------------------------------------------------
§4. Setup

Builds the networks, runs Lux.setup, computes z-score parameters, makes
predictors, and stows the architecture + fixed ode_params for the stage losses.
Call once; reuse across stages.
-------------------------------------------------------------------------------=#

"""
    build_setup(data, architecture, ode_params, y0;
                seed, build_state_mlp, build_g_mlp) -> NamedTuple

`data` provides `t_train` (1×Ntrain), `Y_train` (n_states×Ntrain), and
`t_dense` (1×Ndense). `y0` is the raw initial state (length n_states).
`build_state_mlp` / `build_g_mlp` are zero-arg constructors for the two
networks (their input/output dims must match the problem).
"""
function build_setup(data, architecture, ode_params, y0;
                     seed::Int = 2101,
                     build_state_mlp::Function,
                     build_g_mlp::Function)
    rng = MersenneTwister(seed)
    state_mlp = build_state_mlp()
    g_mlp     = build_g_mlp()
    ps_state, st_state = Lux.setup(rng, state_mlp)
    ps_g,     st_g     = Lux.setup(rng, g_mlp)

    μ = Float32.(vec(mean(data.Y_train; dims = 2)))
    σ = Float32.(max.(vec(std(data.Y_train; dims = 2)), 1f-6))
    Z_train = scale(data.Y_train, μ, σ)
    y0_z    = scale(y0, μ, σ)

    predict_state, predict_deriv, predict_g =
        make_predictors(state_mlp, st_state, g_mlp, st_g, y0_z)

    ps0 = ComponentArray(StateMLP = ps_state, gMLP = ps_g)

    return (ps0           = ps0,
            μ             = μ,
            σ             = σ,
            Z_train       = Z_train,
            predict_state = predict_state,
            predict_deriv = predict_deriv,
            predict_g     = predict_g,
            architecture  = architecture,
            ode_params    = ode_params,
            state_mlp     = state_mlp,
            g_mlp         = g_mlp)
end

#=-------------------------------------------------------------------------------
§5. Loss-history container + minimal callback
-------------------------------------------------------------------------------=#

mutable struct LossHistory
    iter     :: Vector{Int}
    stage    :: Vector{Int}
    data_mse :: Vector{Float32}
    ode_mse  :: Vector{Float32}
end
LossHistory() = LossHistory(Int[], Int[], Float32[], Float32[])

# data_eval(ps) / ode_eval(ps) return diagnostic MSEs; pass `nothing` to skip.
function make_callback(history::LossHistory, stage::Int, iter_offset::Int,
                       log_every::Int, data_eval, ode_eval)
    function cb(state, l)
        state.iter % log_every != 0 && return false
        ps = state.u
        d = data_eval === nothing ? NaN32 : Float32(data_eval(ps))
        o = ode_eval  === nothing ? NaN32 : Float32(ode_eval(ps))
        gi = iter_offset + state.iter
        push!(history.iter,     gi)
        push!(history.stage,    stage)
        push!(history.data_mse, d)
        push!(history.ode_mse,  o)
        @printf("[stage %d] iter=%6d  loss=%.4e  data_mse=%.4e  ode_mse=%.4e\n",
                stage, gi, l, d, o)
        return false
    end
    return cb
end

#=-------------------------------------------------------------------------------
§6. Per-stage solvers

Each returns (ps, history). `iter_offset` keeps the history's global-iteration
axis contiguous across stages.
-------------------------------------------------------------------------------=#

"""
    solve_stage1(setup, data, history; maxiters, lr, log_every)

Supervised data fit. The gMLP block rides in `ps0` and is updated by the
optimizer, but the loss is independent of it so its gradient is zero.
"""
function solve_stage1(setup, data, history::LossHistory;
                      maxiters::Int = 50_000,
                      lr::Float32   = 1f-3,
                      log_every::Int = 1000)
    println("########## Stage 1: supervised ##########")
    loss(ps, _) = W_DATA * loss_data(ps, setup.predict_state, data.t_train, setup.Z_train)
    data_eval(ps) = loss_data(ps, setup.predict_state, data.t_train, setup.Z_train)

    cb   = make_callback(history, 1, 0, log_every, data_eval, nothing)
    optf = Optimization.OptimizationFunction(loss, Optimization.AutoZygote())
    prob = Optimization.OptimizationProblem(optf, setup.ps0, nothing)
    res  = Optimization.solve(prob, OptimizationOptimisers.Adam(lr);
                              maxiters = maxiters, callback = cb)
    return (ps = res.u, history = history)
end

"""
    solve_stage2(ps_init, setup, data, history; maxiters, lr, log_every, iter_offset)

Joint data + ODE residual + g-L1. The ODE residual uses `setup.architecture`
and `setup.ode_params`.
"""
function solve_stage2(ps_init, setup, data, history::LossHistory;
                      maxiters::Int    = 120_000,
                      lr::Float32      = 1f-3,
                      log_every::Int   = 1000,
                      iter_offset::Int = 50_000)
    println("########## Stage 2: joint ##########")
    function loss(ps, _)
        W_DATA * loss_data(ps, setup.predict_state, data.t_train, setup.Z_train) +
        W_ODE  * loss_ode(ps, setup.predict_state, setup.predict_deriv, setup.predict_g,
                          data.t_dense, setup.μ, setup.σ, setup.architecture, setup.ode_params) +
        W_L1G  * l1_g(ps)
    end
    data_eval(ps) = loss_data(ps, setup.predict_state, data.t_train, setup.Z_train)
    ode_eval(ps)  = loss_ode(ps, setup.predict_state, setup.predict_deriv, setup.predict_g,
                             data.t_dense, setup.μ, setup.σ, setup.architecture, setup.ode_params)

    cb   = make_callback(history, 2, iter_offset, log_every, data_eval, ode_eval)
    optf = Optimization.OptimizationFunction(loss, Optimization.AutoZygote())
    prob = Optimization.OptimizationProblem(optf, ps_init, nothing)
    res  = Optimization.solve(prob, OptimizationOptimisers.Adam(lr);
                              maxiters = maxiters, callback = cb)
    return (ps = res.u, history = history)
end

"""
    solve_stage3(ps_init, setup, data, history; maxiters, lr, log_every, iter_offset)

g-only fine-tune. The StateMLP is frozen, so the StateMLP-dependent quantities
(ẑ, dẑ/dt, y_raw) are precomputed once and each iteration is a gMLP
forward+backward against the ODE residual. (Uses Adam here; swap in
`OptimizationOptimJL.LBFGS(m = …)` — and add that import — for a quasi-Newton
fine-tune if desired.)
"""
function solve_stage3(ps_init, setup, data, history::LossHistory;
                      maxiters::Int    = 10_000,
                      lr::Float32      = 1f-4,
                      log_every::Int   = 1000,
                      iter_offset::Int = 170_000)
    println("########## Stage 3: g-only fine-tune ##########")
    # Build ps_full from concrete-typed flat data so Zygote's pullback through
    # ComponentArrays.getproperty does not produce a Vector{Any}.
    frozen_state = copy(ComponentArrays.getdata(ps_init.StateMLP))
    ax_full      = ComponentArrays.getaxes(ps_init)

    function build_full(ps_g)
        flat = vcat(frozen_state, ComponentArrays.getdata(ps_g.gMLP))
        return ComponentArray(flat, ax_full)
    end

    # Precompute frozen-StateMLP quantities once.
    ps_frozen_full = build_full(ComponentArray(gMLP = ps_init.gMLP))
    ẑ_const       = setup.predict_state(ps_frozen_full, data.t_dense)
    dẑdt_const    = setup.predict_deriv(ps_frozen_full, data.t_dense)
    y_raw_const    = unscale(ẑ_const, setup.μ, setup.σ)

    function loss(ps_g, _)
        ps_full = build_full(ps_g)
        g_arr   = setup.predict_g(ps_full, data.t_dense)
        f_raw   = setup.architecture(y_raw_const, g_arr, setup.ode_params)
        f_z     = scale_rhs(f_raw, setup.σ)
        W_ODE * mean(abs2, dẑdt_const .- f_z) + W_L1G * l1_g(ps_full)
    end
    function ode_eval(ps_g)
        ps_full = build_full(ps_g)
        g_arr   = setup.predict_g(ps_full, data.t_dense)
        f_raw   = setup.architecture(y_raw_const, g_arr, setup.ode_params)
        f_z     = scale_rhs(f_raw, setup.σ)
        return mean(abs2, dẑdt_const .- f_z)
    end

    cb    = make_callback(history, 3, iter_offset, log_every, nothing, ode_eval)
    ps0_g = ComponentArray(gMLP = ps_init.gMLP)
    optf  = Optimization.OptimizationFunction(loss, Optimization.AutoZygote())
    prob  = Optimization.OptimizationProblem(optf, ps0_g, nothing)
    res   = Optimization.solve(prob, OptimizationOptimisers.Adam(lr);
                               maxiters = maxiters, callback = cb)
    ps_final = build_full(res.u)
    return (ps = ps_final, history = history)
end

#=-------------------------------------------------------------------------------
§7. Output-contract bridge

Expose a trained run through the convention eval_and_recover.jl consumes.
predict_g is already raw; predict_state is unscaled back to raw coordinates.
-------------------------------------------------------------------------------=#

"""
    as_contract(setup, ps) -> (predict_state_raw, predict_g_raw, n_params)

predict_state_raw(t_grid) = unscale(state network) ; predict_g_raw = g network
(already raw); n_params = total weights in the two networks.
"""
function as_contract(setup, ps)
    predict_state_raw(t_grid) = unscale(setup.predict_state(ps, t_grid), setup.μ, setup.σ)
    predict_g_raw(t_grid)     = setup.predict_g(ps, t_grid)
    n_params = length(ComponentArrays.getdata(ps.StateMLP)) +
               length(ComponentArrays.getdata(ps.gMLP))
    return (predict_state_raw = predict_state_raw,
            predict_g_raw     = predict_g_raw,
            n_params          = n_params)
end

#=-------------------------------------------------------------------------------
§8. Checkpointing (generic; not method-specific)

Persists the flat parameter buffer + axes (robust across versions) and the
LossHistory columns. `setup` is rebuilt on reload via `build_setup(...)` with
the same seed, then the saved `ps` is attached.
-------------------------------------------------------------------------------=#

"""
    save_stage(path, stage, ps, history; data=nothing, meta=nothing) -> path
"""
function save_stage(path::AbstractString, stage::Int, ps, history::LossHistory;
                    data = nothing, meta = nothing)
    mkpath(dirname(abspath(path)))
    ps_flat = collect(ComponentArrays.getdata(ps))
    ps_axes = ComponentArrays.getaxes(ps)
    JLD2.jldopen(path, "w") do io
        io["stage"]            = stage
        io["ps_flat"]          = ps_flat
        io["ps_axes"]          = ps_axes
        io["history_iter"]     = history.iter
        io["history_stage"]    = history.stage
        io["history_data_mse"] = history.data_mse
        io["history_ode_mse"]  = history.ode_mse
        data === nothing || (io["data"] = data)
        meta === nothing || (io["meta"] = meta)
    end
    @printf("[checkpoint] stage %d written to %s (%.1f KiB)\n",
            stage, path, stat(path).size / 1024)
    return path
end

"""
    load_stage(path) -> NamedTuple (stage, ps, history[, data, meta])
"""
function load_stage(path::AbstractString)
    JLD2.jldopen(path, "r") do io
        ps      = ComponentArray(io["ps_flat"], io["ps_axes"])
        history = LossHistory(io["history_iter"],
                              io["history_stage"],
                              io["history_data_mse"],
                              io["history_ode_mse"])
        nt = (stage = io["stage"], ps = ps, history = history)
        haskey(io, "data") && (nt = merge(nt, (data = io["data"],)))
        haskey(io, "meta") && (nt = merge(nt, (meta = io["meta"],)))
        return nt
    end
end
