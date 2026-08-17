# ============================================================================
# virus_ifn_data.jl — synthetic data generator for the Virus/IFN/M system.
#
# Integrates the PUBLISHED full model (Hill IFN induction + macrophage coupling)
# and emits the eval/UDE data contract. This is the ground-truth ORACLE: it is
# the ONLY place the true parameters live, so the "known" physics handed to the
# UDE (virus_ifn_ude_sr.jl) cannot silently drift from what generated the data.
#
# Output contract (consumed by fit_ude in UDE.jl and evaluate in EvalAndRecover.jl):
#   t_train :: 1×(T·m)   Y_train :: 3×(T·m)   NOISY training samples: T timepoints,
#                                             m mice/timepoint (replicates)
#   t_dense :: 1×N       Y_dense :: 3×N       CLEAN ground-truth states (eval target)
#   f_true  :: 3×N       true missing terms g1,g2,g3 on the CLEAN trajectory
#   t_span  :: (0f0, 8f0)
#
# Grey-box split the UDE assumes (see virus_ifn_ude_sr.jl):
#   dV   = k*V*(1-V/K) - V*g1 - d_v*V      g1 = r_v_ifn*IFN + r_v_M*M
#   dIFN = g2 - d_ifn*IFN                  g2 = k1*V^n/(k2+V^n) + r_ifn_M*M
#   dM   = g3 - d_m*M                      g3 = r_M_ifn*IFN
# so the "known" params are (k, K, d_v, d_ifn, d_m); everything else is in g.
# ============================================================================
using DrWatson; @quickactivate "UDE"


using OrdinaryDiffEq
using Random, Statistics          # noise draws + per-state σ scale

# All 12 true parameters + IC — single source of truth (Float64 for accuracy).
const TRUE_PARAMS = (
    k       = 7.68245046e-01, K     = 3.74536691e+01,
    r_v_ifn = 7.91020870e-03, d_v   = 1.25531214e-01,
    r_M_ifn = 1.07219620e+00, d_ifn = 1.89415901e+00,
    k1      = 1.35869271e+02, k2    = 2.23644344e+04,
    d_m     = 3.96629962e-01, n     = 5.71840052e+00,
    r_v_M   = 2.73522610e-03, r_ifn_M = 7.15053633e-02,
)
const Y0_TRUE64 = [1.73987502e+00, 0.0, 0.0]     # V0, IFN0, M0
const Y0        = Float32.(Y0_TRUE64)            # IC handed to fit_ude
const T_SPAN    = (0f0, 8f0)

# Full published RHS (in-place, Float64). `p` is TRUE_PARAMS (a NamedTuple).
function virus_ifn_rhs!(du, u, p, t)
    V, IFN, M = u
    du[1] = p.k*V*(1 - V/p.K) - p.r_v_ifn*IFN*V - p.r_v_M*V*M - p.d_v*V
    du[2] = (p.k1*V^p.n)/(p.k2 + V^p.n) + p.r_ifn_M*M - p.d_ifn*IFN
    du[3] = p.r_M_ifn*IFN - p.d_m*M
    return nothing
end

# True missing terms (the grey-box split) evaluated on a 3×N state matrix.
function true_g(X, p = TRUE_PARAMS)
    V, IFN, M = X[1, :], X[2, :], X[3, :]
    g1 = p.r_v_ifn .* IFN .+ p.r_v_M .* M
    g2 = (p.k1 .* V .^ p.n) ./ (p.k2 .+ V .^ p.n) .+ p.r_ifn_M .* M
    g3 = p.r_M_ifn .* IFN
    permutedims(hcat(g1, g2, g3))          # 3 × N
end

# Integrate the true (oracle) model and sample the CLEAN states at `saveat`.
# Float64, tight tol; returns 3 × length(saveat).
function solve_true(saveat)
    prob = ODEProblem(virus_ifn_rhs!, Y0_TRUE64, (0.0, 8.0), TRUE_PARAMS)
    sol  = solve(prob, AutoTsit5(Rosenbrock23()); saveat = saveat,
                 reltol = 1e-8, abstol = 1e-10)
    reduce(hcat, sol.u)                    # 3 × length(saveat)
end

"""
    generate_data(; dt=0.1, timepoints=nothing, mice_per_timepoint=1,
                    noise_frac=0.0, seed=0) -> NamedTuple

Return the Float32 data contract.

Training set (what fit_ude sees): `timepoints` distinct sample times uniformly
spaced on [0,8] (or the dense grid when `timepoints === nothing`), each replicated
`mice_per_timepoint` times with an independent σ-scaled Gaussian draw when
`noise_frac > 0`. Total mice = timepoints · mice_per_timepoint. fit_ude's
`col_of_obs` already sums the loss over these replicates.

Diagnostics set (what `evaluate` scores against): a CLEAN dense grid
(`0:dt:8`) — `Y_dense`/`f_true` are the noise-free ground truth, so state and
missing-term rel-L2 measure error against truth rather than against the noisy
samples. Per-state noise σ is the clean-trajectory std.

Defaults (`timepoints=nothing, mice_per_timepoint=1, noise_frac=0`) reproduce the
original clean full-grid contract, so `t_train == t_dense` and `Y_train == Y_dense`.
"""
function generate_data(; dt = 0.1, timepoints = nothing, mice_per_timepoint::Int = 1,
                         noise_frac = 0.0, seed = 0)
    # ── Clean diagnostics grid (ground truth) ──────────────────────────────
    td = collect(0.0:dt:8.0)
    Xd = solve_true(td)                    # 3 × Nd  clean states
    Fd = true_g(Xd)                        # 3 × Nd  clean missing terms

    # f_true must reconstruct the full RHS through the grey-box split. Checked
    # in Float64 on the clean states, BEFORE the Float32 store: here it holds to
    # machine eps. (Checking after storage fails near derivative turning points,
    # where the net du crosses ~0 and a benign Float32-rounding perturbation of
    # f_true dwarfs it — that is not an algebra error.)
    let du = zeros(3), p = TRUE_PARAMS
        for j in axes(Xd, 2)
            u = Xd[:, j]; g = Fd[:, j]; V, IFN, M = u
            virus_ifn_rhs!(du, u, p, 0.0)
            grey = [p.k*V*(1 - V/p.K) - V*g[1] - p.d_v*V,
                    g[2] - p.d_ifn*IFN,
                    g[3] - p.d_m*M]
            @assert isapprox(du, grey; rtol = 1e-8, atol = 1e-9) "grey split != full RHS at col $j"
        end
    end

    # ── Training grid: T timepoints × m mice, one noisy draw per mouse ──────
    tu = timepoints === nothing ? td : collect(range(0.0, 8.0, length = timepoints))
    Xu = solve_true(tu)                                    # 3 × T  clean
    t_tr = repeat(tu; inner = mice_per_timepoint)          # length T·m
    Ytr  = repeat(Xu; inner = (1, mice_per_timepoint))     # 3 × T·m clean
    if noise_frac > 0
        s   = vec(std(Xd; dims = 2))                       # per-state scale
        rng = MersenneTwister(seed)
        Ytr = Ytr .+ noise_frac .* s .* randn(rng, size(Ytr))
    end

    return (t_train = permutedims(Float32.(t_tr)),   # 1 × (T·m)
            Y_train = Float32.(Ytr),                 # 3 × (T·m)  (noisy)
            t_dense = permutedims(Float32.(td)),     # 1 × Nd
            Y_dense = Float32.(Xd),                  # 3 × Nd     (CLEAN)
            f_true  = Float32.(Fd),                  # 3 × Nd     (CLEAN)
            t_span  = T_SPAN)
end

const data = generate_data()

# ── Preflight (storage-level): shapes, IC, finiteness, grid endpoints.
#    The grey-box decomposition is checked at full Float64 precision inside
#    generate_data, above.
let d = data
    N = size(d.Y_dense, 2)
    @assert size(d.Y_dense) == (3, N) && size(d.f_true) == (3, N)
    @assert size(d.t_dense) == (1, N)
    @assert all(isfinite, d.Y_dense) && all(isfinite, d.f_true)
    @assert isapprox(d.Y_dense[:, 1], Y0; atol = 1f-4) "IC not reproduced at t=0"
    @assert d.t_dense[1] == 0f0 && d.t_dense[end] == 8f0
    @info "data preflight passed" N V=extrema(d.Y_dense[1, :]) IFN=extrema(d.Y_dense[2, :]) M=extrema(d.Y_dense[3, :])
end

