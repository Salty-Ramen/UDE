#=-------------------------------------------------------------------------------
eval_and_recover.jl

Shared evaluation + symbolic recovery for the model-discovery benchmark.

This file knows ONLY the output contract and standard tooling (SymbolicRegression
+ stats). It contains no knowledge of any fitting method (AI-Aristotle / UDE /
SINDy): every input arrives either as the contract NamedTuple or as plain sample
arrays. There are no `isa`/method branches on any fitter type.

──────────────────────────────────────────────────────────────────────────────
STEP 1 — Output contract (a documented convention, not a type)

A fitter `fit_X(data, architecture; ...)` returns the NamedTuple:

    predict_state_raw(t_grid::1×B) -> raw states      (n_states × B)
    predict_g_raw(t_grid::1×B)     -> raw missing term (n_g × B); already raw
    n_params::Int                  -> network capacity (REPORTED ONLY, never
                                      fed into any information criterion)

Symbolic-recovery parameters (SR.jl path — the primary path):
    binary_operators = [+, -, *]   (covers PK's kg·G − kb·B and bilinear terms)
    unary_operators  = none
    selection        = AICc or BIC over SR's Pareto front
    IC parameter count = SR node-count `complexity` (DEFAULT)

ABLATION (revisit later — `complexity_measure` kwarg on `recover_symbolic`):
    The IC parameter count can be either SR's node-count `complexity` (:nodes,
    default) or the number of fitted numeric constants (:constants). On the
    noisy synthetic PK case, :constants let BIC select a spurious U-term model
    (a structural false positive) because nested tree structure buys complexity
    for few extra constants; :nodes penalises that bloat and recovers the true
    kg·G − kb·B. The two measures coincide for a flat polynomial library, so the
    future DataDrivenDiffEq/STLSQ path is unaffected. This node-vs-constant
    penalty choice is a flagged ablation to study against real cohort data.

(The polynomial-library + degree + STLSQ-threshold parameters belong to the
 future DataDrivenDiffEq path, NOT here.)

──────────────────────────────────────────────────────────────────────────────
Division of labour:
  `evaluate`         is deterministic and cheap: rel-L2 fit metrics + n_params.
  `recover_symbolic` is the (stochastic, slower) SR search yielding the symbolic
                     model and its AIC/AICc/BIC.
  The benchmark harness (Step 5) calls both and merges them. We deliberately do
  NOT run SR inside `evaluate`.
-------------------------------------------------------------------------------=#

using LinearAlgebra
using Random
using SymbolicRegression:
    Options, equation_search, calculate_pareto_frontier,
    compute_complexity, string_tree, eval_tree_array

# Reference rel-L2 — identical formula to No-API-pharmacokinetics-repro.jl so
# `evaluate` reproduces the numbers that file already prints.
relative_l2(yhat, y) = norm(vec(yhat .- y)) / max(norm(vec(y)), eps(Float32))

#=-------------------------------------------------------------------------------
Evaluation (deterministic; no SR)
-------------------------------------------------------------------------------=#

"""
    evaluate(contract, data) -> NamedTuple

Deterministic fit diagnostics against the output contract.

`data` must provide `t_dense` (1×N) and `Y_dense` (n_states×N). If it also
provides `f_true` (length-N vector, or n_g×N matrix), the missing-term rel-L2
is reported; otherwise that field is `nothing`. The i-th row of `f_true` is
compared against the i-th row of `predict_g_raw`.

Returns:
  rel_l2_state   :: Vector{Float32}                  one per state row
  rel_l2_missing :: Union{Nothing,Vector{Float32}}   one per g row, or nothing
  n_params       :: Int                              contract passthrough
"""
function evaluate(contract, data)
    Ŷ = contract.predict_state_raw(data.t_dense)                  # n_states × N
    rel_state = Float32[relative_l2(Ŷ[i, :], data.Y_dense[i, :])
                        for i in axes(Ŷ, 1)]

    rel_missing = nothing
    if hasproperty(data, :f_true) && data.f_true !== nothing
        ĝ = contract.predict_g_raw(data.t_dense)                  # n_g × N
        F = data.f_true isa AbstractVector ?
            reshape(data.f_true, 1, :) : data.f_true              # 1×N or n_g×N
        rel_missing = Float32[relative_l2(ĝ[i, :], F[i, :])
                              for i in axes(F, 1)]
    end

    return (rel_l2_state   = rel_state,
            rel_l2_missing = rel_missing,
            n_params       = contract.n_params)
end

#=-------------------------------------------------------------------------------
Information criteria
-------------------------------------------------------------------------------=#

"""
    info_criteria(rss, n, k) -> (aic, aicc, bic)

Gaussian least-squares information criteria for `n` samples, residual sum of
squares `rss`, and `k` free parameters. K = k + 1 counts the residual variance
σ² as a parameter (Burnham–Anderson). On a near-exact fit (`rss → 0`) the log
term diverges to −∞; `rss` is floored at `eps` to keep that finite-but-large
rather than `NaN`. AICc is `Inf` when the small-sample correction is undefined
(`n ≤ K + 1`).
"""
function info_criteria(rss::Real, n::Integer, k::Integer)
    rss = max(float(rss), eps(Float64))
    K   = k + 1
    aic  = n * log(rss / n) + 2K
    aicc = (n - K - 1) > 0 ? aic + 2K * (K + 1) / (n - K - 1) : Inf
    bic  = n * log(rss / n) + K * log(n)
    return (aic = aic, aicc = aicc, bic = bic)
end

# Number of fitted numeric constants in an SR expression tree = the IC `k`.
# DynamicExpressions trees are iterable; a constant leaf has `degree == 0` and
# `constant == true` (DynamicExpressions Node docs; cf. PySR discussion #449,
# which counts leaves with exactly `node.degree == 0 && node.constant`).
# If a future package version changes this, swap for:
#   length(first(SymbolicRegression.get_scalar_constants(tree)))
_count_constants(tree) = count(node -> node.degree == 0 && node.constant,
                               hasproperty(tree, :tree) ? tree.tree : tree)

#=-------------------------------------------------------------------------------
Symbolic recovery (SymbolicRegression.jl)
-------------------------------------------------------------------------------=#

"""
    recover_symbolic(state_samples, g_samples; kwargs...) -> Vector{NamedTuple}

Symbolic regression of each missing-term row in `g_samples` (n_g × B) against
the states in `state_samples` (n_states × B). SR's low-level interface expects
column-major `[features, rows]`, which is exactly this layout, so the arrays
pass through unchanged.

One NamedTuple is returned per g row, describing the model selected from SR's
Pareto front by the requested information criterion (`select`, default `:bic`):

  expr_string :: String     printed expression
  expression  :: <tree>     SR expression (use `.predict` to evaluate it)
  predict     :: X -> ŷ     closure: maps a (n_states × M) sample matrix to ŷ
  complexity  :: Int        SR node count
  n_constants :: Int        number of fitted numeric constants
  ic_k        :: Int        the count actually fed to the IC (= `complexity` for
                            `complexity_measure = :nodes`, else `n_constants`)
  mse         :: Float64    SR loss (default loss is MSE)
  rss         :: Float64    n * mse
  aic, aicc, bic :: Float64
  complexity_measure :: Symbol   which count drove the IC
  pareto      :: Vector     the full scored dominating front (for inspection)

`complexity_measure` (default `:nodes`) selects whether the IC penalty counts
SR node complexity or fitted constants; see the ABLATION note in the file header.

Reproducibility: each row runs `parallelism = :serial` after seeding the global
RNG with `seed + i`. (Add `deterministic = true` to `Options` if your SR version
supports it and you need bit-for-bit repeats; serial + seed is enough for the
*selected equation* to be stable on clean targets.)
"""
function recover_symbolic(state_samples::AbstractMatrix,
                          g_samples::AbstractMatrix;
                          binary_operators = [+, -, *],
                          unary_operators  = Function[],
                          niterations::Int = 120,
                          maxsize::Int     = 20,
                          seed::Int        = 0,
                          variable_names   = nothing,
                          select::Symbol   = :bic,
                          complexity_measure::Symbol = :nodes,
                          complexity_of_operators = nothing,
                          nested_constraints = nothing,
                          parallelism::Symbol = :multithreading)

    X = Float64.(state_samples)
    n = size(X, 2)

    opt_extra = merge(
        complexity_of_operators === nothing ? (;) : (; complexity_of_operators),
        nested_constraints      === nothing ? (;) : (; nested_constraints),
    )
    options = Options(; binary_operators = binary_operators,
                        unary_operators  = unary_operators,
                        maxsize          = maxsize,
                        opt_extra...)
    
    results = map(axes(g_samples, 1)) do i
        y = Float64.(vec(g_samples[i, :]))
        Random.seed!(seed + i)
        hof = variable_names === nothing ?
            equation_search(X, y; niterations = niterations,
                            options = options, parallelism = parallelism) :
            equation_search(X, y; niterations = niterations,
                            options = options, parallelism = parallelism,
                            variable_names = variable_names)

        front = calculate_pareto_frontier(hof)

        scored = map(front) do m
            treeᵢ       = m.tree
            complexity  = compute_complexity(m, options)
            n_constants = _count_constants(m.tree)
            ic_k = complexity_measure === :nodes ? complexity : n_constants
            mse = Float64(m.loss)          # NB: older SR versions use `m.score`
            rss = n * mse
            ic  = info_criteria(rss, n, ic_k)
            (member      = m,
             expr_string = string_tree(m.tree, options),
             expression  = m.tree,
             predict     = M -> first(eval_tree_array(treeᵢ, Float64.(M), options)),
             complexity  = complexity,
             n_constants = n_constants,
             ic_k = ic_k, mse = mse, rss = rss,
             aic = ic.aic, aicc = ic.aicc, bic = ic.bic)
        end

        # PySR-style score = −Δlog(loss)/Δcomplexity along the complexity-sorted
        # front; elbow member = argmax score. (PySR also restricts to members
        # whose loss is near the front minimum — omitted here.)
        ord    = sortperm([s.complexity for s in scored])
        scores = zeros(Float64, length(scored))
        for j in 2:length(ord)
            a, b = scored[ord[j-1]], scored[ord[j]]
            dc = b.complexity - a.complexity
            scores[ord[j]] = dc > 0 ?
                (log(max(a.mse, eps())) - log(max(b.mse, eps()))) / dc : 0.0
        end
        scored = [merge(s, (score = scores[k],)) for (k, s) in enumerate(scored)]

        best = select === :score ? argmax(s -> s.score, scored) :
                                    argmin(s -> getproperty(s, select), scored)

        (expr_string = best.expr_string,
         expression  = best.expression,
         predict     = best.predict,
         complexity  = best.complexity,
         n_constants = best.n_constants,
         ic_k = best.ic_k, mse = best.mse, rss = best.rss,
         aic = best.aic, aicc = best.aicc, bic = best.bic,
         score = best.score,
         complexity_measure = complexity_measure,
         pareto = scored)
    end

    return results
end
