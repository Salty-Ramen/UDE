# ============================================================================
# sr-sweep.jl — symbolic regression over cached UDE fits.
# experiments/UDE-on-noisy-synthetic-data/Ablating-on-regularization/
#
# SR targets are g_sr (the NETWORK's learned missing term), features are X_sr
# (the network's predicted states). No ground truth enters this file, so
# synthetic-data-gen.jl is deliberately NOT included: it would run
# generate_data() and its preflight in every shard process.
#
# One payload per (cell, term, sr_seed). No success flag, no classification:
# loss thresholds and complexity cutoffs calibrated on exactly-representable
# targets do not transfer to a network's g. The reporting pass makes that call
# from the persisted front.
#
# NOT reproducible per cell: parallelism = :multithreading. The deliverable is a
# recovery frequency over K_SEEDS, reproducible in distribution only. Flip
# PARALLELISM to :serial for bit-reproducible single runs.
#
# Run — interactive (the PROGRAM_FILE guard means include() does not launch):
#   include(".../sr-sweep.jl")        # runs preflight, prints the null ladder
#   run_sr_sweep(CONFIGS[1:1], SR_DIR)   # smoke: one real run
#   sr_status(CONFIGS, SR_DIR)
#
# Run — GNU parallel, 4 shards. Divide threads: :multithreading spawns
# Threads.nthreads() workers PER SHARD.
#   seq 4 | parallel -j4 julia --project -t 4 \
#     experiments/UDE-on-noisy-synthetic-data/Ablating-on-regularization/sr-sweep.jl {} 4
# ============================================================================

using DrWatson
@quickactivate "UDE"
using JLD2, DataFrames, Statistics, LinearAlgebra, Printf
using SymbolicRegression: Options
include(projectdir("Src", "EvalAndRecover.jl"))   # recover_symbolic

const SWEEP_DIR = projectdir("experiments", "UDE-on-noisy-synthetic-data",
                             "Ablating-on-regularization", "Results",
                             "sweep-errmodel-pinned-data")
const SR_DIR    = projectdir("experiments", "UDE-on-noisy-synthetic-data",
                             "Ablating-on-regularization", "Results",
                             "sr-sweep")          # ← SET THIS

# ── Cell selection ──────────────────────────────────────────────────────────
# Score = WORST of the three g channels (gating: a fit is good only if all three
# are). Gate = that score's 0.25 quantile over the filtered frame, i.e. the
# notebook's Q1_CUT. Within Q1, N_CELLS rows evenly spaced BY RANK, so the
# sample spans the quartile instead of collapsing onto its best few.
const SCORE_COLS    = [:rel_l2_gV, :rel_l2_gIFN, :rel_l2_gM]
const CELL_QUANTILE = 0.25
const N_CELLS       = 12
const K_SEEDS       = 8          # SR seeds per (cell, term); raise freely

# Cell identity: these keys are the payload provenance AND the savename.
const CELL_KEYS = ["timepoints", "mice_per_timepoint", "lambda_w", "lambda_dtt",
                   "stop_kappa", "init_seed", "seed", "err_model", "noise_frac"]

# ── SR configuration. ONE operator set for every channel and cell. ──────────
hill(x, n) = (xa = max(x, zero(x)) + oftype(x, 1e-6); inv(one(xa) + xa^(-n)))

const BINOPS      = [+, -, *, /, hill]
const NITERATIONS = 300
const MAXSIZE     = 20                # fronts saturate well below this
const PARALLELISM = :serial
const SR_OPTS     = (complexity_of_operators = [hill => 3],
                     nested_constraints      = [hill => [hill => 0]],
                     constraints             = [hill => (-1, 1)],
                     save_to_file = false, verbosity = 0, progress = false)
# populations / population_size are library defaults BY DESIGN (a 3x islands arm
# underperformed: islands inherit the trapped global best-so-far). Constructed
# here only to read the resolved values into the payload — mirror of the call
# inside recover_symbolic, so keep the two in step.
const SR_OPTIONS = Options(; binary_operators = BINOPS,
                             unary_operators  = Function[],
                             maxsize          = MAXSIZE, SR_OPTS...)

# term -> (X_sr rows, variable names). Hands SR the grey-box support.
const TERM_FEATURES = [([2, 3], ["IFN", "M"]),
                       ([1, 3], ["V", "M"]),
                       ([2],    ["IFN"])]

const NULL_DEGREES = [1, 2, 3, 5]
const NULL_ZERO    = 1e-10

# ── Load + filter ───────────────────────────────────────────────────────────
_yes(v) = coalesce.(v, false)

const DF = let d = collect_results(SWEEP_DIR)
    for c in [:error, :err_model, :noise_frac, :X_sr, :g_sr, :tg,
              SCORE_COLS..., Symbol.(CELL_KEYS)...]
        @assert hasproperty(d, c) "missing column $c in $SWEEP_DIR"
    end
    s = d[_yes(d.error .== "") .& _yes(d.err_model .== "prop") .&
          _yes(d.noise_frac .== 0.2), :]
    disallowmissing!(s, [SCORE_COLS..., Symbol.(CELL_KEYS)...])
    s
end
@assert nrow(DF) > 0 "no prop / noise 0.2 / error-free cells in $SWEEP_DIR"

const CELLS = let
    score = [maximum(Float64(r[c]) for c in SCORE_COLS) for r in eachrow(DF)]
    ord   = sortperm(score)
    q1    = ord[score[ord] .<= quantile(score, CELL_QUANTILE)]
    @assert length(q1) >= N_CELLS "Q1 has $(length(q1)) rows, need $N_CELLS"
    pick  = q1[round.(Int, range(1, length(q1); length = N_CELLS))]
    @printf("frame %d rows, Q1 %d rows (cut %.4g), picked %d spanning %.4g–%.4g\n",
            nrow(DF), length(q1), quantile(score, CELL_QUANTILE), N_CELLS,
            score[first(pick)], score[last(pick)])
    DF[pick, :]
end

cellkey(c)  = Tuple(c[k] for k in CELL_KEYS)
const ROWS  = Dict(Tuple(r[Symbol(k)] for k in CELL_KEYS) => r for r in eachrow(CELLS))
@assert length(ROWS) == N_CELLS "CELL_KEYS do not uniquely identify a cell"

# sr_seed INNERMOST: with nshards == K_SEEDS each shard gets one seed from every
# (cell, term), and cost tracks (cell, term) — engaging hill triggers expensive
# constant optimisation — so stride sharding balances.
const CONFIGS = [Dict{String,Any}(Dict(k => cell[Symbol(k)] for k in CELL_KEYS)...,
                                  "term" => term, "sr_seed" => s)
                 for cell in eachrow(CELLS) for term in 1:3 for s in 1:K_SEEDS]

cellname(c)         = savename(c, "jld2")
cellpath(dir, c)    = joinpath(dir, cellname(c))

# ── Null ladder: best total-degree-<=d polynomial in the same features, closed
#    form. Relative to var(target) so it is comparable to sel_rel_mse.
function poly_rel_mse(F, y, d)
    p    = size(F, 1)
    exps = [e for e in Iterators.product(ntuple(_ -> 0:d, p)...) if sum(e) <= d]
    A    = reduce(hcat, [vec(prod(F .^ collect(e); dims = 1)) for e in exps])
    Statistics.mean(abs2, A * (A \ y) .- y) / Statistics.var(y; corrected = false)
end

# ── One run ─────────────────────────────────────────────────────────────────
# wall_s is partly an OUTCOME, not just cost: engaging hill triggers constant
# optimisation, and successful runs have measured ~2.3x their siblings.
function run_sr_cell(c)
    row           = ROWS[cellkey(c)]
    term          = c["term"]
    frows, vnames = TERM_FEATURES[term]
    F = Float64.(row.X_sr[frows, :])
    y = Float64.(row.g_sr[term, :])

    t = @elapsed res = recover_symbolic(F, permutedims(y);
            binary_operators = BINOPS, unary_operators = Function[],
            niterations = NITERATIONS, maxsize = MAXSIZE,
            seed = c["sr_seed"], variable_names = vnames,
            select = :score, parallelism = PARALLELISM, SR_OPTS...)
    r  = res[1]
    # in run_sr_cell
    vy = Statistics.var(y; corrected = false)

    front = [(s.complexity, s.mse, s.expr_string) for s in r.pareto]
    hc    = [cx for (cx, _, e) in front if occursin("hill", e)]

    Dict{String,Any}(
        c...,
        "cell_score"      => maximum(Float64(row[s]) for s in SCORE_COLS),
        "niterations"     => NITERATIONS, "maxsize" => MAXSIZE,
        "parallelism"     => String(PARALLELISM),
        "populations"     => SR_OPTIONS.populations,
        "population_size" => SR_OPTIONS.population_size,
        "pareto"          => front,          # (complexity, mse, expr) — ALL members
        "sel_expr"        => r.expr_string,
        "sel_complexity"  => r.complexity,
        "sel_mse"         => r.mse,
        "sel_rel_mse"     => r.mse / vy,
        "target_var"      => vy,
        "null_degrees"    => NULL_DEGREES,
        "null_rel_mse"    => [poly_rel_mse(F, y, d) for d in NULL_DEGREES],
        # DESCRIPTORS, not success tests: hill has appeared at complexity 19
        # with relative loss 2e-4, i.e. decoration inside a polynomial.
        "hill_on_front"       => !isempty(hc),
        "hill_min_complexity" => isempty(hc) ? -1 : minimum(hc),
        "wall_s"          => t,
        "error"           => "",
    )
end

function error_payload(c, err)
    Dict{String,Any}(
        c...,
        "cell_score"      => maximum(Float64(ROWS[cellkey(c)][s]) for s in SCORE_COLS),
        "niterations"     => NITERATIONS, "maxsize" => MAXSIZE,
        "parallelism"     => String(PARALLELISM),
        "populations"     => SR_OPTIONS.populations,
        "population_size" => SR_OPTIONS.population_size,
        "pareto"          => Tuple{Int,Float64,String}[],
        "sel_expr"        => "", "sel_complexity" => -1,
        "sel_mse"         => NaN, "sel_rel_mse" => NaN, "target_var" => NaN,
        "null_degrees"    => NULL_DEGREES,
        "null_rel_mse"    => fill(NaN, length(NULL_DEGREES)),
        "hill_on_front"   => false, "hill_min_complexity" => -1,
        "wall_s"          => NaN,
        "error"           => sprint(showerror, err),
    )
end

# ── Preflight: dies before any SR. Covers the two version-dependent reads
#    (Options field names) and the two silent-corruption paths (non-finite
#    X_sr in a "successful" fit, savename collisions).
function preflight(configs)
    @assert hasproperty(SR_OPTIONS, :populations) &&
            hasproperty(SR_OPTIONS, :population_size) "SymbolicRegression names these npopulations/npop in older versions — fix the two payload reads"
    @assert length(unique(cellname.(configs))) == length(configs) "savename collision — raise sigdigits in cellname"
    for r in eachrow(CELLS)
        @assert size(r.X_sr) == (3, 200) && size(r.g_sr) == (3, 200) "X_sr/g_sr not 3x200"
        @assert all(isfinite, r.X_sr) && all(isfinite, r.g_sr) "non-finite X_sr/g_sr in a cell with error == \"\""
    end
    row = first(eachrow(CELLS))
    @printf("\nnull ladder, first cell (T=%d m=%d lw=%g ldtt=%g k=%g is=%d)\n",
            row.timepoints, row.mice_per_timepoint, row.lambda_w, row.lambda_dtt,
            row.stop_kappa, row.init_seed)
    @printf("%6s | %s\n", "term", join([@sprintf("%11s", "deg $d") for d in NULL_DEGREES]))
    for term in 1:3
        frows, _ = TERM_FEATURES[term]
        F = Float64.(row.X_sr[frows, :]); y = Float64.(row.g_sr[term, :])
        v = [poly_rel_mse(F, y, d) for d in NULL_DEGREES]
        @printf("%6d | %s\n", term, join([@sprintf("%11.3e", x) for x in v]))
        all(<(NULL_ZERO), v) &&
            @warn "every null rung ~0: a low-degree polynomial already fits g_sr to numerical precision — SR on this term measures nothing" term
    end
    @info "sr preflight passed" runs = length(configs) cells = N_CELLS seeds = K_SEEDS threads = Threads.nthreads()
end

# ── Runner: atomic write-then-rename, cached skip, stride shards. ───────────
function run_sr_sweep(configs, sr_dir; force::Bool = false,
                      shard::Int = 1, nshards::Int = 1)
    @assert 1 <= shard <= nshards "shard must be in 1:nshards"
    mkpath(sr_dir)
    cells = configs[shard:nshards:end]
    n = length(cells)
    @info "shard" shard nshards runs = n of = length(configs) dir = sr_dir
    for (i, c) in enumerate(cells)
        path = cellpath(sr_dir, c)
        if !force && isfile(path)
            @info "skip (cached)" shard i n file = basename(path); continue
        end
        @info "sr" shard i n T = c["timepoints"] m = c["mice_per_timepoint"] term = c["term"] sr_seed = c["sr_seed"]
        local payload
        t = @elapsed payload = try
            run_sr_cell(c)
        catch err
            @warn "run failed" term = c["term"] sr_seed = c["sr_seed"] exception = err
            error_payload(c, err)
        end
        tmp = tempname(sr_dir) * ".jld2"
        wsave(tmp, payload)
        mv(tmp, path; force = true)
        @info "done" shard i n minutes = round(t / 60; digits = 1) sel = payload["sel_expr"] rel_mse = payload["sel_rel_mse"] hill = payload["hill_on_front"] file = basename(path)
    end
    @info "shard complete" shard nshards dir = sr_dir
end

function sr_status(configs, sr_dir)
    done = count(c -> isfile(cellpath(sr_dir, c)), configs)
    @info "sr status" cached = done remaining = length(configs) - done total = length(configs) dir = sr_dir
    return done
end

preflight(CONFIGS)

if abspath(PROGRAM_FILE) == @__FILE__
    k, N = length(ARGS) >= 2 ? (parse(Int, ARGS[1]), parse(Int, ARGS[2])) : (1, 1)
    run_sr_sweep(CONFIGS, SR_DIR; shard = k, nshards = N)
end
