# ============================================================================
# sweep-plots.jl — read-only plots of the cached UDE ablation sweep.
#
# Reads one JLD2 result per grid cell from SWEEP_DIR, keeps only successful
# fits (error == ""), and writes two PDFs. Does NOT run any fit or the sweep.
#
# Plot 1: state-fit rel-L2 vs observation noise, one line per mouse allocation.
# Plot 2: true state trajectory vs every cell's predicted states, colored by noise.
# ============================================================================

using DrWatson, JLD2, DataFrames, Statistics, CairoMakie, Pkg

# solve_true(tsave) -> clean 3×N Array of oracle states. NOTE: this include also
# runs `@quickactivate "UDE"`, `using OrdinaryDiffEq`, and one clean oracle solve
# at load time. It does NOT load the fitting stack and never refits — it just
# activates the project (so Pkg.project().path below matches the sweep) and
# defines solve_true. It is heavier than a bare `using`, but not a fit.
include(joinpath(@__DIR__, "synthetic-data-gen.jl"))

# SWEEP_DIR: copied verbatim from sweep.jl. Resolves against the project that
# the include just activated. (If you run this outside the UDE project, replace
# with the absolute path string instead.)
const SWEEP_DIR = joinpath(dirname(Pkg.project().path),
                           "experiments", "UDE-on-noisy-synthetic-data", "Results", "sweep")

# PDFs go next to the other results (the Results/ dir that holds sweep/).
const RESULTS_DIR = dirname(SWEEP_DIR)
res_path(name) = joinpath(RESULTS_DIR, name)

# ── Load + filter ───────────────────────────────────────────────────────────
df = collect_results(SWEEP_DIR)
df = df[df.error .== "", :]                       # drop failed fits (empty arrays / NaN metrics)
# collect_results widens every column to Union{Missing,T} (to pad any file that
# lacks a key). Kept rows have no real missings, so narrow the columns used as
# Makie coordinates back to concrete types — Makie can't convert Union{Missing,_}.
disallowmissing!(df, [:noise_frac,
                      :rel_l2_V, :rel_l2_IFN, :rel_l2_M,
                      :rel_l2_gV, :rel_l2_gIFN, :rel_l2_gM])
df.alloc = string.(df.timepoints, "×", df.mice_per_timepoint)   # e.g. "5×4"

# ── Schema check: run this FIRST. Fails fast if any payload key differs from
#    the assumed schema, before either plot is attempted. ─────────────────────
let r = first(eachrow(df))
    for c in (:timepoints, :mice_per_timepoint, :noise_frac, :seed,
              :rel_l2_V, :rel_l2_IFN, :rel_l2_M, :tg, :X_sr, :error)
        @assert hasproperty(r, c) "missing column $c — check payload keys in sweep.jl's run_cell"
    end
    @assert size(r.X_sr) == (3, 200) "X_sr expected 3×200, got $(size(r.X_sr))"
    @assert length(r.tg) == 200      "tg expected length 200, got $(length(r.tg))"
    @info "schema check passed" n_kept_cells = nrow(df)
end

const STATES    = ["V", "IFN", "M"]
const STATE_ERR = (:rel_l2_V, :rel_l2_IFN, :rel_l2_M)
# Allocations ordered by timepoints descending: 20×1, 10×2, 5×4, 4×5.
allocs = sort(unique(df.alloc); by = a -> parse(Int, split(a, "×")[1]), rev = true)

# ── Plot 1: state error vs noise ────────────────────────────────────────────
# Aggregation: y = MEAN over the (up to 3) seeds per (allocation, noise); seed
# spread shown as a MIN–MAX band. Mean+min/max (not std) because n≤3 per point.
function agg(sub, col)                       # sub = one allocation's rows
    ν  = sort(unique(sub.noise_frac))
    m  = [mean(sub[sub.noise_frac .== v, col])    for v in ν]
    lo = [minimum(sub[sub.noise_frac .== v, col]) for v in ν]
    hi = [maximum(sub[sub.noise_frac .== v, col]) for v in ν]
    (ν, m, lo, hi)
end

function plot_error_vs_noise(err_cols, glabels, fname, suptitle)
    colors = Makie.wong_colors()
    fig = Figure(size = (560, 820))
    for (i, lbl) in enumerate(glabels)
        ax = Makie.Axis(fig[i, 1]; xlabel = "noise fraction",
                        ylabel = "rel-L2 ($lbl)",
                        title = i == 1 ? suptitle : "")
        for (k, a) in enumerate(allocs)
            sub = df[df.alloc .== a, :]
            ν, m, lo, hi = agg(sub, err_cols[i])
            band!(ax, ν, lo, hi; color = (colors[k], 0.15))
            lines!(ax, ν, m; color = colors[k], label = a)
            scatter!(ax, ν, m; color = colors[k], markersize = 6)
        end
        i == 1 && axislegend(ax; position = :lt, labelsize = 9)
    end
    save(res_path(fname), fig)
    fig
end

plot_error_vs_noise(STATE_ERR, STATES, "sweep_state_error_vs_noise.pdf",
                    "state fit error vs noise")
# Optional companion figure for the missing-term error (uncomment if wanted):
# plot_error_vs_noise((:rel_l2_gV, :rel_l2_gIFN, :rel_l2_gM), ["g_V","g_IFN","g_M"],
#                     "sweep_g_error_vs_noise.pdf", "missing-term fit error vs noise")

# ── Plot 2: true trajectory vs predictions, sectioned by sampling scheme ─────
# Rows = states (V, IFN, M); columns = mouse allocation (timepoints×mice). Each
# panel overlays that scheme's predicted trajectories (3 noise × 3 seeds), each
# coloured by its rel-L2 error vs the true model, over the (dotted) true curve.
# Read down a column for one scheme's fanning, across a row to compare schemes.
# Y-axes are linked per row so schemes share a scale.
tg    = first(df.tg)                # all cells share this 200-pt grid (Float32)
Xtrue = solve_true(tg)              # 3×200 clean ground-truth states
# Colour by rel-L2 error vs the true model (stored rel_l2_*, computed vs clean
# truth). rel-L2 is normalised, so one shared scale over all states is comparable.
# Low error = bright (matches Robinson-cohort.jl).
errvals    = vcat(df.rel_l2_V, df.rel_l2_IFN, df.rel_l2_M)
emin, emax = extrema(errvals)
cg = cgrad(:viridis; rev = true)
ncolor(e) = cg[clamp((e - emin) / (emax - emin + eps()), 0, 1)]

fig2 = Figure(size = (250 * length(allocs) + 90, 780))
axs  = Matrix{Makie.Axis}(undef, length(STATES), length(allocs))
for (i, lbl) in enumerate(STATES), (j, a) in enumerate(allocs)
    ax = Makie.Axis(fig2[i, j];
                    xlabel = i == length(STATES) ? "Time (Days)" : "",
                    ylabel = j == 1 ? lbl : "",
                    title  = i == 1 ? a : "")
    axs[i, j] = ax
    for r in eachrow(df[df.alloc .== a, :])
        lines!(ax, tg, r.X_sr[i, :]; color = ncolor(r[STATE_ERR[i]]), linewidth = 0.8)
    end
    lines!(ax, tg, Xtrue[i, :]; color = :black, linewidth = 2.5,
           linestyle = :dot, label = "true")
    i == 1 && j == 1 && axislegend(ax; position = :rt, labelsize = 8)
end
for i in 1:length(STATES)
    linkyaxes!(axs[i, :]...)
end
Colorbar(fig2[1:length(STATES), length(allocs) + 1]; colormap = cg,
         colorrange = (emin, emax), label = "state rel-L2 vs true")
save(res_path("sweep_pred_vs_true_by_scheme.pdf"), fig2)

@info "plots written" dir = RESULTS_DIR
