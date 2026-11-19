using CairoMakie
isdefined(@__MODULE__, :load_fits) || include("fit-population.jl")
module RecoveryTruth
include(joinpath(@__DIR__, "..", "..", "common", "synthetic-data-gen.jl"))
end

"""Draw the stopping-arm distributions and show trajectories in pooled score quartiles."""
function draw_recovery!(layout, fits)
    all(>(0), fits.score) || error("Log-density requires positive validation errors")
    edges = quantile(fits.score, 0:0.25:1)
    bins = [clamp(searchsortedlast(edges, s), 1, 4) for s in fits.score]
    colors = Makie.wong_colors()  # default categorical palette used by AoG
    ax = Axis(layout[1, 1:4], xlabel="log₁₀ validation error (worst channel)", ylabel="Density")
    for arm in (false, true)
        scores = fits.score[fits.stopping .== arm]
        isempty(scores) && error("Missing stopping arm: $arm")
        density!(ax, log10.(scores); color=(colors[1+arm], 0.45), label=arm ? "Stopping on" : "Stopping off")
    end
    axislegend(ax)
    tg = vec(first(fits.tg))
    all(t -> vec(t) == tg, fits.tg) || error("Saved trajectory grids differ")
    times = collect(0:0.1:8)
    idx = [argmin(abs.(tg .- t)) for t in times]  # nearest stored sample, not interpolation
    truth = RecoveryTruth.true_g(RecoveryTruth.solve_true(times))
    dense_truth = RecoveryTruth.true_g(RecoveryTruth.solve_true(tg))
    for q in 1:4, (i, label) in enumerate(["g_V", "g_IFN", "g_M"])
        ax = Axis(layout[i+1, q]; title=i == 1 ? "Q$q (n=$(count(==(q), bins)))" : "",
                  ylabel=q == 1 ? label : "", xlabel=i == 3 ? "Time (days)" : "")
        for arm in (false, true)
            # NaN breaks preserve fit identities even when two fits share hyperparameters.
            pts = Point2f[]
            for j in findall((bins .== q) .& (fits.stopping .== arm))
                append!(pts, Point2f.(tg[idx], fits.g_sr[j][i, idx]))
                push!(pts, Point2f(NaN, NaN))
            end
            isempty(pts) || lines!(ax, pts; color=(colors[1+arm], 0.2), linewidth=1, rasterize=2)
        end
        lines!(ax, times, truth[i, :]; color=:black, linestyle=:dot, linewidth=2.5)
        lo, hi = extrema(dense_truth[i, :]); ylims!(ax, min(0, 1.35lo), 1.35hi)
    end
    @info "Recovery quartiles" fits=nrow(fits) edges
    return layout
end

if abspath(PROGRAM_FILE) == @__FILE__
    fig = notebook_figure(size=(1200, 900))
    draw_recovery!(fig.layout, load_fits())
    save_analysis(fig, "early-stopping-and-recovery")
end
