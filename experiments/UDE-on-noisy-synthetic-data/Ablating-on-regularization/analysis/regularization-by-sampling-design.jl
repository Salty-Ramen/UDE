using CairoMakie
isdefined(@__MODULE__, :load_fits) || include("fit-population.jl")

"""Median worst-channel relative error by λ and design, pooling stopping arms and seeds."""
function draw_design!(layout, fits)
    keys = [:timepoints, :mice_per_timepoint, :lambda_w, :lambda_dtt]
    cells = combine(groupby(fits, keys), :score => median => :med_err, nrow => :n)
    vals(c) = sort(unique(c))
    ws, ds = vals(cells.lambda_w), vals(cells.lambda_dtt)
    lookup = Dict((r.timepoints, r.mice_per_timepoint, r.lambda_w, r.lambda_dtt) => r.med_err
                  for r in eachrow(cells))
    designs = vals(cells.timepoints)
    for (i, m) in enumerate(vals(cells.mice_per_timepoint)), (j, t) in enumerate(designs)
        ax = Axis(layout[i,j]; title="T=$t, m=$m", xlabel="λ_w", ylabel="λ_dtt",
                  xticks=(1:length(ws), string.(ws)), yticks=(1:length(ds), string.(ds)),
                  xticklabelrotation=π/4)
        # Require the complete design rather than silently painting missing cells as zero.
        z = [lookup[(t,m,w,d)] for w in ws, d in ds]
        hm = heatmap!(ax, 1:length(ws), 1:length(ds), z;
                      colormap=Reverse(:dense), colorrange=extrema(cells.med_err))
        if i == 1 && j == 1
            Colorbar(layout[1:length(unique(cells.mice_per_timepoint)), length(designs)+1], hm;
                     label="Median worst-channel error")
        end
    end
    @info "Design summary" fits=nrow(fits) cells=nrow(cells) fits_per_cell=extrema(cells.n)
    return layout
end

if abspath(PROGRAM_FILE) == @__FILE__
    fig = Figure(size=(1100, 850))
    draw_design!(fig.layout, load_fits())
    save_analysis(fig, "regularization-by-sampling-design")
end
