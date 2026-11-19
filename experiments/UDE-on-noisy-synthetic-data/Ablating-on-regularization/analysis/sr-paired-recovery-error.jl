isdefined(@__MODULE__, :SRPopulation) || include("sr-population.jl")
import CairoMakie as CM
using DataFrames, Printf

"Paired comparison on the same predicted states; one point per successful SR run."
function draw_paired_error!(layout,runs)
    rows = map(eachrow(runs)) do r
        i = r.term
        pred = SRPopulation.evaluate(r.sel_tree,r.X_sr,i)
        net, truth = Float64.(r.g_sr[i,:]), Float64.(r.f_sr[i,:])
        (; term=i, x=SRPopulation.rho(net,truth), y=SRPopulation.rho(pred,truth),
           dist=SRPopulation.rho(pred,net))
    end |> DataFrame
    positive(x) = isfinite(x) && x>0
    plotted = filter(r -> positive(r.x) && positive(r.y),rows)
    lo = 0.8min(minimum(plotted.x),minimum(plotted.y)); hi = 1.25max(maximum(plotted.x),maximum(plotted.y))
    cr = extrema(log10.(filter(positive,plotted.dist)))
    for (i,ch) in enumerate(SRPopulation.CHANNELS)
        d = filter(:term => ==(i),plotted); total=count(==(i), rows.term)
        ax = CM.Axis(layout[1,i];title="$ch (n=$(nrow(d)) of $total)",xscale=log10,yscale=log10,
            aspect=1, limits=((lo,hi),(lo,hi)),xlabel="Network vs truth (relative L₂)",
            ylabel=i==1 ? "Expression vs truth (relative L₂)" : "",yticklabelsvisible=i==1)
        CM.lines!(ax,[lo,hi],[lo,hi];color=:black,linestyle=:dash,linewidth=1.5)
        CM.scatter!(ax,d.x,d.y;color=log10.(max.(d.dist,eps())),colorrange=cr,
                    colormap=(:viridis,0.25),markersize=15,rasterize=2)
        isempty(d) && continue
        fraction=count(d.y .< d.x)/nrow(d)
        CM.text!(ax,0.04,0.94;text=@sprintf("below: %.0f%%",100fraction),space=:relative,align=(:left,:top),fontsize=18)
        @info "Paired error" channel=ch fraction plotted=nrow(d) excluded=total-nrow(d)
    end
    CM.Colorbar(layout[1,4];colormap=:viridis,limits=cr,label="log₁₀ distillation error")
end
if abspath(PROGRAM_FILE)==@__FILE__
    SRPopulation.render("sr-paired-recovery-error",draw_paired_error!;size=(1320,520))
end
