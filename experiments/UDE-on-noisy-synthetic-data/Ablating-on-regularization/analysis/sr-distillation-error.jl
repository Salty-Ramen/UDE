isdefined(@__MODULE__, :SRPopulation) || include("sr-population.jl")
import CairoMakie as CM
using Statistics, DataFrames

"Compare the selected expression to its teacher and to truth at identical predicted states."
function draw_distillation!(layout, runs)
    for (i,ch) in enumerate(SRPopulation.CHANNELS)
        d = filter(:term => ==(i), runs)
        ax = CM.Axis(layout[1,i]; title=ch, xscale=log10, yscale=log10,
            xlabel="Expression vs network (relative L₂)", ylabel="Expression vs truth (relative L₂)")
        x, y, baseline = Float64[], Float64[], Float64[]
        for r in eachrow(d)
            pred = SRPopulation.evaluate(r.sel_tree,r.X_sr,i)
            net, truth = Float64.(r.g_sr[i,:]), Float64.(r.f_sr[i,:])
            all(isfinite,pred) || continue
            loss = mean(abs2,pred .- net)
            abs(loss-r.sel_mse) <= 1e-3 * max(abs(r.sel_mse),eps()) || error("Tree loss mismatch")
            a,b,c = SRPopulation.rho(pred,net), SRPopulation.rho(pred,truth), SRPopulation.rho(net,truth)
            all(v->isfinite(v) && v>0,(a,b,c)) || continue
            push!(x,a); push!(y,b); push!(baseline,c)
        end
        @info "Error panel" channel=ch plotted=length(x) excluded=nrow(d)-length(x)
        CM.scatter!(ax,x,y; markersize=15,color=(:black,0.25),rasterize=2)
        # Notebook overlay: median response within log-x deciles.
        lx = log10.(x); edges = quantile(lx,0:0.1:1)
        bx, by = Float64[], Float64[]
        for j in 1:10
            keep = findall(v -> edges[j] <= v <= edges[j+1], lx)
            length(keep) < 5 && continue
            push!(bx,10^median(lx[keep])); push!(by,median(y[keep]))
        end
        CM.scatterlines!(ax,bx,by;color=:firebrick,linewidth=3,markersize=9)
        CM.hlines!(ax,[median(baseline)]; color=:black,linestyle=:dash,linewidth=1.5,label="Median network vs truth")
        i==1 && CM.axislegend(ax)
    end
end
if abspath(PROGRAM_FILE)==@__FILE__
    SRPopulation.render("sr-distillation-error",draw_distillation!;size=(1320,520))
end
