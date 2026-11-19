isdefined(@__MODULE__, :SRPopulation) || include("sr-population.jl")
import CairoMakie as CM
using DataFrames, Statistics

"A hit means any successful seed of the same fit/channel; compare saved front versus pick."
function search_selection_cells(runs,i)
        trees = SRPopulation.truth_trees()
        truth = SRPopulation.SRSkeleton.skeleton_info(trees[i],i).str
        memo = Dict{Any,Union{Nothing,String}}()
        key(t) = get!(memo,t) do
            try
                SRPopulation.SRSkeleton.skeleton_info(t,i).str
            catch err
                err isa DomainError || rethrow()
                nothing
            end
        end
        invalid = 0
        # Groups share the per-channel memo and invalid counter; fold them serially.
        cells = combine(groupby(filter(:term => ==(i),runs),SRPopulation.CELL); threads=false) do d
            found = selected = false
            for r in eachrow(d)
                length(r.pareto)==length(r.pareto_trees) || error("Misaligned front metadata")
                r.sel_tree in r.pareto_trees || error("Selected tree not on front")
                keys = key.(r.pareto_trees); invalid += count(isnothing,keys)
                hit = truth in keys; pick = key(r.sel_tree)==truth
                pick && !hit && error("Selected truth absent from front")
                found |= hit; selected |= pick
            end
            err = SRPopulation.rho(Float64.(first(d.g_sr)[i,:]), Float64.(first(d.f_sr)[i,:]))
            (; found,selected,err,seeds=nrow(d))
        end
        @info "Unkeyable front members" term=i invalid
        return cells
end

function draw_search_selection!(layout,runs)
    for (i,ch) in enumerate(SRPopulation.CHANNELS)
        cells = search_selection_cells(runs,i)
        good = filter(:err => e->isfinite(e)&&e>0,cells)
        @info "Front audit" channel=ch cells=nrow(cells) seeds=extrema(cells.seeds) excluded=nrow(cells)-nrow(good) found=sum(good.found) selected=sum(good.selected)
        edges = quantile(good.err,0:0.1:1)
        good.bin = [clamp(searchsortedlast(edges,e),1,10) for e in good.err]
        g = sort(combine(groupby(good,:bin),:err=>median=>:x,:found=>sum=>:found,:selected=>sum=>:selected,nrow=>:n),:bin)
        ax = CM.Axis(layout[1,i];title=ch,xscale=log10,xlabel="Network vs truth (relative L₂)",ylabel="Fraction of fits with a hit")
        for (col,color,label) in ((:found,:firebrick,"On stored front"),(:selected,:steelblue,"Selected"))
            p = g[!,col] ./ g.n; z = 1.96
            center = (p .+ z^2 ./ (2 .* g.n)) ./ (1 .+ z^2 ./ g.n)
            half = z .* sqrt.(p .* (1 .- p) ./ g.n .+ z^2 ./ (4 .* g.n.^2)) ./ (1 .+ z^2 ./ g.n)
            CM.errorbars!(ax,g.x,p,max.(0,p.-center.+half),max.(0,center.+half.-p);color=(color,0.6),whiskerwidth=8,linewidth=1.2)
            CM.scatterlines!(ax,g.x,p;color,label,markersize=12,linewidth=2)
        end
        CM.ylims!(ax,-0.03,1.03); i==1 && CM.axislegend(ax)
    end
end
if abspath(PROGRAM_FILE)==@__FILE__
    SRPopulation.render("sr-search-versus-selection",draw_search_selection!;size=(1420,540))
end
