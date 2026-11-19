isdefined(@__MODULE__, :SRPopulation) || include("sr-population.jl")
import CairoMakie as CM
using DataFrames, Statistics

"Count selected structural keys; colour by median expression error on predicted states."
function draw_structures!(layout,runs)
    key_errors = map(eachrow(runs)) do r
        i = r.term
        pred = SRPopulation.evaluate(r.sel_tree,r.X_sr,i)
        (term=i, key=SRPopulation.SRSkeleton.skeleton_info(r.sel_tree,i).str,
         err=SRPopulation.rho(pred,Float64.(r.f_sr[i,:])))
    end |> DataFrame
    finite = filter(:err => isfinite,key_errors)
    stats = combine(groupby(finite,[:term,:key]), nrow=>:n, :err=>median=>:med)
    # Shared range, including forms pooled into 'other', as in Pluto.
    cr = extrema(log10.(stats.med))
    for (i,ch) in enumerate(SRPopulation.CHANNELS)
        g = sort(filter(:term => ==(i),stats),[:n,:key];rev=[true,false])
        top = reverse(first(g,min(8,nrow(g))))
        rest = sum(g.n)-sum(top.n); off = Int(rest>0)
        labels = [length(s)>32 ? first(s,31)*"…" : s for s in top.key]
        rest>0 && pushfirst!(labels,"other ($(nrow(g)-nrow(top)) forms)")
        ax = CM.Axis(layout[1,i];title="$ch: $(nrow(g)) forms",xlabel="Successful SR runs",
                     yticks=(1:length(labels),labels),yticklabelsize=14)
        bars = CM.barplot!(ax,(1:nrow(top)).+off,top.n;direction=:x,color=log10.(top.med),
                           colormap=CM.Reverse(:dense),colorrange=cr)
        rest>0 && CM.barplot!(ax,[1],[rest];direction=:x,color=:grey70)
        i==3 && CM.Colorbar(layout[1,4],bars;label="log₁₀ median error vs truth")
    end
end
if abspath(PROGRAM_FILE)==@__FILE__
    SRPopulation.render("sr-selected-structures",draw_structures!;size=(1560,560))
end
