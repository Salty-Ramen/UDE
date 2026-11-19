isdefined(@__MODULE__, :SRPopulation) || include("sr-population.jl")
import CairoMakie as CM
using DataFrames

"Count selected sign-conditioned structural keys across successful searches, not unique fits."
function draw_structures!(layout,runs)
    for (i,ch) in enumerate(SRPopulation.CHANNELS)
        counts = Dict{String,Int}()
        d = filter(:term => ==(i),runs)
        for tree in d.sel_tree
            key = SRPopulation.SRSkeleton.skeleton_info(tree,i).str
            counts[key] = get(counts,key,0)+1
        end
        ordered = sort(collect(counts); by=p->(-last(p),first(p)))
        top = reverse(first(ordered,min(8,length(ordered))))
        labels = first.(top); n = last.(top)
        rest = nrow(d)-sum(n)
        if rest>0
            pushfirst!(labels,"other ($(length(ordered)-length(top)) forms)"); pushfirst!(n,rest)
        end
        ax = CM.Axis(layout[1,i]; title="$ch: $(length(counts)) forms", xlabel="Successful SR runs",
            yticks=(1:length(n),[length(s)>32 ? first(s,31)*"…" : s for s in labels]),yticklabelsize=12)
        CM.barplot!(ax,1:length(n),n;direction=:x,color=:steelblue)
    end
end
if abspath(PROGRAM_FILE)==@__FILE__
    SRPopulation.render("sr-selected-structures",draw_structures!)
end
