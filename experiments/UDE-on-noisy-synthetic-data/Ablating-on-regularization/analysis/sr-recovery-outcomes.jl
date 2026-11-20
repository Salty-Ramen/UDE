isdefined(@__MODULE__, :search_selection_cells) || include("sr-search-versus-selection.jl")

"Fig. 3C: partition fits, not individual SR runs, by their outcome across seeds."
function draw_recovery_outcomes!(layout, runs)
    outcomes = map(enumerate(SRPopulation.CHANNELS)) do (i, channel)
        d = search_selection_cells(runs, i)
        @assert !isempty(d) && all(==(8), d.seeds) "Expected eight successful SR seeds per fit/channel"
        @assert all(.!d.selected .| d.found)
        counts = (count(!, d.found), count(d.found .& .!d.selected), count(d.selected))
        @assert sum(counts) == nrow(d)
        @info "Fit outcomes" channel absent=counts[1] passed_over=counts[2] selected=counts[3] total=nrow(d)
        (; counts, total=nrow(d))
    end
    ax = CM.Axis(layout[1,1]; xlabel="Percentage of fits", xticks=0:20:100,
                 yticks=(1:length(outcomes), SRPopulation.CHANNELS), yreversed=true)
    CM.xlims!(ax, 0, 100)
    left = zeros(length(outcomes))
    for (j,(color,label)) in enumerate(((:grey80,"Absent from saved fronts"),
                                         (:firebrick,"Found, never selected"),
                                         (:steelblue,"Selected by ≥1 of 8 SR seeds")))
        pct = [100*r.counts[j]/r.total for r in outcomes]
        CM.barplot!(ax, eachindex(left), left .+ pct; fillto=copy(left), direction=:x, color, label, width=0.65)
        for i in eachindex(pct)
            pct[i] < 6 && continue
            CM.text!(ax, left[i]+pct[i]/2, i; text=string(outcomes[i].counts[j]),
                     color=j==1 ? :black : :white, align=(:center,:center))
        end
        left .+= pct
    end
    CM.Legend(layout[2,1],ax; orientation=:horizontal,framevisible=false,labelsize=16)
end

if abspath(PROGRAM_FILE) == @__FILE__
    SRPopulation.render("sr-recovery-outcomes",draw_recovery_outcomes!;size=(940,360))
end
