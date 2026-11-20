include("sr-paired-recovery-error.jl")
include("sr-search-versus-selection.jl")
include("sr-recovery-outcomes.jl")

if abspath(PROGRAM_FILE)==@__FILE__
    SRPopulation.render("sr-recovery-preview", (layout,runs) -> begin
        panels = (("A", draw_paired_error!), ("B", draw_search_selection!), ("C", draw_recovery_outcomes!))
        for (i,(label,draw)) in enumerate(panels)
            CM.Label(layout[i,1],label; font=:bold,fontsize=30,valign=:top)
            row = layout[i,2] = CM.GridLayout()
            draw(row,runs)
        end
        CM.rowsize!(layout,3,CM.Relative(0.23))
    end; size=(1650,1350))
end
