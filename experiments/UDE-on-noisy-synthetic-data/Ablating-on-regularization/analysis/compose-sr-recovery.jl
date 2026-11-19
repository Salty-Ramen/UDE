include("sr-distillation-error.jl")
include("sr-selected-structures.jl")
include("sr-search-versus-selection.jl")

if abspath(PROGRAM_FILE)==@__FILE__
    SRPopulation.render("sr-recovery-preview", (layout,runs) -> begin
        for (i,draw) in enumerate((draw_distillation!, draw_structures!, draw_search_selection!))
            row = layout[i,1] = CM.GridLayout()
            draw(row,runs)
        end
    end; size=(1650,1350))
end
