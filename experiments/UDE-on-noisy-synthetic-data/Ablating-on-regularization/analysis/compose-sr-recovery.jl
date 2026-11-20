include("sr-paired-recovery-error.jl")
include("sr-search-versus-selection.jl")
include("sr-recovery-outcomes.jl")

if abspath(PROGRAM_FILE)==@__FILE__
    "Index proxy: panel cell [r,c] lands at outer [row+r-1, cols(c)], so panels share one grid."
    struct Shifted; g; row::Int; cols; end
    Base.getindex(s::Shifted,r,c) = s.g[s.row+r-1, s.cols(c)]

    SRPopulation.render("sr-recovery-preview", (layout,runs) -> begin
        draw_paired_error!(Shifted(layout,1,c->c+1),runs)
        draw_search_selection!(Shifted(layout,2,c->c+1),runs)
        draw_recovery_outcomes!(Shifted(layout,3,_->2:4),runs)
        for (row,label) in ((1,"A"),(2,"B"),(3,"C"))
            CM.Label(layout[row,1],label; font=:bold,fontsize=30,valign=:top,tellheight=false)
        end
        CM.rowsize!(layout,1,CM.Aspect(2,1.0))
        CM.rowsize!(layout,3,CM.Relative(0.2))
    end; size=(1650,1350))
end
