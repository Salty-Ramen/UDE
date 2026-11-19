# Example manuscript layout, not a commitment to a figure number.
include("early-stopping-and-recovery.jl")
include("regularization-by-sampling-design.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    fits = load_fits()
    fig = notebook_figure(size=(1400, 1750))
    recovery = fig[1,1] = GridLayout()
    design = fig[2,1] = GridLayout()
    draw_recovery!(recovery, fits)
    draw_design!(design, fits)
    save_analysis(fig, "recovery-and-design-preview")
end
