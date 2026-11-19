# Figure-scoped defaults from the two exploratory Pluto notebooks.
# No global set_theme!: including a script must not restyle unrelated plots.
import CairoMakie
notebook_figure(;size) = CairoMakie.Figure(;size, fontsize=20,
    fonts=(;regular="CMU Serif", bold="CMU Serif Bold", italic="CMU Serif Italic"),
    Axis=(;titlesize=23, xlabelsize=20, ylabelsize=20, xticklabelsize=18, yticklabelsize=18),
    Colorbar=(;labelsize=20, ticklabelsize=18), Legend=(;labelsize=18, titlesize=20))
