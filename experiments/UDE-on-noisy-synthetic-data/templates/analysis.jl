### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 99cdf187-f4a4-430c-84e9-7b71aec49f91
md"""
# Question: …

Input sweep and selection criteria: …

Finding (including limitations): …
"""

# ╔═╡ 05cfd390-b97e-4b07-98a8-d66356068b91
begin
    import Pkg
    project_file = Base.current_project(@__DIR__)
    isnothing(project_file) && error("Save this notebook inside the UDE project")
    project_root = dirname(project_file)
    Pkg.activate(project_root)
    # Set these once, while still in scratch. Moving the notebook won't change them.
    study = "Ablating-on-regularization"
    input_run = "sr-sweep"
    study_dir = joinpath(project_root, "experiments", "UDE-on-noisy-synthetic-data", study)
    sweep_dir = joinpath(study_dir, "Results", input_run)
    figures_dir = joinpath(study_dir, "figures")
    summaries_dir = joinpath(study_dir, "Results", "summaries")
    # Create output directories only when exporting, e.g. mkpath(summaries_dir).
end

# ╔═╡ Cell order:
# ╟─99cdf187-f4a4-430c-84e9-7b71aec49f91
# ╠═05cfd390-b97e-4b07-98a8-d66356068b91
