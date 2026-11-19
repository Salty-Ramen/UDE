using DataFrames
import JLD2  # backend for reading sweep records
include("figure-style.jl")
using DrWatson
using Statistics

const STUDY_DIR = dirname(@__DIR__)
const FIT_DIR = joinpath(STUDY_DIR, "Results", "sweep-errmodel-pinned-data")
const CHANNEL_ERRORS = [:rel_l2_gV, :rel_l2_gIFN, :rel_l2_gM]

"""Common population and score for both questions; no cache refresh or fit required."""
function prepare_fits(records)
    keep = filter(records) do r
        isequal(r.error, "") && isequal(r.err_model, "prop") && isequal(r.noise_frac, 0.2)
    end
    isempty(keep) && error("No successful proportional-error fits at noise_frac=0.2")
    keep.score = map(eachrow(keep)) do r
        values = [r[c] for c in CHANNEL_ERRORS]
        all(x -> x isa Real && isfinite(x) && x >= 0, values) || error("Invalid channel error")
        maximum(values)
    end
    keep.stopping = keep.stop_kappa .> 0
    return keep
end

function load_fits(dir=FIT_DIR)
    isdir(dir) || error("Missing input sweep: $dir")
    prepare_fits(collect_results(dir))
end

function save_analysis(fig, name)
    path = joinpath(mkpath(joinpath(STUDY_DIR, "figures")), name * ".pdf")
    CairoMakie.save(path, fig)
    @info "Saved figure" path
    return path
end
