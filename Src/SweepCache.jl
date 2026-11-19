module SweepCache

using Dates
using DataFrames
using DrWatson: collect_results
import JLD2

export load_sweep_cached, FIT_KEYS, SR_KEYS

const SCHEMA_VERSION = 1
const FIT_KEYS = [:timepoints, :mice_per_timepoint, :noise_frac, :seed,
                  :init_seed, :lambda_w, :lambda_dt, :lambda_dtt,
                  :stop_kappa, :output_rescale, :err_model]
const SR_KEYS = [:timepoints, :mice_per_timepoint, :lambda_w, :lambda_dtt,
                 :stop_kappa, :init_seed, :seed, :err_model, :noise_frac,
                 :term, :sr_seed]

function validate_table(df, required_columns, run_keys)
    df isa AbstractDataFrame || error("Snapshot payload is not a DataFrame")
    isempty(run_keys) && error("Provide run_keys to validate run identities")
    needed = unique([Symbol.(required_columns); Symbol.(run_keys)])
    absent = setdiff(needed, propertynames(df))
    isempty(absent) || error("Sweep is missing required columns: $absent")
    for key in run_keys
        any(ismissing, df[!, key]) && error("Missing run identity in column $key")
    end
    nrow(unique(df, run_keys)) == nrow(df) || error("Duplicate sweep run identities")
    return df
end

# Resolve existing ancestors too, so a symlink cannot place a snapshot inside
# the source directory. The destination itself need not exist yet.
function resolved_path(path)
    p = abspath(path)
    ispath(p) && return realpath(p)
    return joinpath(resolved_path(dirname(p)), basename(p))
end

"""
    load_sweep_cached(source; cache_path, run_keys, required_columns=[:error], refresh=false)

Collect a sweep once and persist its complete DataFrame (including error rows).
Ordinary calls read only the snapshot, NOT the source files or their timestamps.
Use `refresh=true` to rebuild after runs are added, replaced, or deleted; then
set it back to false. The first build still pays the full collection cost.

The cache must be outside the source tree. Source provenance, schema, row count,
required columns, and unique/nonmissing run identities are checked on every load.
Invalid snapshots raise an error rather than silently triggering a slow rebuild.
Writes use a temporary file in the destination directory followed by a rename.
This is a single-writer helper: do not refresh the same cache concurrently.
"""
function load_sweep_cached(source; cache_path, run_keys,
                           required_columns=[:error], refresh::Bool=false)
    src = realpath(source)
    isdir(src) || error("Sweep source is not a directory: $src")
    dest = resolved_path(cache_path)
    relative = relpath(dest, src)
    (relative == "." || first(splitpath(relative)) != "..") &&
        error("Snapshot must be outside the sweep source directory: $dest")
    keys = Symbol.(run_keys)

    if isfile(dest) && !refresh
        saved = JLD2.load(dest)
        get(saved, "schema_version", nothing) == SCHEMA_VERSION ||
            error("Incompatible snapshot schema; rebuild with refresh=true")
        get(saved, "source", nothing) == src || error("Snapshot source mismatch: $dest")
        get(saved, "run_keys", nothing) == keys || error("Snapshot run_keys mismatch: $dest")
        all(k -> haskey(saved, k), ["table", "row_count", "created_at"]) ||
            error("Incomplete snapshot: $dest")
        df = validate_table(saved["table"], required_columns, keys)
        nrow(df) == saved["row_count"] || error("Snapshot row count mismatch: $dest")
        @info "Loaded sweep snapshot (source freshness NOT checked; refresh=true to rebuild)" cache=dest created_at=saved["created_at"] rows=nrow(df)
        return df
    end

    @info "Collecting full sweep for snapshot" source=src
    df = validate_table(collect_results(src), required_columns, keys)
    created_at = string(now(UTC)) * "Z"
    mkpath(dirname(dest))
    tmp, io = mktemp(dirname(dest))
    close(io)
    try
        JLD2.jldsave(tmp; schema_version=SCHEMA_VERSION, source=src,
                     run_keys=keys, row_count=nrow(df), created_at, table=df)
        # Unlike mv(...; force=true), rename does not first remove the old file.
        Base.Filesystem.rename(tmp, dest)
    finally
        isfile(tmp) && rm(tmp)
    end
    @info "Saved sweep snapshot" cache=dest created_at rows=nrow(df)
    return df
end

end # module
