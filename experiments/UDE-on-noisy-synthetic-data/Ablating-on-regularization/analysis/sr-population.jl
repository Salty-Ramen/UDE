module SRPopulation
using DataFrames, Statistics, DrWatson
import JLD2
import CairoMakie as CM
const STUDY = dirname(@__DIR__)
const ROOT = normpath(joinpath(STUDY, "..", "..", ".."))
include(joinpath(ROOT, "Src", "SweepCache.jl"))
include(joinpath(ROOT, "Src", "SRSkeleton.jl"))
module Oracle
include(joinpath(@__DIR__, "..", "..", "common", "synthetic-data-gen.jl"))
end
const CHANNELS = ["g_V", "g_IFN", "g_M"]
const CELL = setdiff(SweepCache.SR_KEYS, [:term, :sr_seed])
const FEATURES = ([2,3], [1,3], [2])
const OPS = (+, -, *, /, (x,n) -> inv(1 + (max(x,zero(x))+oftype(x,1e-6))^(-n)))
rho(a,b) = sqrt(sum(abs2, a .- b)) / max(sqrt(sum(abs2,b)), eps())

function evaluate(rows, X, term)
    values = Any[]
    for (deg, isc, val, feat, op, a, b) in rows
        v = if deg == 0
            isc ? val : Float64.(X[FEATURES[term][feat], :])
        elseif deg == 2
            OPS[op].(values[a], values[b])
        else
            error("Unexpected node degree $deg")
        end
        push!(values,v)
    end
    v = last(values); return v isa Number ? fill(v,size(X,2)) : v
end

# Encode the oracle in the same postorder representation as the stored candidates.
function truth_trees()
    p = Oracle.TRUE_PARAMS
    c(v) = (0,true,v,0,0,0,0); x(i) = (0,false,0.,i,0,0,0)
    b(op,a,z) = (2,false,0.,0,op,a,z)
    [[c(p.r_v_ifn), x(1), b(3,1,2), c(p.r_v_M), x(2), b(3,4,5), b(1,3,6)],
     [x(1), c(p.k2^(-1/p.n)), b(3,1,2), c(p.n), b(5,3,4), c(p.k1), b(3,5,6),
      x(2), c(p.r_ifn_M), b(3,8,9), b(1,7,10)],
     [c(p.r_M_ifn), x(1), b(3,1,2)]]
end

function snapshot(source, name, keys, cols)
    df = SweepCache.load_sweep_cached(joinpath(STUDY,"Results",source);
        cache_path=joinpath(ROOT,"outputs","sweep-cache",name*".jld2"), run_keys=keys,
        required_columns=[:error; cols], refresh=get(ENV,"SR_REFRESH","0")=="1")
    ok = filter(:error => e -> isequal(e,""), df)
    @info "Sweep population" source successful=nrow(ok) excluded=nrow(df)-nrow(ok)
    select(ok, unique([keys; cols]))
end

function load_runs()
    sr = snapshot("sr-sweep","sr-sweep",SweepCache.SR_KEYS,
        [:binops,:sel_tree,:pareto_trees,:pareto,:sel_mse])
    all(==(["+","-","*","/","hill"]),sr.binops) || error("Operator mapping changed")
    fits = snapshot("sweep-errmodel-pinned-data","fit-sweep",SweepCache.FIT_KEYS,[:X_sr,:g_sr,:f_sr,:tg])
    joined = innerjoin(sr,select(fits,[CELL; :X_sr; :g_sr; :f_sr; :tg]),on=CELL,validate=(false,true))
    nrow(joined)==nrow(sr)>0 || error("Missing fit records for SR runs")
    grid = vec(first(joined.tg)); all(t->vec(t)==grid,joined.tg) || error("Mixed trajectory grids")
    X = Oracle.solve_true(Float64.(grid)); G = Oracle.true_g(X)
    for (i,t) in enumerate(truth_trees())
        rho(evaluate(t,X,i),G[i,:]) < 1e-4 || error("Oracle tree mismatch for term $i")
    end
    return joined
end

function render(name, draw, runs=load_runs(); size=(1450,480))
    fig = CM.Figure(;size); draw(fig.layout,runs)
    path = joinpath(mkpath(joinpath(STUDY,"figures")),name*".pdf")
    CM.save(path,fig); @info "Saved" path
end
end
