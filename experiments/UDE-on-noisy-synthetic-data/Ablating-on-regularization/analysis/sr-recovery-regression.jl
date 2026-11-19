# Forest plot from tmp/sr-recovery-regression.jl. Only the main-effects models
# are needed here, not the exploratory risk-difference or mediation analyses.
isdefined(@__MODULE__, :search_selection_cells) || include("sr-search-versus-selection.jl")
using GLM, StatsModels, CategoricalArrays, Printf

const SR_FACTORS = [:timepoints, :mice_per_timepoint, :lambda_w, :lambda_dtt, :stop_kappa, :init_seed]
const SR_STAGES = [(:found, "Found on front"), (:selected, "Selected | found")]

function recovery_contrasts(runs)
    contrasts = DataFrame(channel=String[], stage=String[], factor=Symbol[], level=Float64[], beta=Float64[], se=Float64[])
    for (i,ch) in enumerate(SRPopulation.CHANNELS)
        cells = search_selection_cells(runs,i)
        for (outcome,stage) in SR_STAGES
            d = filter(r -> (outcome == :found || r.found) && isfinite(r.err) && r.err > 0, cells)
            k, n = count(d[!,outcome]), nrow(d)
            modelled = min(k,n-k) >= 20
            @info "Regression support" channel=ch stage events=k observations=n modelled
            modelled || continue
            # Categorise AFTER subsetting: omit absent levels, use smallest as reference.
            fac = [c for c in SR_FACTORS if length(unique(d[!,c])) > 1]
            for c in fac
                d[!,c] = categorical(d[!,c]; levels=sort(unique(d[!,c])))
            end
            rhs = foldl(+,term.(fac);init=term(1))
            model = glm(term(outcome) ~ rhs,d,Binomial(),LogitLink())
            p = predict(model)
            if minimum(p)<1e-6 || maximum(p)>1-1e-6
                @warn "Boundary predictions; check separation before interpreting intervals" ch stage
            end
            names, b, se = coefnames(model), coef(model), stderror(model)
            for c in fac, v in levels(d[!,c])[2:end]
                j = findfirst(==("$c: $v"),names)
                isnothing(j) && error("Missing coefficient for $c: $v")
                push!(contrasts,(ch,stage,c,Float64(unwrap(v)),b[j],se[j]))
            end
        end
    end
    return contrasts
end

function draw_recovery_regression!(layout,runs)
    contrasts = recovery_contrasts(runs)
    channels = SRPopulation.CHANNELS
    colors = [:firebrick,:darkorange,:steelblue]
    for (j,c) in enumerate(SR_FACTORS), (i,(_,stage)) in enumerate(SR_STAGES)
        vs = sort(unique(contrasts.level[contrasts.factor .== c]))
        labs = c === :lambda_dtt ? [@sprintf("%.3gλ*",v / 2.7341519f-5) for v in vs] : [@sprintf("%.3g",v) for v in vs]
        ax = CM.Axis(layout[i,j]; title=string(c),titlesize=17,xticks=(1:length(vs),labs),
                     xticklabelrotation=π/4,xticklabelsize=13,ylabel=j==1 ? "$stage\nlog odds ratio" : "")
        CM.hlines!(ax,[0];color=:grey65,linestyle=:dash)
        isempty(vs) && continue
        CM.xlims!(ax,0.5,length(vs)+0.5)
        for (ci,ch) in enumerate(channels)
            s = filter(r->r.factor==c && r.stage==stage && r.channel==ch,contrasts)
            isempty(s) && continue
            x = Float64.(indexin(s.level,vs)) .+ 0.16*(ci-(length(channels)+1)/2)
            CM.errorbars!(ax,x,s.beta,1.96 .* s.se;color=colors[ci],whiskerwidth=8)
            CM.scatter!(ax,x,s.beta;color=colors[ci],markersize=9,label=ch)
        end
        j==1 && i==1 && CM.axislegend(ax;position=:lt,labelsize=12)
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    SRPopulation.render("sr-recovery-regression",draw_recovery_regression!;size=(1450,720))
end
