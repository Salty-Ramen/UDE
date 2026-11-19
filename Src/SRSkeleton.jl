module SRSkeleton
export skeleton, skeleton_info

const FEATURES = (("IFN", "M"), ("V", "M"), ("IFN",))
const Scalar = Rational{BigInt}

# Values are retained for net-sign determination, NEVER used to infer ties.
# fitted=false marks exact structural numbers introduced by algebra (e.g. x+x).
struct Form
    op::Symbol
    args::Vector{Form}
    name::String
    value::Scalar
    fitted::Bool
end
Form(op::Symbol, args::Vector{Form}=Form[], name="") =
    Form(op, args, name, zero(Scalar), false)
function scalar(v, fitted=false)
    isfinite(v) || throw(DomainError(v, "non-finite constant in SR skeleton"))
    q = Scalar(v)
    Form(:scalar, Form[], "", q, fitted && !iszero(q))
end
isscalar(x) = x.op == :scalar
iszeroform(x) = isscalar(x) && iszero(x.value)
hasparam(x) = x.fitted || any(hasparam, x.args)
number(q) = denominator(q) == 1 ? string(numerator(q)) :
            "($(numerator(q))/$(denominator(q)))"
# Structural keys omit fitted magnitudes, but not signs or structural numbers.
key(x) = isscalar(x) ? (x.fitted ? (x.value < 0 ? "-c" : "c") : number(x.value)) :
         x.op == :var ? x.name : string(x.op, "(", join(key.(x.args), ","), ")")

function split_scale(x)
    isscalar(x) && return (x, scalar(1))
    if x.op == :mul
        i = findfirst(isscalar, x.args)
        if i !== nothing
            rest = x.args[setdiff(eachindex(x.args), [i])]
            return (x.args[i], length(rest) == 1 ? only(rest) : Form(:mul, rest))
        end
    end
    (scalar(1), x)
end
scaled(x) = first(split_scale(x)).fitted

function add(xs)
    terms = Form[]
    for x in xs
        append!(terms, x.op == :add ? x.args : [x])
    end
    # Only collect bases with NO fitted parameters: hill(x,c1) and hill(x,c2)
    # are independent atoms even when their fitted values happen to coincide.
    groups = Dict{String,Tuple{Form,Form}}()
    held = Form[]
    for t in terms
        c, base = split_scale(t)
        if hasparam(base)
            push!(held, t)
        else
            k = key(base)
            if haskey(groups, k)
                prev, base = groups[k]
                c = scalar(prev.value + c.value, prev.fitted || c.fitted)
            end
            groups[k] = (c, base)
        end
    end
    for k in sort!(collect(keys(groups)))
        c, base = groups[k]
        iszeroform(c) || push!(held, mul([c, base]; distribute=false))
    end
    filter!(x -> !iszeroform(x), held)
    sort!(held; by=key)
    isempty(held) ? scalar(0) : length(held) == 1 ? only(held) : Form(:add, held)
end

function mul(xs; distribute=true)
    factors = Form[]
    for x in xs
        append!(factors, x.op == :mul ? x.args : [x])
    end
    c = scalar(1)
    rest = Form[]
    for x in factors
        if isscalar(x)
            c = scalar(c.value * x.value, c.fitted || x.fitted)
        else
            push!(rest, x)
        end
    end
    iszeroform(c) && return scalar(0)
    sort!(rest; by=key) # choose distributive rewrites independently of operand order
    # Preserve scope of a common positive magnitude. Distribution of a fitted
    # scale is allowed only if at most one summand lacks its own scale. All
    # products keep their fitted signs; no sign is absorbed into a placeholder.
    if distribute
        j = findfirst(x -> x.op == :add, rest)
        if j !== nothing
            terms = rest[j].args
            others = rest[setdiff(eachindex(rest), [j])]
            movable = filter(x -> !hasparam(x), others)
            held = filter(hasparam, others)
            if (!c.fitted || count(t -> !scaled(t), terms) <= 1) &&
               (!isempty(movable) || c.fitted || c.value != 1)
                expanded = add([mul([c; movable; t]) for t in terms])
                return isempty(held) ? expanded : mul([held; expanded])
            end
        end
    end
    if c.fitted || c.value != 1 || isempty(rest)
        push!(rest, c)
    end
    sort!(rest; by=key)
    length(rest) == 1 ? only(rest) : Form(:mul, rest)
end
neg(x) = mul([scalar(-1), x])

function divide(a, b)
    iszeroform(b) && throw(DomainError(0, "zero denominator in SR skeleton"))
    if isscalar(b)
        return mul([a, scalar(inv(b.value), b.fitted)])
    end
    # Move only an overall denominator sign outside; never erase internal signs.
    c, base = split_scale(b)
    if c.value < 0
        b = mul([scalar(-c.value, c.fitted), base]; distribute=false)
        a = neg(a)
    end
    Form(:div, [a,b])
end

function hill_constant(a, b)
    # Same clamp/epsilon convention as the SR sweep. Arithmetic elsewhere is
    # exact on the stored binary floats, to avoid order-dependent cancellation.
    x, n = Float64(a.value), Float64(b.value)
    isfinite(x) && isfinite(n) || throw(DomainError((x,n), "non-finite Hill input"))
    value = inv(1 + (max(x, 0.0) + 1e-6)^(-n))
    scalar(value, a.fitted || b.fitted) # rejects NaN/Inf; exact zero remains zero
end
function apply(op, a, b)
    op == 1 && return add([a,b])
    op == 2 && return add([a,neg(b)])
    op == 3 && return mul([a,b])
    op == 4 && return divide(a,b)
    op == 5 && return isscalar(a) && isscalar(b) ? hill_constant(a,b) : Form(:hill,[a,b])
    error("unknown binary operator index $op")
end
function decode(rows, term)
    term in 1:3 || throw(ArgumentError("term must be 1, 2 or 3"))
    isempty(rows) && throw(ArgumentError("empty expression tree"))
    nodes = Form[]
    parents = zeros(Int, length(rows))
    for (i, row) in enumerate(rows)
        deg, isc, val, feat, op, a, b = row
        if deg == 0
            if isc
                push!(nodes, scalar(val, true))
            else
                1 <= feat <= length(FEATURES[term]) || error("invalid feature $feat")
                push!(nodes, Form(:var, Form[], FEATURES[term][feat]))
            end
        else
            deg == 2 || error("unexpected degree $deg")
            1 <= a < i && 1 <= b < i || error("invalid post-order child index")
            parents[a] += 1; parents[b] += 1
            push!(nodes, apply(op, nodes[a], nodes[b]))
        end
    end
    all(==(1), parents[1:end-1]) && parents[end] == 0 ||
        error("expected a connected tree, not shared children or unused rows")
    nodes[end]
end
function render(x, counter)
    if isscalar(x)
        if x.fitted
            counter[] += 1
            return x.value < 0 ? "(-c$(counter[]))" : "c$(counter[])"
        end
        return number(x.value)
    elseif x.op == :var
        return x.name
    end
    a = [render(y, counter) for y in x.args]
    x.op == :hill && return "hill($(a[1]),$(a[2]))"
    sep = x.op == :add ? " + " : x.op == :mul ? "*" : "/"
    "(" * join(a, sep) * ")"
end
ispoly(x) = !(x.op in (:hill, :div)) && all(ispoly, x.args)

"""
    skeleton(sel_tree, term)::String

Sign-conditioned fitted structural key (version 2). Every emitted `cN` is
strictly POSITIVE. Fitted magnitudes are omitted; their net signs are explicit,
including inside denominators and ordered, opaque Hill arguments. Exact zero
is `0`, never a positive placeholder. No near-zero tolerance is used.

Constant-only arithmetic and scalar products collapse using fitted net values.
Like terms are collected only when their base contains no fitted parameters;
thus a*x-b*x uses the fitted net coefficient, but hill(x,a) and hill(x,b) are
not identified merely because their fitted values match. Stored floats become
exact binary rationals for arithmetic/sign/zero tests; constant-only Hill calls
use the sweep's Float64 definition. Non-finite literals/results and known zero
denominators raise DomainError. Structural numbers introduced by algebra (x+x
-> 2*x) are retained, not confused with fitted constants.

Parameter-free factors distribute. A fitted overall scale distributes only
when at most one summand lacks an independent scale, preserving a shared scale
on x+y. Coefficients are numbered AFTER sorting, independent of traversal or
hash order. Independent leaves never become tied merely by equal values.

This is NOT complete functional-family equivalence or identifiability analysis:
general rational cancellation, nonlinear parameter ranges, and singular-domain
tracking are unsupported. Zero products simplify algebraically on their common
domain. General collection across parameter-bearing atoms is deliberately
conservative. `nfree` counts remaining positive coefficient slots, not proven
identifiable dimension. Version-1 (sign-blind) keys/counts must be recomputed.
"""
skeleton(rows, term)::String = skeleton_info(rows, term).str
function skeleton_info(rows, term)
    x = decode(rows, term)
    n = Ref(0)
    s = render(x, n)
    (str=s, poly=ispoly(x), nfree=n[])
end
end
