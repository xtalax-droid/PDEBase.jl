# Problem definitions and the worker body for mol_regression.jl (included only inside worker
# processes, after the packages are loaded, because the macros must exist when this is parsed).
const SAMPLES = 10

# Each problem returns (pdesys, disc, discretize_fn, solve_fn, final_fn); the definitions are
# copied from MethodOfLines test/ (file noted per problem).
function build(problem)
    ode_discretize(pdesys, disc) = begin
        sys, tspan = symbolic_discretize(pdesys, disc)
        ODEProblem(mtkcompile(sys), nothing, tspan)
    end
    if problem == "diffusion1d" # test/Diffusion/MOL_1D_Linear_Diffusion.jl, Test 00
        @parameters t x
        @variables u(..)
        Dt = Differential(t); Dxx = Differential(x)^2
        eq = Dt(u(t, x)) ~ Dxx(u(t, x))
        bcs = [u(0, x) ~ cos(x), u(t, 0) ~ exp(-t), u(t, Float64(π)) ~ -exp(-t)]
        domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, Float64(π))]
        @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
        dx_ = Float64(π) / 29
        disc = MOLFiniteDifference([x => dx_], t)
        return (pdesys, disc, discretize, p -> solve(p; saveat = 0.1), s -> s[u(t, x)][end, :])
    elseif problem == "diffusion2d" # test/2D_Diffusion/MOL_2D_Diffusion.jl, Test 00
        @parameters t x y
        @variables u(..)
        Dxx = Differential(x)^2; Dyy = Differential(y)^2; Dt = Differential(t)
        f(t, x, y) = exp(x + y) * cos(x + y + 4t)
        eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))
        bcs = [
            u(0.0, x, y) ~ f(0.0, x, y), u(t, 0.0, y) ~ f(t, 0.0, y), u(t, 2.0, y) ~ f(t, 2.0, y),
            u(t, x, 0.0) ~ f(t, x, 0.0), u(t, x, 2.0) ~ f(t, x, 2.0),
        ]
        domains = [t ∈ Interval(0.0, 2.0), x ∈ Interval(0.0, 2.0), y ∈ Interval(0.0, 2.0)]
        @named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])
        disc = MOLFiniteDifference([x => 0.1, y => 0.2], t; approx_order = 4)
        return (pdesys, disc, discretize, p -> solve(p), s -> s[u(t, x, y)][end, :, :])
    elseif problem == "robin1d" # test/Diffusion/MOL_1D_Linear_Diffusion.jl, Test 05 (Robin, order 4)
        @parameters t x
        @variables u(..)
        Dt = Differential(t); Dx = Differential(x); Dxx = Differential(x)^2
        eq = Dt(u(t, x)) ~ Dxx(u(t, x))
        bcs = [
            u(0, x) ~ sin(x),
            u(t, -1.0) + 3Dx(u(t, -1.0)) ~ exp(-t) * (sin(-1.0) + 3cos(-1.0)),
            4u(t, 1.0) + Dx(u(t, 1.0)) ~ exp(-t) * (4sin(1.0) + cos(1.0)),
        ]
        domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(-1.0, 1.0)]
        @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
        disc = MOLFiniteDifference([x => 0.01], t; approx_order = 4)
        return (pdesys, disc, discretize, p -> solve(p; saveat = 0.1), s -> s[u(t, x)][end, :])
    elseif problem == "periodic_convection" # test/Convection/MOL_1D_Linear_Convection.jl, Test 00
        @parameters t x
        @variables u(..)
        Dt = Differential(t); Dx = Differential(x)
        eq = Dt(u(t, x)) ~ -Dx(u(t, x))
        asf(x) = (0.5 / (0.2 * sqrt(2.0 * 3.1415))) * exp(-(x - 1.0)^2 / (2.0 * 0.2^2))
        bcs = [u(0, x) ~ asf(x), u(t, 0) ~ u(t, 2)]
        domains = [t ∈ Interval(0.0, 2.0), x ∈ Interval(0.0, 2.0)]
        @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
        disc = MOLFiniteDifference([x => 2 / 80], t)
        return (pdesys, disc, ode_discretize, p -> solve(p, Euler(), dt = 0.025, saveat = 0.1),
            s -> s[u(t, x)][end, :])
    elseif problem == "burgers_upwind" # test/Burgers/burgers_eq.jl, first testset
        @parameters x t
        @variables u(..)
        Dx = Differential(x); Dt = Differential(t)
        analytic_u(t, x) = x / (t + 1)
        eq = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x))
        bcs = [u(0, x) ~ x, u(t, 0.0) ~ analytic_u(t, 0.0), u(t, 1.0) ~ analytic_u(t, 1.0)]
        domains = [t ∈ Interval(0.0, 6.0), x ∈ Interval(0.0, 1.0)]
        @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
        disc = MOLFiniteDifference([x => 0.05], t, advection_scheme = UpwindScheme())
        return (pdesys, disc, discretize, p -> solve(p), s -> s[u(t, x)][end, :])
    elseif problem == "brusselator" # test/Brusselator/brusselator_eq.jl, Test 01 (periodic, nonlinear, 2 vars)
        @parameters x y t
        @variables u(..) v(..)
        Difft = Differential(t); Dxx = Differential(x)^2; Dyy = Differential(y)^2
        ∇²(w) = Dxx(w) + Dyy(w)
        brusselator_f(x, y, t) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.0
        α = 10.0
        u0(x, y, t) = 22(y * (1 - y))^(3 / 2)
        v0(x, y, t) = 27(x * (1 - x))^(3 / 2)
        eq = [
            Difft(u(x, y, t)) ~ 1.0 + v(x, y, t) * u(x, y, t)^2 - 4.4 * u(x, y, t) +
                α * ∇²(u(x, y, t)) + brusselator_f(x, y, t),
            Difft(v(x, y, t)) ~ 3.4 * u(x, y, t) - v(x, y, t) * u(x, y, t)^2 + α * ∇²(v(x, y, t)),
        ]
        domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0), t ∈ Interval(0.0, 11.5)]
        bcs = [
            u(x, y, 0) ~ u0(x, y, 0), u(0, y, t) ~ u(1, y, t), u(x, 0, t) ~ u(x, 1, t),
            v(x, y, 0) ~ v0(x, y, 0), v(0, y, t) ~ v(1, y, t), v(x, 0, t) ~ v(x, 1, t),
        ]
        @named pdesys = PDESystem(eq, bcs, domains, [x, y, t], [u(x, y, t), v(x, y, t)])
        disc = MOLFiniteDifference([x => 1 / 32, y => 1 / 32], t; approx_order = 2)
        return (pdesys, disc, ode_discretize, p -> solve(p, TRBDF2(), saveat = 0.01),
            s -> cat(s[u(x, y, t)][:, :, end], s[v(x, y, t)][:, :, end]; dims = 3))
    elseif problem == "schroedinger" # test/Complex/schroedinger.jl, first testset
        @parameters t, x
        @variables ψ(..)
        Dt = Differential(t); Dxx = Differential(x)^2
        V(x) = 0.0
        eq = [im * Dt(ψ(t, x)) ~ (Dxx(ψ(t, x)) + V(x) * ψ(t, x))]
        ψ0 = x -> ((1 + im) / sqrt(2)) * sinpi(2 * x)
        bcs = [ψ(0, x) => ψ0(x), ψ(t, 0) ~ 0, ψ(t, 1) ~ 0]
        domains = [t ∈ Interval(0, 1), x ∈ Interval(0, 1)]
        @named sys = PDESystem(eq, bcs, domains, [t, x], [ψ(t, x)])
        disc = MOLFiniteDifference([x => 100], t)
        return (sys, disc, discretize,
            p -> solve(p; saveat = 0.01, initializealg = BrownFullBasicInit(), reltol = 1.0e-10, abstol = 1.0e-10),
            s -> s[ψ(t, x)][end, :])
    elseif problem == "schroedinger_complex_bcs" # test/Complex/schroedinger.jl, second testset
        @parameters t, x
        @variables ψ(..)
        Dt = Differential(t); Dxx = Differential(x)^2
        ϵ = 1.0e-2
        V2(x) = 0.0
        eq = [(im * ϵ) * Dt(ψ(t, x)) ~ (-0.5 * ϵ^2)Dxx(ψ(t, x)) + V2(x) * ψ(t, x)]
        ψ0 = x -> exp((im / ϵ) * 1.0e-1 * sum(x))
        bcs = [
            ψ(0, x) => ψ0(x),
            ψ(t, 0) ~ exp((im / ϵ) * (1.0e-1 * sum(0) - 0.5e-2 * t)),
            ψ(t, 1) ~ exp((im / ϵ) * (1.0e-1 * sum(1) - 0.5e-2 * t)),
        ]
        domains = [t ∈ Interval(0, 1), x ∈ Interval(0, 1)]
        @named sys = PDESystem(eq, bcs, domains, [t, x], [ψ(t, x)])
        disc = MOLFiniteDifference([x => 300], t)
        return (sys, disc, discretize, p -> solve(p; saveat = 0.01, initializealg = BrownFullBasicInit()),
            s -> s[ψ(t, x)][end, :])
    end
    error("unknown problem $problem")
end

function worker_body(problem, mode, resultfile, finalfile)
    pdesys, disc, disc_fn, solve_fn, final_fn = build(problem)
    res = Dict{String, Any}()
    res["first_discretize_s"] = @elapsed prob = disc_fn(pdesys, disc)
    if mode == "warm"
        b = @benchmark $disc_fn($pdesys, $disc) samples = SAMPLES evals = 1 seconds = 1.0e9
        res["discretize_s"] = b.times ./ 1.0e9
        res["discretize_allocs"] = b.allocs
        res["discretize_bytes"] = b.memory
        res["first_solve_s"] = @elapsed sol = solve_fn(prob)
        res["solve_s"] = [@elapsed(solve_fn(prob)) for _ in 1:SAMPLES]
        serialize(finalfile, Array(final_fn(sol)))
        res["retcode"] = string(sol.retcode)
    end
    serialize(resultfile, res)
    return nothing
end

