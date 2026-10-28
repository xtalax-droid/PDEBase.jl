# MethodOfLines regression benchmark for PDEBase: compares two PDEBase checkouts under the
# same MethodOfLines on problems taken from MethodOfLines' own test suite.
#
# Setup (once per side), with MOL a MethodOfLines checkout used unchanged by both sides:
#
#   julia -e 'using Pkg; Pkg.activate("envA"); Pkg.develop(path="MOL"); Pkg.develop(path="PDEBase_A");
#             Pkg.add(["OrdinaryDiffEq","OrdinaryDiffEqSDIRK","OrdinaryDiffEqLowOrderRK",
#                      "DomainSets","ModelingToolkit","DiffEqBase","BenchmarkTools","SciMLBase"])'
#   cp -r envA envB   # then point the PDEBase entry of envB/Manifest.toml at PDEBase_B
#                     # (change `path = ...`), so that only PDEBase differs.
#
# Run (drives child Julia processes, interleaved A,B,A,B... so machine load hits both sides):
#
#   MOL_BENCH_JULIA="julia"  julia benchmark/mol_regression.jl envA envB OUTDIR [fresh_reps=3] [warm_runs=3] [problems,comma,sep]
#   MOL_BENCH_JULIA="forge julia"  ...   # on the forge box, to respect the memory cap
#
# Per problem and side it records: first-discretize time in a fresh process (fresh_reps runs),
# warm discretize time and allocations and warm solve time (warm_runs processes x 10 samples,
# pooled), the final-state array (OUTDIR/out/<side>_<problem>.jls), the machine load average and
# available memory at every run, and then prints a markdown table plus the max absolute
# difference between the two sides' final states. Spread is shown as min..max over all runs.
#
# Problem definitions live in mol_problems.jl.
# Internal modes (called by the driver): `worker ENV_NAME PROBLEM fresh|warm RESULTFILE FINALFILE`.

using Serialization, Statistics, Printf

const PROBLEMS = [
    "diffusion1d", "diffusion2d", "robin1d", "periodic_convection",
    "burgers_upwind", "brusselator", "schroedinger", "schroedinger_complex_bcs",
]
# ---------------------------------------------------------------- worker side
function worker(problem, mode, resultfile, finalfile)
    Core.eval(Main, :(using ModelingToolkit, MethodOfLines, DomainSets, SciMLBase, OrdinaryDiffEq,
        BenchmarkTools))
    Core.eval(Main, :(using OrdinaryDiffEqSDIRK: TRBDF2))
    Core.eval(Main, :(using OrdinaryDiffEqLowOrderRK: Euler))
    Core.eval(Main, :(using DiffEqBase: BrownFullBasicInit))
    Core.eval(Main, :(using ModelingToolkit: Differential))
    Base.include(Main, joinpath(@__DIR__, "mol_problems.jl"))
    Base.invokelatest(Main.worker_body, problem, mode, resultfile, finalfile)
end

# ---------------------------------------------------------------- driver side
loadavg() = split(read("/proc/loadavg", String))[1:3]
function memavail_gb()
    for l in eachline("/proc/meminfo")
        startswith(l, "MemAvailable:") && return parse(Int, split(l)[2]) / 1024^2
    end
    return NaN
end

function run_worker(julia_cmd, env, problem, mode, resultfile, finalfile)
    cmd = `$(julia_cmd) --project=$env --threads=1 $(@__FILE__) worker $env $problem $mode $resultfile $finalfile`
    load = loadavg(); mem = memavail_gb()
    ok = open(io -> success(pipeline(cmd; stdout = io, stderr = io)), resultfile * ".log", "w")
    ok || error("worker failed: $cmd (see $(resultfile).log)")
    res = deserialize(resultfile)
    res["load"] = load; res["mem_avail_gb"] = mem
    return res
end

fmt(x) = x < 1 ? @sprintf("%.1f ms", 1000x) : @sprintf("%.2f s", x)
spread(v) = "$(fmt(median(v))) [$(fmt(minimum(v)))..$(fmt(maximum(v)))]"
# A ratio is only called when the two sides' whole ranges are separated by more than 10%.
function verdict(a, b)
    maximum(b) < 0.9 * minimum(a) && return "faster"
    minimum(b) > 1.1 * maximum(a) && return "REGRESSION"
    maximum(a) < 0.9 * minimum(b) && return "REGRESSION"
    (median(b) / median(a) > 1.1 || median(b) / median(a) < 0.9) && return "INCONCLUSIVE"
    return "ok"
end

function driver(envs, outdir, fresh_reps, warm_runs, problems, julia_cmd)
    mkpath(joinpath(outdir, "out"))
    sides = ["A", "B"]
    data = Dict((s, p) => Dict{String, Any}("fresh" => Float64[], "disc" => Float64[], "solve" => Float64[],
        "allocs" => Int[], "bytes" => Int[], "loads" => String[]) for s in sides for p in problems)
    for p in problems, rep in 1:max(fresh_reps, warm_runs)
        order = isodd(rep) ? sides : reverse(sides) # alternate who goes first
        for (mode, n) in (("fresh", fresh_reps), ("warm", warm_runs))
            rep <= n || continue
            for s in order
                env = envs[s]
                rf = joinpath(outdir, "$(s)_$(p)_$(mode)_$rep.res")
                ff = joinpath(outdir, "out", "$(s)_$(p).jls")
                res = run_worker(julia_cmd, env, p, mode, rf, ff)
                d = data[(s, p)]
                push!(d["fresh"], res["first_discretize_s"])
                mode == "warm" && (append!(d["disc"], res["discretize_s"]); append!(d["solve"], res["solve_s"]);
                    push!(d["allocs"], res["discretize_allocs"]); push!(d["bytes"], res["discretize_bytes"]))
                push!(d["loads"], "$(p) $(s) $(mode)#$rep load=$(join(res["load"], '/')) memavail=$(round(res["mem_avail_gb"]; digits = 1))GB")
                println(d["loads"][end], " first_disc=", fmt(res["first_discretize_s"]))
                flush(stdout)
            end
        end
    end
    println("\n| problem | metric | A (median [min..max]) | B (median [min..max]) | B/A median | verdict |\n|---|---|---|---|---|---|")
    for p in problems
        a = data[("A", p)]; b = data[("B", p)]
        for (name, key) in (("first discretize (fresh)", "fresh"), ("warm discretize", "disc"), ("warm solve", "solve"))
            ratio = median(b[key]) / median(a[key])
            println("| $p | $name | $(spread(a[key])) | $(spread(b[key])) | $(@sprintf("%.2f", ratio)) | $(verdict(a[key], b[key])) |")
        end
        println("| $p | discretize allocs / MiB | $(a["allocs"][1]) / $(round(a["bytes"][1] / 2^20; digits = 2)) | $(b["allocs"][1]) / $(round(b["bytes"][1] / 2^20; digits = 2)) | $(@sprintf("%.3f", b["bytes"][1] / a["bytes"][1])) | |")
    end
    println("\n| problem | max abs diff A vs B at final time | max abs of A |\n|---|---|---|")
    for p in problems
        xa = deserialize(joinpath(outdir, "out", "A_$(p).jls")); xb = deserialize(joinpath(outdir, "out", "B_$(p).jls"))
        println("| $p | $(size(xa) == size(xb) ? maximum(abs.(xa .- xb)) : "SIZE MISMATCH") | $(maximum(abs.(xa))) |")
    end
    println("\nPer-run load:"); foreach(k -> foreach(println, data[k]["loads"]), sort(collect(keys(data))))
    serialize(joinpath(outdir, "summary.jls"), data)
end

if !isempty(ARGS) && ARGS[1] == "worker"
    worker(ARGS[3], ARGS[4], ARGS[5], ARGS[6])
elseif abspath(PROGRAM_FILE) == @__FILE__
    envA, envB, outdir = abspath.(ARGS[1:3])
    fresh_reps = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 3
    warm_runs = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 3
    problems = length(ARGS) >= 6 ? String.(split(ARGS[6], ',')) : PROBLEMS
    driver(Dict("A" => envA, "B" => envB), outdir, fresh_reps, warm_runs, problems,
        Cmd(String.(split(get(ENV, "MOL_BENCH_JULIA", "julia")))))
end
