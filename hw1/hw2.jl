# PART 3
using Distributed

@everywhere const r_vec = 2.9:0.001:4
@everywhere const num_attract=150
@everywhere const warmup = 400
@everywhere const start_val = 0.25

Threads.nthreads()

using BenchmarkTools
using Plots
# Create a constant u on the heap
# Define a function which takes Threads.thread() as input, and run the dynamical system
# calc_attractor_wrapper!(r) 
#   calc_attractor(, _u_cache)
# calc_attractor!(start_val, u, r, num_attract, warmup, )
# run tmap to run calc_attractor_wrapper with every possible value of r
# 

const _v_cache = [Vector{typeof(start_val)}(undef, num_attract) for i in 1:1101]

function calc_attractor!(u, r=2.9, start_val = start_val, num_attract=num_attract, warmup=warmup)
    r0 = r
    u0 = start_val
    for i in 1:warmup+1
        u0 = r0*u0*(1-u0)
    end
    u[1] = r0*u0*(1-u0)
    @inbounds for i in 1:num_attract - 1
        u[i+1] = r * u[i] * (1 - u[i])
    end
end

function calc_attractor_wrapper(i)
    calc_attractor!(_v_cache[i], r_vec[i])
end

function tmap(f, p)
    Threads.@threads for i in 1:length(p)
        f(i)
    end
end

using SplitApplyCombine
tmap(calc_attractor_wrapper, 1:1101)
x = plot(2.9:0.001:4, invert(_v_cache), legend=false)
savefig(x, "p2part3.png")

@btime tmap(calc_attractor_wrapper, 1:1101) # parallel
@btime map(calc_attractor_wrapper, 1:1101) # sequential

# PART 4

println(workers())

if nworkers()==1
  addprocs(5)  # Unlike threads you can addprocs in the middle of a julia session
  println(workers())
end

@everywhere function dist_calc_attractor(r=2.9, start_val = start_val, num_attract=num_attract, warmup=warmup)
    u = Vector{typeof(start_val)}(undef, num_attract)
    r0 = r
    u0 = start_val
    for i in 1:warmup+1
        u0 = r0*u0*(1-u0)
    end
    u[1] = r0*u0*(1-u0)
    @inbounds for i in 1:num_attract - 1
        u[i+1] = r * u[i] * (1 - u[i])
    end
    return u
end

res = @sync @distributed hcat for r in r_vec
    dist_calc_attractor(r)
end 

x = plot(2.9:0.001:4, transpose(res), legend=false)
@btime @sync @distributed hcat for r in r_vec
    dist_calc_attractor(r)
end 

# PART 5

@everywhere using SharedArrays

procs()
u_shared = SharedMatrix{typeof(start_val)}(length(r_vec), num_attract, pids = procs())
@everywhere u_shared = $u_shared
@everywhere function pmap_calc_attractor(ind, start_val = start_val, num_attract=num_attract, warmup=warmup)
    r0 = r_vec[ind]
    u0 = start_val
    for i in 1:warmup+1
        u0 = r0*u0*(1-u0)
    end
    u_shared[ind, 1] = r0*u0*(1-u0)
    @inbounds for i in 1:num_attract - 1
        u_shared[ind, i+1] = r0 * u_shared[ind, i] * (1 - u_shared[ind, i])
    end
end

pmap(pmap_calc_attractor, 1:1101)
x = plot(2.9:0.001:4, u_shared, legend=false)
@btime pmap(pmap_calc_attractor, 1:1101)

# PART 5

# The threaded method is the fastest at around 257 us. Then single threaded at 1.408 ms, 
# distributed at 4.750 ms, and pmap at 113 ms
# Threaded is the fastest because it can take advantage of multiple cores, without spending
# overhead on allocating a whole new process, with its associate heap and data sections. 
# With 6 threads, the threaded method is approximately 6 times faster than single threaded.