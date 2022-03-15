
### PART 1 ###

# Implementing Dormand-Prince Method
# 1. graph with DifferentialEquations.jl
using Plots
using OrdinaryDiffEq
using StaticArrays
using BenchmarkTools
using LinearAlgebra
function lotka_volterra_ode!(du, u, p, t)
    du[1] = (p[1] - p[2]*u[2])*u[1]
    du[2] = (-p[3]+p[4]*u[1])*u[2]
end

prob = ODEProblem(lotka_volterra_ode!, [1.0, 1.0], (0.0, 10.0), [1.5, 1.0, 3.0, 1.0])
@btime sol = solve(prob, DP5(), dense=false, adaptive=false, dt=1/4)
plot(sol)

# 2. Implement

const s = 7
const a = @SMatrix[
    1/5 0 0 0 0 0 ;
    3/40 9/40 0 0 0 0 ;
    44/45 -56/15 32/9 0 0 0 ;
    19372/6561 -25360/2187 64448/6561 -212/729 0 0 ;
    9017/3168 -355/33 46732/5247 49/176 -5103/18656 0 ;
    35/384 0 500/1113 125/192 -2187/6784 11/84]
const b = @SVector[35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
const c = @SVector[0, 1/5, 3/10, 4/5, 8/9, 1, 1]

const u0 = @SVector[1.0, 1.0]
const p = @SVector[1.5, 1.0, 3.0, 1.0]
const tspan = (0.0, 10.0)


function lotka_volterra!(du, u, p)
    du[1] = (p[1] - p[2]*u[2])*u[1]
    du[2] =  (-p[3]+p[4]*u[1])*u[2]
    return
end


function dormand_prince(f!, u0, tspan, p, dt=0.25)
    TYPE = typeof(u0[1])
    k = zeros(MMatrix{2, 6}) # store k's
    du = zeros(MVector{2}) # 
    
    time_steps = round(Int64, (tspan[2]-tspan[1]) ÷ dt+1)
    u = Array{TYPE}(undef, 2, time_steps)
    u[:, 1] = u0
    
    
    @inbounds for n in 1:time_steps-1
        # define k's
        u_n = @view(u[:, n])

        f!(@view(k[:, 1]), u_n, p)

        for s in 2:6
            mul!(du, @view(k[:, 1:s-1]), @view(a[s-1, 1:s-1]))
            f!(@view(k[:, s]), SVector(u_n + dt*du), p)
        end
        # update u
        mul!(du, k, @view(b[1:6]))
        u[:, n+1] = SVector(u_n + dt*du)
    end
    return u
end

sol = dormand_prince(lotka_volterra!, u0, tspan, p)

plot(0.0:0.25:10.0, transpose(sol))
xlabel!("t")

@btime dormand_prince(lotka_volterra!, u0, tspan, p)
