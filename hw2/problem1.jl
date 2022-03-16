
### PART 1 ### Implementing Dormand-Prince Method


# Graph Lotka with DifferentialEquations.jl
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

# Implement Dormand Prince

const s = 7
const a = @SMatrix[
    1/5 0 0 0 0 0 ;
    3/40 9/40 0 0 0 0 ;
    44/45 -56/15 32/9 0 0 0 ;
    19372/6561 -25360/2187 64448/6561 -212/729 0 0 ;
    9017/3168 -355/33 46732/5247 49/176 -5103/18656 0 ;
    35/384 0 500/1113 125/192 -2187/6784 11/84] # Store as a 2D array for easier indexing
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


function run_dormand(f!, du, k, u, u0, tspan, p, dt=0.25)
    u[:, 1] .= u0
    time_steps = round(Int64, (tspan[2]-tspan[1]) ÷ dt+1)

    @inbounds for t in 1:time_steps-1
        # define k's
        u_t = @view(u[:, t])

        f!(@view(k[:, 1]), u_t, p)

        for s in 2:6
            mul!(du, @view(k[:, 1:s-1]), @view(a[s-1, 1:s-1]))
            f!(@view(k[:, s]), SVector(u_t + dt *du), p)
        end
        # update u
        mul!(du, k, @view(b[1:6]))
        u[:, t+1] .= SVector(u_t + dt*du)
    end
end

dt = 0.25
time_steps = round(Int64, (tspan[2]-tspan[1]) ÷ dt+1)

du = zero(MVector{2, Float64}) # 
k = zero(MMatrix{2, 6, Float64}) # store k's
sol = Array{Float64}(undef, 2, time_steps)
@btime run_dormand(lotka_volterra!, du, k, sol, u0, tspan, p)

plot(0.0:0.25:10.0, transpose(sol))
xlabel!("t")

### PART 2 ### Propagating Sensitivity

# edit lotka to include du/dp in u
# change dormand prince to have larger k, du, u0 matrices
# init u0 with du/dp = 0


function lotka_volterra_sensitivity!(du, u, p)
    du[1] = (p[1] - p[2]*u[2])*u[1]
    du[2] =  (-p[3]+p[4]*u[1])*u[2]

    f_u = @SMatrix [
        (p[1]-p[2]*u[2]) (-p[2]*u[1]);
        (p[4]*u[2]) (-p[3]+p[4]*u[1])
    ]
    x_p = @view(u[3:6])
    y_p = @view(u[7:10])
    f_p =  @SArray [
        u[1], -u[1]*u[2], 0, 0, 0, 0, -u[2], u[1]*u[2]
    ]
    x_pt = @view(du[3:6])
    y_pt = @view(du[7:10])

    # f_pt = df/du * du/dp, matrix multiplication written out
    x_pt .= f_u[1, 1].*x_p .+ f_u[1, 2].*y_p
    y_pt .=  f_u[2, 1].*x_p .+ f_u[2, 2].*y_p
    @view(du[3:10]) .+= f_p
end

const n_s = 10
const u0_s =  @SVector[1.0, 1.0, 0, 0, 0, 0, 0 ,0, 0, 0]
du = zeros(MVector{10}) # 
k = zeros(MMatrix{10, 6}) # store k's
sol = Array{typeof(u0_s[1])}(undef, 10, time_steps)
@btime run_dormand(lotka_volterra_sensitivity!, du, k, sol, u0_s, tspan, p)
plot(transpose(sol[1:2, :]))
plot(transpose(sol[3:10, :]))


### PART 3 ### Using gradient descent to retrieve the true parameters

p0 = @SVector[1.2, 0.8, 2.8, 0.8]

# define L2 loss function 
# define dL/dp = mean(dL/du * du/dp), 
    # du/dp can be calculated from p_hat using dormand_prince_sensitivity()
    # dL/du = mean(2[x_hat - x, y_hat - y])
# update p -= 0.01 dL/dp




function grad_descent!(p_hat, data, p0, epochs, lr = 1e-5)
    x = @view(data[1, :])
    y = @view(data[2, :])
    dt = 0.25
    time_steps = round(Int64, (tspan[2]-tspan[1]) ÷ dt+1)
    dlx_dp = @MVector[0., 0., 0., 0.]
    dly_dp = @MVector[0., 0., 0., 0.]
    du = zeros(MVector{10}) # 
    k = zeros(MMatrix{10, 6}) # store k's
    sol_hat = Array{Float64}(undef, 10, time_steps)
    p_hat .= p0

    dl_dx = zeros(MVector{time_steps})
    dl_dy = zeros(MVector{time_steps})
    for epoch in 1:epochs
        run_dormand(lotka_volterra_sensitivity!, du, k, sol_hat, u0_s, tspan, p_hat)
        x_hat = @view(sol_hat[1, :])
        y_hat = @view(sol_hat[2, :])

        dl_dx .= 2 .* (x_hat .- x)
        mul!(dlx_dp, @view(sol_hat[3:6, :]), dl_dx)
        dl_dy .= 2 .* (y_hat .- y)
        mul!(dly_dp, @view(sol_hat[7:10, :]), dl_dy)

        if mod(epoch, round(epochs/20)) == 0
            loss = sum((x_hat .- x).^2 + (y_hat .- y).^2)/time_steps
            println("epoch: ", epoch, ", loss: ", loss)
        end

        p_hat .-= lr *(dlx_dp .+ dly_dp)/time_steps
    end
    println("###")
    println("found params: ", p_hat)
    println("###")
end



p_hat = @MVector[0., 0., 0., 0.]
@time grad_descent!(p_hat, sol, p0, 200000)
println("True Params: ", p)