
using Distributions
using BenchmarkTools

# Problem 2

function my_quantile(d, y, x0 = mean(d))
    for i in 1:10
        x0 = x0 - 1/pdf.(d, x0)*(cdf.(d, x0) - y)
    end
    return x0
end


@btime my_quantile(Normal(0, 1), 0.5)
@btime my_quantile(Gamma(5, 1), 0.5)
@btime my_quantile(Beta(2, 4), 0.5)

@btime quantile.(Normal(0, 1), 0.5)
@btime quantile.(Gamma(5, 1), 0.5)
@btime quantile.(Beta(2, 4), 0.5)

function calc_attractor!(out, r, num_attract=150, warmup=400)
    samples = Vector{typeof(out)}(undef, num_attract)
    for i in 1:warmup
        out = r*out*(1-out)
    end
    samples[1] = r*out*(1-out)
    for i in 1:num_attract-1
        samples[i+1] = r*samples[i]*(1-samples[i])
    end
    return samples
end

calc_attractor!(0.25, 2.9)


# Problem 3


function calc_attractor_matrix!(start_val = 0.25, r=2.9:0.001:4, num_attract=150, warmup=400)
    u = Matrix{typeof(start_val)}(undef, length(r), num_attract)

    for j in 1:length(r)
        r0 = r[j]
        u0 = start_val
        for i in 1:warmup+1
            u0 = r0*u0*(1-u0)
        end
        u[j, 1] = r0*u0*(1-u0)
    end
    for i in 1:num_attract - 1
        u[:, i+1] = r .* @view(u[:, i]) .* (1 .- @view(u[:, i]))
    end
    return u
end

v = calc_attractor_matrix!()

using Plots
x = plot(2.9:0.001:4, v, legend=false)
savefig(x, "plot_2.png")