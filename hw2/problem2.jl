using MPI

using BenchmarkTools

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nproc = MPI.Comm_size(comm)

# print("Hello world: $(rank) of $(nproc)\n")



const N = 25
const num_iters = 100

function pass_message(my_send_msg, my_recv_buf, i, n)
    src = rank
    dst = mod(rank+1, 2)
    
    unique_flag = 2*N*i + 2*n

    sreq = MPI.Isend(my_send_msg, dst, unique_flag + src, comm)
    rreq = MPI.Irecv!(my_recv_buf, dst, unique_flag + dst, comm)
    MPI.Waitall!([sreq, rreq])
end


# send_msg = Array{Array{Int8, 1}}(undef, 2)

x = Array{Float64, 1}(undef, N)
for n in 1:N
    my_send_msg = Array{Int8, 1}(undef, 2^n)
    fill!(my_send_msg, rank)
    my_recv_buf = Array{Int8, 1}(undef, 2^n)

    min_time = Inf
    pass_message(my_send_msg, my_recv_buf, num_iters, n)
    for i in 1:num_iters
        min_time = min(min_time, @elapsed pass_message(my_send_msg, my_recv_buf, i, n))
    end
    x[n] = min_time
end

using Plots
if rank == 0
    sizes = [2^n for n in 1:N]
    println(sizes ./x)
    fn = plot(1:N, log.(10, sizes./x), title="log10(Bandwidth) vs log2(Message Size)", xlabel="log2(Message Size) Floats", ylabel="log10(Bandwidth) Floats/s")
    savefig(fn,"bandwidth.png")

    fn = plot(1:N, log.(10,  x), title="log10(Latency) vs log2(Message Size)", xlabel="log2(Message Size) Floats", ylabel="log10(Latency) s")
    savefig(fn,"latency.png")
    p = 1.5948380514723215e11 # peakflops()
    fn = plot(1:N, log.(10, x ./ sizes *p), title="log10(Peak Flops/Bandwidth) vs log2(Message Size)", xlabel="log2(Message Size) Floats", ylabel="log10(Peak Flops/Bandwidth) Flops/(Floats/s)")
    savefig(fn,"inverse_bandwidth.png")
end
