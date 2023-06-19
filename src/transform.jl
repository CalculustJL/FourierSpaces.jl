#
function Spaces.form_transform(
                               W::FourierSpace{<:Any,D},
                               u::Union{Nothing,AbstractVecOrMat}=nothing;
                               p=nothing,
                               t::Union{Real,Nothing} = nothing,
                              ) where{D}

    # size dictionary
    #
    # sz_input : (N,K) - input size to SciMLOperator
    # sz_output: (M,K) - output size
    #
    # sz_in : (n1,...,nd) - input  size to FFT st n1*...*nd=N
    # sz_out: (m1,...,md) - output size to FFT st m1*...*md=M

    u = u isa Nothing ? points(W) |> first : u
    T = eltype(u)
    t = zero(T)

    sz_input = size(u)
    sspace = size(W)
    N = length(W)

    @assert size(u, 1) == N """size mismatch. input array must have length
        $(length(W)) in its first dimension"""
    K = size(u, 2)

    # transform input shape
    sz_in = (sspace..., K)
    U   = reshape(u, sz_in)

    # transform object
    FFTLIB = _fft_lib(u)
    ftr = FFTLIB.plan_rfft(U, 1:D)

    # transform output shape
    V    = ftr * U
    sz_out = size(V)

    # output prototype
    M = length(V) ÷ K
    sz_output = u isa AbstractMatrix ? (M, K) : (M,)
    v = reshape(V, sz_output)

    # in-place
    function fwd(v, u, p, t)
        U = reshape(u, sz_in)
        V = reshape(v, sz_out)
        mul!(V, ftr, U)

        v
    end

    function bwd(v, u, p, t)
        U = reshape(u, sz_out)
        V = reshape(v, sz_in)
        ldiv!(V, ftr, U)

        v
    end

    # out-of-place
    function fwd(u, p, t)
        U = reshape(u, sz_in)
        V = ftr * U

        reshape(V, sz_output)
    end

    function bwd(u, p, t)
        U = reshape(u, sz_out)
        V = ftr \ U

        reshape(V, sz_input)
    end

    FunctionOperator(fwd, u, v;
                     # awaiting https://github.com/SciML/OrdinaryDiffEq.jl/pull/1967
                     # then uncomment `batch` kwarg, remove SciMLOps 0.2 compat
                     batch = true,
                     op_inverse = bwd,
                     op_adjoint = bwd,
                     op_adjoint_inverse = fwd,
                     p = p)
end
#
