#
function Spaces.form_transform(
                               space::FourierSpace{<:Any,D},
                               u::Union{Nothing,AbstractVecOrMat}=nothing;
                               p=nothing,
                               t::Union{Real,Nothing}=nothing,
                              ) where{D}

    # size dictionary
    #
    # sinput : (N,K) - input size to SciMLOperator
    # soutput: (M,K) - output size
    #
    # sin : (n1,...,nd) - input  size to FFT st n1*...*nd=N
    # sout: (m1,...,md) - output size to FFT st m1*...*md=M

    u = u isa Nothing ? points(space) |> first : u
    T = eltype(u)
    t = zero(T)

    sinput = size(u)
    sspace = size(space)
    N = length(space)

    @assert size(u, 1) == N "size mismatch. input array must have length
    $(length(space)) in its first dimension"
    K = size(u, 2)

    # transform input shape
    sin = (sspace..., K)
    U   = reshape(u, sin)

    # transform object
    FFTLIB = _fft_lib(u)
    ftr = FFTLIB.plan_rfft(U, 1:D)

    # transform output shape
    V    = ftr * U
    sout = size(V)

    # output prototype
    M = length(V) ÷ K
    soutput = u isa AbstractMatrix ? (M, K) : (M,)
    v = reshape(V, soutput)

    # in-place
    function fwd(v, u, p, t)
        U = reshape(u, sin)
        V = reshape(v, sout)
        mul!(V, ftr, U)

        v
    end

    function bwd(v, u, p, t)
        U = reshape(u, sout)
        V = reshape(v, sin)
        ldiv!(V, ftr, U)

        v
    end

    # out-of-place
    function fwd(u, p, t)
        U = reshape(u, sin)
        V = ftr * U

        reshape(V, soutput)
    end

    function bwd(u, p, t)
        U = reshape(u, sout)
        V = ftr \ U

        reshape(V, sinput)
    end

    FunctionOperator(fwd, u, v;
                     op_inverse = bwd,
                     op_adjoint = bwd,
                     op_adjoint_inverse = fwd,
                     p = p)
end
#
