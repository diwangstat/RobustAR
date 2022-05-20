function BandedAR(ts, C, tau1, tau2, lambda_sqr, ϵ=1e-5, max_itr=100)

    p,n = size(ts);
    y = ts;
    Y = ts[:,2:end]';
    X = ts[:,1:end-1]';
    r = size(C)[2];

    B = Array{Array{Float64}}(undef,p);
    for i in 1:p
        B[i] = C[((i-1)*p+1):(i*p),:];
    end
    # truncated data (d=1)
    Σ_0_list = Array{Array{Float64,2}}(undef,p);
    for s in 1:p
        y_tr = zeros(p,n);
        for i in 1:n
            y_tr[:,i] = min(sum((B[s]'*y[:,i]).^4)^0.25,tau1)*y[:,i]/sum((B[s]'*y[:,i]).^4)^0.25;
        end
        Σ_0_list[s] = y_tr * y_tr'/n;
    end
    Ω_tr = BlockDiagonal(Σ_0_list);
    y_tr1 = sign.(y).*min.(abs.(y),tau2);
    ω_tr = vec(y_tr1[:,1:end-1]*y_tr1[:,2:end]'/(n-1));

    C_Ω = C'*Ω_tr*C; C_ω = C'*ω_tr;
    Z = [C_Ω;diagm(ones(r))];
    Z_qr = qr(Z);
    Z_qr_ls = pinv(Z_qr.R)*Z_qr.Q';

    # initialization
    θ_init = pinv(C_Ω)*C_ω;
    θ = θ_old = θ_init;
    w = zeros(r);
    d = C'*Ω_tr*C*θ - C'*ω_tr;

    ρ = 2;

    for i in 1:max_itr

        # θ-step
        θ = Z_qr_ls * [d + w/ρ - C_ω; zeros(r)];

        # d-step
        d_inter = C_Ω*θ - C_ω - w/ρ;
        d = min(norm(d_inter),lambda_sqr)*d_inter/norm(d_inter);

        # w-step
        w = w + ρ * (C_ω - C_Ω * θ + d);

        # early stopping
        if max(norm(d + C_ω - C_Ω * θ),norm(θ-θ_old))<ϵ*norm(A) break end
        θ_old = θ;

    end

    return θ
end

function banded_admm_single(Ω, ω, lambda, ϵ=1e-4, max_itr=100)

    # initialization
    r = size(ω)[1];
    θ_init = pinv(Ω)*ω;
    θ = θ_old = θ_init;
    w = zeros(r);
    d = Ω*θ-ω;
    Z = [Ω;diagm(ones(r))];

    ρ = 2;

    for i in 1:max_itr

        # θ-step
        θ = Z\[d+w/ρ-ω;zeros(r)];

        # d-step
        d_inter = Ω*θ-ω-w/ρ;
        d = min(norm(d_inter),lambda)*d_inter/norm(d_inter);

        # w-step
        w = w + ρ*(ω-Ω*θ+d);

        # early stopping
        if max(norm(d+ω-Ω*θ),norm(θ-θ_old))<ϵ*norm(A) break end
        θ_old = θ;

    end

    return θ

end

function BandedAR_LP(ts, C, tau, lambda)

    p,n = size(ts);
    y = ts;
    Y = ts[:,2:end]';
    X = ts[:,1:end-1]';
    r = size(C)[2];

    y_tr = sign.(y).*min.(abs.(y),tau);
    Σ_0 = y*y'/n;
    Σ_1 = y_tr[:,2:end]*y_tr[:,1:end-1]'/(n-1);
    Σ_0_tr = y_tr*y_tr'/n;
    Σ_1_tr = y_tr[:,2:end]*y_tr[:,1:end-1]'/(n-1);

    Ω = kron(eye(p),Σ_0_tr);
    ω = vec(Σ_1_tr');
    CtΩC = C'*Ω*C;
    Ctω = C'*ω;

    # LP solved by GLPK
    m = Model(with_optimizer(GLPK.Optimizer));

    @variable(m,θ[1:r])
    @variable(m,t)

    @objective(m, Min, t)

    @constraint(m, constraint1[j in 1:r], θ[j]-t <= 0)
    @constraint(m, constraint2[j in 1:r], θ[j]+t >= 0)
    @constraint(m, constraint3[j in 1:r], sum( CtΩC[j,i]*θ[i] for i in 1:r)-Ctω[j] + lambda >= 0)
    @constraint(m, constraint4[j in 1:r], sum( CtΩC[j,i]*θ[i] for i in 1:r)-Ctω[j] - lambda <= 0)

    JuMP.optimize!(m)
    θ_opt = zeros(r);
    for i in 1:r
        θ_opt[i] = JuMP.value(θ[i])
    end

    return θ_opt
end

function banded_robust_LP(ts, C, tau_seq, lambda_seq, ϵ=1e-4, max_itr=100)

    p,n = size(ts);
    r = size(C)[2];

    output_A = Array{Array{Float64}}(undef,length(lambda_seq)*length(tau_seq));

    for tau_ind = 1:length(tau_seq)

        # robust autocovariance estimators
        tau = tau_seq[tau_ind];



        for lambda_ind = 1:length(lambda_seq)

            lambda = lambda_seq[lambda_ind];

            # solve by LP

            output_A[((tau_ind-1)*length(lambda_seq)+lambda_ind)] = reshape(C*BandedAR_LP(ts, C, tau, lambda),p,p)';

        end

    end

    return output_A

end

function BandedAR_ols(ts, C)

    p,n = size(ts);
    y = ts;
    d = 1;
    Y = y[:,(d+1):end]';
    X = y[:,1:end-d]';

    b_ols = inv(C'*kron(diagm(ones(p)),X'*X)*C)*C'*vec(X'*Y);

    return b_ols

end

function BandedIndicator(p,k)

    ind_mat = reshape(1:p^2,p,p)';
    ind_vec = [];
    for i in 1:p
        for j in 1:p
            if abs(i-j) <= k
                append!(ind_vec,ind_mat[i,j]);
            end
        end
    end

    col = length(ind_vec);
    col_ind = 1:col;
    return sparse(ind_vec,col_ind,ones(col))

end
