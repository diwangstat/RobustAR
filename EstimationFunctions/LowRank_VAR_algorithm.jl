# This file contains the estimation algorithm for sparse VAR models

# Nuclear-norm-regularized Yule-Walker estimator with truncation parameter τ and regularization parameter λ

# Method 1
# Linearized ADMM algorithm
# Input: truncated autocovariances

function lowrank_admm_single(Σ_0_tr, Σ_1_tr, p, d, lambda, ϵ=1e-4, max_itr=500)

    # initialization
    A_init = Σ_1_tr*pinv(Σ_0_tr);
    A = A_old = A_init;
    W = reshape(zeros(p*p*d),p,p*d);
    D = A * Σ_0_tr - Σ_1_tr;

    γ = 1; μ = 2*maximum(eigen(Σ_0_tr^2).values);

    for i in 1:max_itr

        # A-step
        interm_svd = svd(A.-(2/μ)*(A*Σ_0_tr-Σ_1_tr-D-W/γ)*Σ_0_tr);
        S = ((interm_svd.S.-2/(γ*μ)).>0).*(interm_svd.S.-2/(γ*μ));
        A = interm_svd.U * diagm(S) * interm_svd.Vt;

        # D-step
        interm_svd2 = svd(A*Σ_0_tr-Σ_1_tr-W/γ);
        D = interm_svd2.U * diagm(min.(interm_svd2.S,lambda)) * interm_svd2.Vt;

        # W-step
        W = W + γ*(Σ_1_tr - A*Σ_0_tr + D);

        # early stopping
        if max(norm(D-Σ_1_tr+A*Σ_0_tr),norm(A-A_old))<ϵ*norm(A) break end
        A_old = A;

    end

    return A

end

# ADMM algorithm for a sequence of lambda with warm start

function lowrank_admm_seq(Σ_0_tr, Σ_1_tr, p, d, lambda_seq, ϵ=1e-4, max_itr=500)

    A_init = Σ_1_tr*pinv(Σ_0_tr);
    A = A_old = A_init;
    W = reshape(zeros(p*p*d),p,p*d);
    D = A * Σ_0_tr - Σ_1_tr;

    γ = 1; μ = 2*maximum(eigen(Σ_0_tr^2).values);

    output_A = Array{Array{Float64}}(undef,length(lambda_seq));

    for lambda_ind in 1:length(lambda_seq)

        lambda = lambda_seq[lambda_ind];

        for i in 1:max_itr

            # A-step
            interm_svd = SVD{Float64}
            try
                interm_svd=svd(A.-(2/μ)*(A*Σ_0_tr-Σ_1_tr-D-W/γ)*Σ_0_tr)
            catch e
                interm_svd=svd(A.-(2/μ)*(A*Σ_0_tr-Σ_1_tr-D-W/γ)*Σ_0_tr,alg=LinearAlgebra.QRIteration())
            end
            #interm_svd = svd(A.-(2/μ)*(A*Σ_0_tr-Σ_1_tr-D-W/γ)*Σ_0_tr);
            S = ((interm_svd.S.-2/(γ*μ)).>0).*(interm_svd.S.-2/(γ*μ));
            A = interm_svd.U * diagm(S) * interm_svd.Vt;

            # D-step
            interm_svd2 = SVD{Float64}
            try
                interm_svd2=svd(A*Σ_0_tr-Σ_1_tr-W/γ)
            catch e
                interm_svd2=svd(A*Σ_0_tr-Σ_1_tr-W/γ,alg=LinearAlgebra.QRIteration())
            end
            #interm_svd2 = svd(A*Σ_0_tr-Σ_1_tr-W/γ);
            D = interm_svd2.U * diagm(min.(interm_svd2.S,lambda)) * interm_svd2.Vt;

            # W-step
            W = W + γ*(Σ_1_tr - A*Σ_0_tr + D);

            # early stopping
            if max(norm(D-Σ_1_tr+A*Σ_0_tr),norm(A-A_old))<ϵ*norm(A) break end
            A_old = A;

        end

        output_A[lambda_ind] = A;

    end

    return output_A
end

function lowrank_robust_admm(ts, lag_order, tau_seq, lambda_seq, ϵ=1e-4, max_itr=500)

    p,n = size(ts);
    d = lag_order;

    output_A = Array{Array{Float64}}(undef,length(lambda_seq)*length(tau_seq));

    for tau_ind = 1:length(tau_seq)

        tau = tau_seq[tau_ind];
        # data truncation
        ts_tr = zeros(p,n);
        for i in 1:n
            ts_tr[:,i] = min(sum(ts[:,i].^2)^0.5,tau)*ts[:,i]/sum(ts[:,i].^2)^0.5;
        end
        if d == 1
            Σ_0_tr = ts_tr*ts_tr'/n;
            Σ_1_tr = ts_tr[:,2:end]*ts_tr[:,1:end-1]'/(n-1);
        else
            X_tr = ts_tr[:,1:end-d];
            for s in 2:d
                X_tr = [X_tr; ts_tr[:,s:end-d+s-1]];
            end
            Σ_0_tr = X_tr*X_tr'/(n-d);
            Σ_1_tr = ts_tr[:,(d+1):end]*X_tr'/(n-d);
        end

        output_A[((tau_ind-1)*length(lambda_seq)+1):tau_ind*length(lambda_seq)] = lowrank_admm_seq(Σ_0_tr, Σ_1_tr, p, d, lambda_seq, ϵ, max_itr);

    end

    return output_A

end


function LR_VAR_SDP_single(Σ_1, Σ_0, lambda, p)

    # solver = SCS.Optimizer(linear_solver = SCS.GpuIndirectSolver)
    m = Model(SCS.Optimizer);
    # m = Model(with_optimizer(CSDP.Optimizer));
    # m = Model(optimizer_with_attributes(Mosek.Optimizer));
    set_silent(m)

    @variable(m, W1[1:p, 1:p], PSD)
    @variable(m, W2[1:p, 1:p], PSD)
    @variable(m, A[1:p, 1:p])

    @objective(m, Min, tr(W1)+tr(W2))

    @SDconstraint(m, [W1 A; A' W2] >= zeros(2*p,2*p))
    @SDconstraint(m, [lambda*eye(p) Σ_1-A*Σ_0; (Σ_1-A*Σ_0)' lambda*eye(p)] >= zeros(2*p,2*p))

    JuMP.optimize!(m)
    A_opt = zeros(p,p);
    for i in 1:p
        for j in 1:p
            A_opt[i,j] = JuMP.value(A[i,j])
        end
    end

    return A_opt

end


function NN_VAR(ts,lag_order,lambda_seq,ϵ=1e-5,max_itr=500)

    K,T = size(ts); P = lag_order;
    Y = ts[:,(P+1):T]'; global X = ts[:,P:(T-1)]';
    for s = 2:P
        X = [X';ts[:,(P-s+1):(T-s)]]';
    end
    n = T-P;

    # OLS initalization
    A_init = (X\Y)*1;

    A = A_old = W = A_init;
    C = reshape(zeros(K^2*P),K*P,K);
    rho = 1;

    output_A = Array{Array{Float64}}(undef,length(lambda_seq));

    for i_lambda = 1:length(lambda_seq);
        lambda = lambda_seq[i_lambda];
        for i = 1:max_itr
            A = inv(X'X/n + rho*eye(K*P))*(X'*Y/n+rho*(W-C));
            S1 = svd(A+C);
            d1 = (S1.S.-lambda/(2*rho)); d1 = (abs.(d1)+d1)/2;
            W = S1.U*Diagonal(d1)*S1.Vt;
            C = C + A - W;

            # breaking rule
            if max(norm(A-W),norm(A-A_old))<ϵ*norm(A) break end

        end

        output_A[i_lambda] = A';

    end
    return output_A
end
