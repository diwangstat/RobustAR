# This file contains the estimation algorithm for sparse VAR models

# L1 Yule-Walker estimator with truncation parameter τ and regularization parameter λ

# Method 1
# Linearized ADMM algorithm
# Input: truncated autocovariances

function sparse_admm_single(Σ_0_tr, Σ_1_tr, p, d, lambda, ϵ=1e-4, max_itr=500)

    # initialization
    A_init = Σ_1_tr*pinv(Σ_0_tr);
    A = A_old = A_init;
    W = reshape(zeros(p*p*d),p,p*d);
    D = A * Σ_0_tr - Σ_1_tr;

    γ = 1; μ = 2*maximum(eigen(Σ_0_tr^2).values);

    for i in 1:max_itr

        # A-step
        interm = A.-(2/μ)*(A*Σ_0_tr-Σ_1_tr-D-W/γ)*Σ_0_tr;
        A = ((interm.-2/(γ*μ)).>0).*(interm.-2/(γ*μ)).-((-interm.-2/(γ*μ)).>0).*(-interm.-2/(γ*μ));

        # D-step
        D = min.(max.(A*Σ_0_tr-Σ_1_tr-W/γ,-lambda),lambda);

        # W-step
        W = W + γ*(Σ_1_tr - A*Σ_0_tr + D);

        # early stopping
        if max(norm(D-Σ_1_tr+A*Σ_0_tr),norm(A-A_old))<ϵ*norm(A) break end
        A_old = A;
    end

    return A

end

# ADMM algorithm for a sequence of lambda with warm start

function sparse_admm_seq(Σ_0_tr, Σ_1_tr, p, d, lambda_seq, ϵ=1e-4, max_itr=500)

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
            interm = A.-(2/μ)*(A*Σ_0_tr-Σ_1_tr-D-W/γ)*Σ_0_tr;
            A = ((interm.-2/(γ*μ)).>0).*(interm.-2/(γ*μ)).-((-interm.-2/(γ*μ)).>0).*(-interm.-2/(γ*μ));

            # D-step
            D = min.(max.(A*Σ_0_tr-Σ_1_tr-W/γ,-lambda),lambda);

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

# ADMM algorithm

# Input: time series, d, tau_seq, lambda_seq

function sparse_robust_admm(ts, lag_order, tau_seq, lambda_seq, ϵ=1e-4, max_itr=500)

    p,n = size(ts);
    d = lag_order;

    output_A = Array{Array{Float64}}(undef,length(lambda_seq)*length(tau_seq));

    for tau_ind = 1:length(tau_seq)

        tau = tau_seq[tau_ind];

        # data truncation
        ts_tr = sign.(ts).*min.(abs.(ts),tau);
        if d == 1
            Σ_0_tr = ts_tr*ts_tr'/n;
            Σ_1_tr = ts_tr[:,2:end]*ts_tr[:,1:end-1]'/(n-1);
        else
            global X_tr = ts_tr[:,1:end-d];
            for s in 2:d
                global X_tr = [X_tr; ts_tr[:,s:end-d+s-1]];
            end
            global Σ_0_tr = X_tr*X_tr'/(n-d);
            global Σ_1_tr = ts_tr[:,(d+1):end]*X_tr'/(n-d);
        end

        output_A[((tau_ind-1)*length(lambda_seq)+1):tau_ind*length(lambda_seq)] = sparse_admm_seq(Σ_0_tr, Σ_1_tr, p, d, lambda_seq, ϵ, max_itr);

    end

    return output_A

end

function Lasso_VAR(ts, lag_order, lambda_seq, ϵ=1e-5, max_itr=1000)

    p,n = size(ts);
    d = lag_order;
    y = ts;
    Y = ts[:,(d+1):end]';
    global X = ts[:,1:end-d]';
    for s in 2:d
        global X = [X';ts[:,s:end-d+s-1]]';
    end

    A_init = X\Y;
    A = θ = A_old = A_init;
    c = zeros(p*d,p);
    κ = 0.1;

    X_td = [X;eye(p*d)];
    qr_X_td = qr(X_td); M = inv(qr_X_td.R)*qr_X_td.Q';

    output_A = Array{Array{Float64}}(undef,length(lambda_seq));

    for i_lambda = 1:length(lambda_seq)

        λ = lambda_seq[i_lambda];

        for i in 1:max_itr

            A = M*[Y;κ*(θ-c)];
            # θ-step (soft thresholding)
            θ = ((A+c.-λ/κ^2).>0).*(A+c.-λ/κ^2).-((-A-c.-λ/κ^2).>0).*(-A-c.-λ/κ^2);
            # c-step
            c = c + A - θ;
            # early stopping
            if max(norm(A-A_old),norm(A-θ))<ϵ*norm(A) break end
            A_old = A;
        end

        output_A[i_lambda] = A';

    end

    return output_A

end

function sparse_robust_LP(ts, lag_order, tau_list, lambda_list)

    p,n = size(ts);
    d = lag_order;
    tau_length = length(tau_list);
    lambda_length = length(lambda_list);

    y = ts;
    Y = ts[:,(d+1):end]';
    global X = ts[:,1:end-d]';
    for s in 2:d
        global X = [X';ts[:,s:end-d+s-1]]';
    end

    output_A = Array{Array{Float64}}(undef,lambda_length*tau_length);

    for i_tau = 1:tau_length

        tau = tau_list[i_tau];

        # truncated data (d=1)
        y_tr = sign.(y).*min.(abs.(y),tau);
        if d == 1
            Σ_0 = y*y'/n;
            Σ_1 = y_tr[:,2:end]*y_tr[:,1:end-1]'/(n-1);
            Σ_0_tr = y_tr*y_tr'/n;
            Σ_1_tr = y_tr[:,2:end]*y_tr[:,1:end-1]'/(n-1);
        else
            global X_tr = y_tr[:,1:end-d];
            for s in 2:d
                global X_tr = [X_tr; y_tr[:,s:end-d+s-1]];
            end
            global Σ_0_tr = X_tr*X_tr'/(n-d);
            global Σ_1_tr = y_tr[:,(d+1):end]*X_tr'/(n-d);
        end

        big_Σ_0 = kron(Σ_0_tr,eye(p));
        vec_Σ_1 = vec(Σ_1_tr);

        for i_lambda = 1:lambda_length

            lambda = lambda_list[lambda_length-i_lambda+1];

            A = L1_VAR_LP_single2(Σ_1_tr, Σ_0_tr, lambda, p, d);

            output_A[i_lambda+(i_tau-1)*lambda_length] = A;

        end
    end

    return output_A

end

function L1_VAR_LP_single2(Y, X, λ, p, d)

    m = Model(with_optimizer(GLPK.Optimizer));

    @variable(m, B[1:p,1:p*d])
    @variable(m, U[1:p,1:p*d])

    @objective(m, Min, sum(U[i,j] for i in 1:p for j in 1:p*d))

    @constraint(m, constraint1[i in 1:p, j in 1:p*d], B[i,j]+U[i,j] >= 0)
    @constraint(m, constraint2[i in 1:p, j in 1:p*d], B[i,j]-U[i,j] <= 0)
    @constraint(m, constraint3[i in 1:p, j in 1:p*d], sum( B[i,k]*X[k,j] for k in 1:p*d) - Y[i,j] + λ >= 0)
    @constraint(m, constraint4[i in 1:p, j in 1:p*d], sum( B[i,k]*X[k,j] for k in 1:p*d) - Y[i,j] - λ <= 0)

    JuMP.optimize!(m)
    B_opt = zeros(p,p*d);
    for i in 1:p
        for j in 1:p*d
            B_opt[i,j] = JuMP.value(B[i,j])
        end
    end

    return B_opt

end
