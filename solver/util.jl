#=
Assorted utility functions used throughout the code.
Dependencies: none
=#

using LinearAlgebra, Gurobi, Distributions

"""
	cost_least_squares(X, c, intercept=false)
Solve a least-squares problem to fit a d x p matrix B such that c approx == B*X
# Arguments
- `X`: p x n feature matrix
- `c`: d x n matrix of cost vectors
"""
function cost_least_squares(X, c; intercept=false)
	(p, n) = size(X)
	(d, n2) = size(c)

	if n != n2
		error("dimensions are mismatched")
	end

	if intercept
		X = [X; ones(1,n)]
	end

	Xt = X'
	ct = c'

	Bt = Xt\ct

	return Bt'
end

"""
	ridge(X, c, reg_param)
Solve a ridge regression problem min_B (1/n)0.5*||c - B*X||^2 + 0.5*reg_param*||B||^2
Here ||.|| is the Frobenius norm
# Arguments
- `X`: p x n feature matrix
- `c`: d x n matrix of cost vectors
"""
function ridge(X, c, reg_param)
	(p, n) = size(X)
	(d, n2) = size(c)

	if n != n2
		error("dimensions are mismatched")
	end

	Xt = X'
	ct = c'

	Bt = (X*Xt + n*reg_param*eye(p))\(X*ct)

	return Bt'
end

"""
	oracle_dataset(c, oracle)
Apply the optimization oracle to each column of the d x n matrix c, return
an n vector of z_star values and a d x n matrix of optimal solutions
"""
function oracle_dataset(c, oracle)
	(d, n) = size(c)
	z_star_data = zeros(n)
    w_star_data = zeros(d, n)
    for i = 1:n
        (z_i, w_i) = oracle(c[:,i])
        z_star_data[i] = z_i
        w_star_data[:,i] = w_i
    end
	return (z_star_data, w_star_data)
end

"""
	spo_loss(B_new, X, c, oracle; z_star=[])
Compute the SPO loss of `B_new` on the training/holdout/test set (`X`, `c`),
given the optimization oracle, `oracle`.
# Arguments
- `B_new`: d x p model matrix B
- `X`: p x n feature matrix
- `c`: d x n matrix of cost vectors
- `oracle`: the optimization oracle function
- `z_star`: optional vector of the z^\ast(c_i) values
"""
function spo_loss(B_new, X, c, oracle; z_star=[])

	if z_star == []
		(z_star, w_star) = oracle_dataset(c, oracle)
	end

	n = length(z_star)

	spo_sum = 0
	for i = 1:n
		c_hat = B_new*X[:,i]
		(z_oracle, w_oracle) = oracle(c_hat)
		spo_loss_cur = dot(c[:,i], w_oracle) - z_star[i]
		spo_sum = spo_sum + spo_loss_cur
	end
	spo_loss_avg = spo_sum/n
	return spo_loss_avg
end

"""
	compare_models_percent(B_modA, B_modB, X, c, oracle; eps=0.000001)
Compute the percentage of times on the training/test set (X, c) such that modA produces a
solution wtih smaller true cost than modB.
# Arguments
- `B_modA`: d x p model matrix B of modA
- `B_modB`: d x p model matrix B of modB
- `X`: p x n feature matrix
- `c`: d x n matrix of (true) cost vectors
- `oracle`: the optimization oracle function
- `eps`: tolerance paramter. modA beats modB if cost(modA) <= cost(modB) + eps
"""
function compare_models_percent(B_modA, B_modB, X, c, oracle; eps=0.000001)
	(d, n) = size(c)

	count = 0
	for i = 1:n
		c_hatA = B_modA*X[:,i]
		c_hatB = B_modB*X[:,i]

		(z_oracleA, w_oracleA) = oracle(c_hatA)
		(z_oracleB, w_oracleB) = oracle(c_hatB)

		diff = dot(c[:,i], w_oracleA) - dot(c[:,i], w_oracleB)
		if diff <= eps
			count = count + 1
		end
	end
	return count/n
end

"""
	spo_plus_loss(B_new, X, c, oracle; z_star=[], w_star=[])
Compute the SPO plus loss of `B_new` on the training/holdout/test set (`X`, `c`),
given the optimization oracle, `oracle`.
# Arguments
- `B_new`: d x p model matrix B
- `X`: p x n feature matrix
- `c`: d x n matrix of cost vectors
- `oracle`: the optimization oracle function
- `z_star`: optional vector of the z^\ast(c_i) values
- `w_star`: optional d x n matrix of w^\ast(c_i)
"""
function spo_plus_loss(B_new, X, c, oracle; z_star=[], w_star=[])

	if z_star == [] || w_star == []
		(z_star, w_star) = oracle_dataset(c, oracle)
	end

	n = length(z_star)

	spo_plus_sum = 0
	for i = 1:n
		c_hat = B_new*X[:,i]
		spoplus_cost_vec = 2*c_hat - c[:,i]
		(z_oracle, w_oracle) = oracle(spoplus_cost_vec)
		spo_plus_cost = -z_oracle + 2*dot(c_hat, w_star[:,i]) - z_star[i]
		spo_plus_sum = spo_plus_sum + spo_plus_cost
	end
	spo_plus_avg = spo_plus_sum/n
	return spo_plus_avg
end

"""
	least_squares_loss(B_new, X, c)
Compute the least squares loss of `B_new` on the training/holdout/test set (`X`, `c`).
# Arguments
- `B_new`: d x p model matrix B
- `X`: p x n feature matrix
- `c`: d x n matrix of cost vectors
"""
function least_squares_loss(B_new, X, c)
	(p, n) = size(X)
	residuals = B_new*X - c
	error = (1/n)*(1/2)*(norm(residuals)^2)

	return error
end

"""
Absoulte (i.e. ell_1) loss
"""
function absolute_loss(B_new, X, c)
	(p, n) = size(X)
	residuals = B_new*X - c
	error = (1/n)*norm(residuals, 1)

	return error
end

"""
Huber loss with parameter delta
"""
function huber_loss(B_new, X, c, delta)
	(p, n) = size(X)
	(d, n_2) = size(c)
	if n != n_2
		error("Dimension mismatch in Huber loss calucation.")
	end

	residuals = B_new*X - c
	total_error = 0
	for i = 1:n
		for j = 1:d
			a_res = abs(residuals[j, i])
			if a_res <= delta
				cur_error = 0.5*a_res^2
			else
				cur_error = delta*(a_res - 0.5*delta)
			end
			total_error = total_error + cur_error
		end
	end

	error = (1/n)*total_error
	return error
end


#addind r_2 function

##
function r_2(B_new, X, c_train, c_bar_vector, n_train, p_features)
    residuals = B_new*X - c_train
    squared_residuals = (norm(residuals)^2)

    total = c_bar_vector - c_train 
    squared_total = (norm(total)^2)

    #compute R^2
    r_squared = 1 - (squared_residuals/squared_total)
    
    #compute adjusted R^2
    r_squared_adj = 1 - (1 - r_squared)*((n_train-1)/(n_train-p_features-1))

    return (r_squared, r_squared_adj)
end

"""
	generate_poly_kernel_data(B_true, n, degree; inner_constant=1, outer_constant = 1, kernel_damp_normalize=true,
		kernel_damp_factor=1, noise=true, noise_half_width=0, normalize_c=true)
Generate (X, c) from the polynomial kernel model X_{ji} ~ N(0, 1) and
c_i(j) = ( (alpha_j * B_true[j,:] * X[:,i]  + inner_constant)^degree + outer_constant ) * epsilon_{ij} where
alpha_j is a damping term and epsilon_{ij} is a noise term.
# Arguments
- `kernel_damp_normalize`: if true, then set
alpha_j = kernel_damp_factor/norm(B_true[j,:]). This results in
(alpha_j * B_true[j,:] * X[:,i]  + inner_constant) being normally distributed with
mean inner_constant and standard deviation kernel_damp_factor.
- `noise`:  if true, generate epsilon_{ij} ~  Uniform[1 - noise_half_width, 1 + noise_half_width]
- `normalize_c`:  if true, normalize c at the end of everything
"""
function generate_poly_kernel_data(B_true, n, degree; inner_constant=1, outer_constant = 1, kernel_damp_normalize=true,
	kernel_damp_factor=1, noise=true, noise_half_width=0, normalize_c=true, normalize_small_threshold = 0.0001)

    (d, p) = size(B_true)
    X_observed = randn(p, n)
    #try exponential to compare performance
    #X_observed = randexp(p, n)
   
    dot_prods = B_true*X_observed

	# first generate c_observed without noise
    c_observed = zeros(d, n)
    for j = 1:d
		if kernel_damp_normalize
			cur_kernel_damp_factor = kernel_damp_factor/norm(B_true[j,:])
		else
			cur_kernel_damp_factor = kernel_damp_factor
		end
        for i = 1:n
			c_observed[j, i] = (cur_kernel_damp_factor*dot_prods[j, i] + inner_constant)^degree + outer_constant
			if noise
				epsilon = (1 - noise_half_width) + 2*noise_half_width*rand()
				c_observed[j, i] = c_observed[j, i]*epsilon
			end
        end
    end

	if normalize_c
		for i = 1:n
			c_observed[:, i] = c_observed[:, i]/norm(c_observed[:, i])
			for j = 1:d
				if abs(c_observed[j, i]) < normalize_small_threshold
					c_observed[j, i] = 0
				end
			end
		end
	end

    return (X_observed, c_observed)
end

function generate_poly_kernel_data_simple(B_true, n, polykernel_degree, noise_half_width)
	(d, p) = size(B_true)
	alpha_factor = 1/sqrt(p)

	if noise_half_width == 0
		noise_on = false
	else
		noise_on = true
	end

	return generate_poly_kernel_data(B_true, n, polykernel_degree;
        kernel_damp_factor = alpha_factor,
		noise = noise_on, noise_half_width = noise_half_width,
        kernel_damp_normalize = false,
        normalize_c = false,
		inner_constant = 3)
end



function setup_gurobi_env(; quiet_mode = true, method_type = :barrier, use_time_limit = true, time_limit = 60.0)
	env = Gurobi.Env()

	if quiet_mode
		setparams!(env; OutputFlag = 0)
	end

	if method_type == :barrier
		setparams!(env; Method = 2)
	elseif method_type == :method3
		setparams!(env; Method = 3)
	elseif method_type != :default
		error("Enter a valid method type for Gurobi.")
	end

	if use_time_limit
		setparams!(env; TimeLimit = time_limit)
	end

	return env
end
