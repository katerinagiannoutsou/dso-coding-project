using CSV, DataFrames, DecisionTree, Distributions, Gurobi, JuMP, LightGraphs, Parameters, SparseArrays, Statistics, ArgParse, Random

include("oracles/shortest_path_oracle.jl")
include("solver/util.jl")
include("solver/sgd.jl")
include("solver/reformulation.jl")
include("solver/random_forests_po.jl")
include("solver/validation_set.jl")
include("experiments/replication_functions.jl")

"""
Test to run one instance of the shortest path experiment. 
Using the reformulation approach (spo+ formulation). (reformulation.jl)
replication_functions.jl : Functions for generating a single replication of a particular experiment
		(Dependencies: util.jl, sgd.jl, reformulation.jl, random_forests_po.jl, validation_set.jl) 
		To change validation loss , will need to adjust: util.jl
		To change data generation method , will need to adjust: util.jl
		To change training loss algorithm (currently spo+), will need to adjust: util.jl, sgd.jl, validation_set.jl

Above dependencies from SPO repository:
https://github.com/paulgrigas/SmartPredictThenOptimize

"""

#environment
envOracle = setup_gurobi_env(method_type = :default, use_time_limit = false)
envReform = setup_gurobi_env(method_type = :method3, use_time_limit = false)

# parameter settings 
p_features = 5
grid_dim = 5

num_lambda = 10
lambda_max = 100
lambda_min_ratio = 10.0^(-6)
holdout_percent = 0.25
n_holdout = round(Int, holdout_percent*n_train)

#For regularization used :lasso and :ridge

#setseed
rng_seed = 2128
Random.seed!(rng_seed)

#one instance of the problem:
n_train = 100
n_test = 1000
polykernel_degree = 2  #used 1, 2, 4, 8
polykernel_noise_half_width = 0 #and 0.5 


sp_results = shortest_path_replication(grid_dim,
                    	n_train, n_holdout, n_test,
                        p_features, polykernel_degree, polykernel_noise_half_width;
                        num_lambda = 10, lambda_max = missing, lambda_min_ratio = 0.0001, regularization = :ridge,
                        gurobiEnvOracle = envOracle, gurobiEnvReform = envReform,
                        different_validation_losses = false,
                        include_rf = true)



