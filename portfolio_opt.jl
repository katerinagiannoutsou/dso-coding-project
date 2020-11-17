using CSV, DataFrames, DecisionTree, Distributions, Gurobi, JuMP, LightGraphs, Parameters, SparseArrays, Statistics, ArgParse, Random

include("oracles/portfolio_oracle.jl")
include("solver/util.jl")
include("solver/sgd.jl")
include("solver/reformulation.jl")
include("solver/random_forests_po.jl")
include("solver/validation_set.jl")
include("experiments/replication_functions.jl")


"""
Test to run one instance of the portfolio optimization experiment. 
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


# Fixed parameter settings (these never change)
num_assets = 50
num_factors = 4
n_train = 100

holdout_percent = 0.25
n_holdout = round(Int, holdout_percent*n_train)

n_test = 10000
p_features = 5
n_sigmoid_polydegree = 4
noise_multiplier_tau = 1


rng_seed = 2128
Random.seed!(rng_seed)

port_results = portfolio_replication(num_assets, num_factors, n_train, n_holdout, n_test, p_features,
        n_sigmoid_polydegree, noise_multiplier_tau;
        num_lambda = 10, lambda_max = 100.0, lambda_min_ratio = 10.0^(-6),
        gurobiEnvOracle = envOracle, gurobiEnvReform = envReform, different_validation_losses = false,
        data_type = :poly_kernel)




