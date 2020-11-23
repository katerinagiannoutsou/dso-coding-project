using CSV, DataFrames, DecisionTree, Distributions, Gurobi, JuMP, LightGraphs, Parameters, SparseArrays, Statistics, ArgParse
cd("~/code")

include("oracles/shortest_path_oracle.jl")
include("solver/util.jl")
include("solver/sgd.jl")
include("solver/reformulation.jl")
include("solver/random_forests_po.jl")
include("solver/validation_set.jl")
include("experiments/replication_functions.jl")

# Fix parameters 
p_features = 20 #try: 10, 20
grid_dim = 5
num_trials = 30 #30 #50
n_test = 1000

num_lambda = 10
lambda_max = 100
lambda_min_ratio = 10.0^(-8)
holdout_percent = 0.25
regularization = :lasso
different_validation_losses = false


n_train_vec = [100] #[100; 1000]
polykernel_degree_vec = [1; 2; 4; 6; 8] #also tried: [1; 4; 8; 12; 16; 20]
polykernel_noise_half_width_vec = [0.5]

# random seed 
rng_seed = 5352 #for comparison

# Run experiment and get results: save separate dataframes based on runs with parameters above
# Note that Gurobi enviornments are set within this function call
expt_results = shortest_path_multiple_replications(rng_seed, num_trials, grid_dim,
    n_train_vec, n_test,
    p_features, polykernel_degree_vec, polykernel_noise_half_width_vec;
    num_lambda = num_lambda, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio,
    holdout_percent = holdout_percent, regularization = regularization,
    different_validation_losses = different_validation_losses)

#save respective dataframe
csv_string = "shortest_path_100n_20p.csv"
CSV.write(csv_string, expt_results)

