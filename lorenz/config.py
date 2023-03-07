import torch

config = {}
config["DATA"] = {}
config["DATA"]["n_train"] = 1600
config["DATA"]["n_val"] = 4
config["DATA"]["n_test"] = 4
config["DATA"]["l_trajectories"] = 100
config["DATA"]["l_trajectories_test"] = 500
config["DATA"]["parameters"] = {}
config["DATA"]["parameters"]['sigma'] = 10.0
config["DATA"]["parameters"]['rho'] = 28.0
config["DATA"]["parameters"]['beta'] = 8.0/3.0
config["DATA"]["max_warmup"] = 50
config["PATH"] = "examples/lorenz/data/"

config["TRAINING"] = {}
config["TRAINING"]["epochs"] = 1000
config["TRAINING"]["batch_size"] = 400
config["TRAINING"]["learning_rate"] = 5e-3
config["TRAINING"]["ridge"] = True
config["TRAINING"]["dtype"] = torch.float64
config["TRAINING"]["gh_num_eigenpairs"] = 100
config["TRAINING"]["offset"] = 10
config["TRAINING"]["ridge_factor"] = 2e-6
config["TRAINING"]["device"] = 'cpu'

config["MODEL"] = {}
# Number of variables to use when using the lorenz system
config["MODEL"]["input_size"] = 1
config["MODEL"]["reservoir_size"] = 2048
config["MODEL"]["hidden_size"] = []
config["MODEL"]["scale_rec"] = 0.9
config["MODEL"]["scale_in"] = 0.02
config["MODEL"]["leaking_rate"] = 0.5

# Train loss: 0.000068
# Val loss: 0.000051
