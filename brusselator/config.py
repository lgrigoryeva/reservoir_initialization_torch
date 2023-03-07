import torch

config = {}
config["DATA"] = {}
# config["DATA"]["n_train"] = 400
# config["DATA"]["n_val"] = 50
# config["DATA"]["n_test"] = 50
config["DATA"]["n_train"] = 1600
config["DATA"]["n_val"] = 4
config["DATA"]["n_test"] = 4
config["DATA"]["l_trajectories"] = 100
config["DATA"]["l_trajectories_test"] = 500
config["DATA"]["parameters"] = {}
config["DATA"]["parameters"]["a"] = 1.0
config["DATA"]["parameters"]["b"] = 2.1
config["DATA"]["max_warmup"] = 50
config["PATH"] = "examples/brusselator/data/"

config["TRAINING"] = {}
config["TRAINING"]["epochs"] = 4000
config["TRAINING"]["batch_size"] = 256
config["TRAINING"]["learning_rate"] = 5e-3
# config["TRAINING"]["ridge"] = True
config["TRAINING"]["ridge"] = True
config["TRAINING"]["dtype"] = torch.float64
config["TRAINING"]["gh_num_eigenpairs"] = 100
config["TRAINING"]["offset"] = 10
config["TRAINING"]["ridge_factor"] = 1e-7
config["TRAINING"]["device"] = "cpu"

config["MODEL"] = {}
# Number of variables to use when using the lorenz system
config["MODEL"]["input_size"] = 1
config["MODEL"]["reservoir_size"] = 1024
config["MODEL"]["hidden_size"] = []
config["MODEL"]["scale_rec"] = 0.9
config["MODEL"]["scale_in"] = 0.02
config["MODEL"]["leaking_rate"] = 0.5

# Train loss: 0.000068
# Val loss: 0.000051
