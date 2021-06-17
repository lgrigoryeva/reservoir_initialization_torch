# Learning PDEs for algorithms

In this repository, there a are two run files:

1. One file trains an echo state network on discrete-time trajectories of either the Brusselator or Lorenz system.
   This can be done by runnning  
   ```
   python main.py
   ```
2. Having trained an echo state network, one can learn a mapping from the data manifold to the internal states of the network.
   This can be done by running  
   ```
   python do_geometric_harmonics.py
   ```

All the hyperparameters as well as utility functions are specified in `utils.py`.
Part of the echo state network code is taken from [this website](https://github.com/danieleds/TorchRC/blob/master/torch_rc/nn/esn.py).
