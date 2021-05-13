import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm.auto import tqdm

from int.lorenz import integrate

import scipy.integrate as sp

config = {}
config["EXAMPLE"] = 'brusselator'
config["PATH"] = "data/"

config["DATA"] = {}
config["DATA"]["n_train"] = 1
config["DATA"]["n_val"] = 1
config["DATA"]["n_test"] = 1
config["DATA"]["l_trajectories"] = 2000
config["DATA"]["l_trajectories_test"] = 500
config["DATA"]["lenght_chunks"] = 10
config["DATA"]["shift_betw_chunks"] = 1
config["DATA"]["initial_set_off"] = 10
config["DATA"]["max_n_transients"] = 200
config["DATA"]["max_warmup"] = 50
config["DATA"]["integration_steps"] = 250
config["DATA"]["gh_lenght_chunks"] = 5

config["MODEL"] = {}
config["MODEL"]["input_size"] = 3
config["MODEL"]["reservoir_size"] = 100

config["TRAINING"] = {}
config["TRAINING"]["epochs"] = 500
config["TRAINING"]["batch_size"] = 128
config["TRAINING"]["learning_rate"] = 5e-1
config["TRAINING"]["ridge"] = True
config["TRAINING"]['dtype'] = torch.float64


class LorenzDataset(torch.utils.data.Dataset):
    """Dataset of transients obtained from a Brusselator."""

    def __init__(self, n_train, l_trajectories, input_size, verbose=False):
        self.ids = np.arange(n_train)

        self.x = []
        self.y = []
        self.y_data = []
        for i in tqdm(range(n_train), leave=True, position=0):
            data_dict = integrate(T=l_trajectories, ic='random')
            self.x.append(data_dict["data"][:-1, :input_size])
            self.y.append(data_dict["data"][1:, :input_size])
        self.tt = data_dict["tt"]
        if verbose:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(data_dict["data"][:, 0], data_dict["data"][:, 1], data_dict["data"][:, 2])
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.set_zlabel(r'$z$')
            plt.savefig('fig/lorenz_trajectory.pdf')
            plt.show()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x = self.x[self.ids[index]]
        y = self.y[self.ids[index]]
        return torch.tensor(x, dtype=torch.get_default_dtype()), \
            torch.tensor(y, dtype=torch.get_default_dtype())

def f_brusselator(t, y, arg_a, arg_b):
    """Temporal evolution of the Brusselator."""
    dx_dt = arg_a + np.real(y)**2*np.imag(y) - (arg_b+1)*np.real(y)
    dy_dt = arg_b*np.real(y) - np.real(y)**2*np.imag(y)
    return dx_dt+1.0j*dy_dt


def integrate_brusselator(initial_condition, l_trajectories, tmin=0,
                          delta_t=1e-3, parameters={'a': 1.0, 'b': 2.1}):
    """Integrate Brusselator using initial condition y0."""
    initial_time = 0.0
    tmax = l_trajectories/5
    tt_arr = np.linspace(tmin, tmax, l_trajectories+1)
    data = []
    runner = sp.ode(f_brusselator).set_integrator('zvode', method='Adams')
    runner.set_initial_value(initial_condition, initial_time).set_f_params(parameters['a'],
                                                                           parameters['b'])
    data.append(runner.y)

    i = 0
    while runner.successful() and np.abs(runner.t) < np.abs(tmax):
        i = i + 1
        runner.integrate(runner.t + delta_t)
        if (i % int(float(tmax) / float(l_trajectories*delta_t)) == 0):
            data.append(runner.y)
    return tt_arr, np.array(data, dtype='complex')


class BrusselatorDataset(torch.utils.data.Dataset):
    """Dataset of transients obtained from a Brusselator."""

    def __init__(self, n_train, l_trajectories, verbose=False):
        self.ids = np.arange(n_train)

        self.x = []
        self.y = []
        self.y_data = []
        for i in tqdm(range(n_train), leave=True, position=0):
            y0 = 2.0*np.random.random() + 1.0j*3.0*np.random.random()
            tt_arr, trajectory = integrate_brusselator(y0, l_trajectories)
            self.x.append(trajectory[:-1].real)
            self.y.append(trajectory[1:].real)
            self.y_data.append(trajectory[:-1].imag)
        self.tt = tt_arr
        if verbose:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(trajectory[:-1].real, trajectory[:-1].imag)
            ax.set_xlabel(r'$u$')
            ax.set_ylabel(r'$v$')
            plt.savefig('fig/brusselator_trajectory.pdf')
            plt.show()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x = self.x[self.ids[index]]
        y = self.y[self.ids[index]]
        return torch.tensor(x, dtype=torch.get_default_dtype()), torch.tensor(y, dtype=torch.get_default_dtype())



class ESN(nn.Module):
    """Taken from https://github.com/danieleds/TorchRC/blob/master/torch_rc/nn/esn.py."""

    def __init__(self,
                 input_size: int,
                 reservoir_size: int,
                 output_size: int,
                 scale_rec: float = 1/1.1,
                 scale_in: float = 1.0/40.,
                 density_in: float = 1.0,
                 leaking_rate: float = 1.0,
                 rec_rescaling_method: str = 'specrad',  # Either "norm" or "specrad"
                 quadratic: bool = True
                 ):
        super(ESN, self).__init__()

        self.reservoir_size = reservoir_size
        self.input_size = input_size
        self.output_size = output_size
        self.quadratic = quadratic

        self.leaking_rate = leaking_rate

        # Reservoir
        # W_in = torch.rand((output_size, input_size)) * 2 - 1
        W_in = torch.rand((reservoir_size, input_size)) - 1/2.
        # W_hat = torch.rand((output_size, output_size)) * 2 - 1
        W_hat = torch.randn((reservoir_size, reservoir_size))

        W_in = scale_in*W_in
        W_hat = self.rescale_contractivity(W_hat, scale_rec, rec_rescaling_method)

        # Assign as buffers
        self.register_buffer('W_in', W_in)
        self.register_buffer('W_hat', W_hat)

        if self.quadratic:
            self.readout = nn.Linear(int(2*reservoir_size), output_size, bias=False)
        else:
            self.readout = nn.Linear(reservoir_size, output_size, bias=False)

    @staticmethod
    def rescale_contractivity(W, coeff, rescaling_method):
        if rescaling_method == 'norm':
            return W * coeff / W.norm()
        elif rescaling_method == 'specrad':
            return W * coeff / (W.eig()[0].abs().max())
        else:
            raise Exception("Invalid rescaling method used (must be either 'norm' or 'specrad')")

    def forward_reservoir(self, input, hidden):
        """
        input: (batch, input_size)
        hidden: (batch, hidden_size)
        output: (batch, hidden_size)
        """

        # h_tilde = torch.tanh(self.W_in @ input.t() + self.W_hat @ hidden.t()).t()
        h_tilde = torch.tanh(torch.mm(input, self.W_in.t()) + torch.mm(hidden, self.W_hat.t()))
        h = (1 - self.leaking_rate) * hidden + self.leaking_rate * h_tilde
        return h

    # def forward(self, input, h_0=None):
    #     batch = input.shape[1]

    #     if h_0 is None:
    #         h_0 = input.new_zeros(
    #             (batch, self.reservoir_size))

    #     # Separate the layers to avoid in-place operations
    #     # h_l = list(h_0.unbind())

    #     next_layer_input = input  # (sequence_length, batch, input_size)
    #     layer_outputs = []  # list of (batch, hidden_size)
    #     step_h = h_0
    #     for x_t in next_layer_input:
    #         h = self.forward_reservoir(x_t, step_h)  # (batch, hidden_size)
    #         step_h = h
    #         layer_outputs.append(self.readout(h))
    #     h_n = step_h
    #     layer_outputs = torch.stack(layer_outputs)

    #     return layer_outputs, h_n

    def forward(self, input, h_0=None, return_states=False):
        batch = input.shape[0]
        # print(input.shape)

        if h_0 is None:
            h_0 = input.new_zeros(
                (batch, self.reservoir_size))

        # Separate the layers to avoid in-place operations
        # h_l = list(h_0.unbind())

        next_layer_input = input  # (sequence_length, batch, input_size)
        layer_outputs = []  # list of (batch, hidden_size)
        step_h = h_0
        # for x_t in next_layer_input:
        #     x_t = next_layer_input[:, i]
        for i in range(next_layer_input.shape[1]):
            x_t = next_layer_input[:, i]
            h = self.forward_reservoir(x_t, step_h)  # (batch, hidden_size)
            step_h = h
            # print(h.shape)
            if self.quadratic:
                h = torch.cat([h, h**2], axis=1)
                # print(h.shape)
            if return_states:
                layer_outputs.append(h)
            else:
                layer_outputs.append(self.readout(h))
        h_n = step_h
        layer_outputs = torch.stack(layer_outputs, axis=1)
        # print(layer_outputs.shape)
        # print(layer_outputs.shape)

        return layer_outputs, h_n


class ESNModel():
    def __init__(self, dataloader_train, dataloader_val, network, learning_rate=0.05, device=None):
        if torch.cuda.is_available() and device is None:
            self.device = 'cuda'
        elif not torch.cuda.is_available() and device is None:
            self.device = 'cpu'
        else:
            self.device = device

        print('Using:', self.device)

        self.net = network.to(self.device)

        self.trainable_parameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        print('Trainable parameters: '+str(self.trainable_parameters))

        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=learning_rate)

        self.criterion = torch.nn.MSELoss().to(self.device)

        self.train_loss = []
        self.val_loss = []

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5, min_lr=0.000001)

    def train(self, ridge=False):
        """Train model."""
        if ridge:
            self.net.train()
            cnt, sum_loss = 0, 0
            for (x, y) in self.dataloader_train:
                out, _ = self.net(x.to(self.device), return_states=True)
                print(y.shape)
                print(out.shape)
                print(torch.eye(out[0].shape[1], out[0].shape[1]).shape)
                # ridge = torch.matmul(torch.matmul(torch.transpose(y[0].to(self.device), 0, 1), out[0]),
                #                      torch.inverse(
                #                          torch.matmul(torch.transpose(out[0], 0, 1), out[0]) +
                #                          torch.tensor(1e-4).to(self.device)*torch.eye(
                #                              out[0].shape[1], out[0].shape[1]).to(self.device)))
                ytmp = torch.transpose(y[0].to('cpu'), 0, 1)
                outtmp = torch.transpose(out[0], 0, 1).to('cpu')
                print(torch.eye(outtmp.shape[0], outtmp.shape[0]).shape)
                ridge = torch.matmul(torch.matmul(ytmp,
                                                  torch.transpose(outtmp, 0, 1)),
                                     torch.inverse(
                                         torch.matmul(outtmp, torch.transpose(outtmp, 0, 1)) +
                                         torch.tensor(5e-6).to('cpu')*torch.eye(
                                             outtmp.shape[0], outtmp.shape[0]).to('cpu')))
                # ridge = torch.matmul(torch.matmul(y[0].to(self.device),
                #                                   torch.transpose(out[0], 0, 1)),
                #                      torch.inverse(
                #                          torch.matmul(out[0], torch.transpose(out[0], 0, 1)) +
                #                          torch.tensor(1e-4).to(self.device)*torch.eye(
                #                              out[0].shape[1], out[0].shape[1]).to(self.device)))
                print(ridge.shape)
                print(self.net.readout.weight.shape)
                self.net.readout.weight = torch.nn.Parameter(ridge.to(self.device))
                loss = self.criterion(self.net.readout(out), y.to(self.device))
                sum_loss += loss.detach().cpu().numpy()
                cnt += 1
        else:
            self.net.train()
            cnt, sum_loss = 0, 0
            for (x, y) in self.dataloader_train:
                self.optimizer.zero_grad()
                out, _ = self.net(x.to(self.device))
                loss = self.criterion(out, y.to(self.device))
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.detach().cpu().numpy()
                cnt += 1
            self.optimizer.zero_grad()
            self.scheduler.step(sum_loss / cnt)
        self.train_loss.append(sum_loss/cnt)
        return sum_loss/cnt

    def validate(self):
        """Validate model."""
        self.net.eval()
        cnt, sum_loss = 0, 0
        with torch.no_grad():
            for (x, y) in self.dataloader_val:
                out, _ = self.net(x.to(self.device))
                loss = self.criterion(out, y.to(self.device))
                sum_loss += loss.detach().cpu().numpy()
                cnt += 1
        self.val_loss.append(sum_loss/cnt)
        return sum_loss/cnt

    def integrate(self, x_0, T, h0=None):
        """Integrate single trajectory."""
        self.net.eval()

        if h0 is None:
            h0 = torch.zeros(1, self.net.reservoir_size).to(self.device)

        x_list = []
        h_list = []
        x_0 = x_0.to(self.device)
        x_list.append(x_0[0].squeeze().detach().cpu().numpy())
        h_list.append(h0.squeeze().detach().cpu().numpy())
        h_t = h0
        # Warmup
        for x_t in x_0:
            x_out, h_t = self.net.forward(x_t.unsqueeze(0).unsqueeze(0), h_0=h_t)
            x_list.append(x_out.squeeze().detach().cpu().numpy())
            h_list.append(h_t.squeeze().detach().cpu().numpy())

        # Autoregressive integration
        for _ in tqdm(range(T), position=0, leave=True):
            if len(x_out.size()) > 2:
                x_out, h_t = self.net.forward(x_out, h_0=h_t)
            else:
                x_out, h_t = self.net.forward(x_out.unsqueeze(0), h_0=h_t)
            x_list.append(x_out.squeeze().detach().cpu().numpy())
            h_list.append(h_t.squeeze().detach().cpu().numpy())
        return np.array(x_list), np.array(h_list)

    def save_network(self, name):
        """Save network weights and training loss history."""
        filename = name+'_reservoir_size_'+str(self.net.reservoir_size)+'.net'
        torch.save(self.net.state_dict(), filename)
        np.save(name+'_training_loss.npy', np.array(self.train_loss))
        np.save(name+'_validation_loss.npy', np.array(self.val_loss))
        return name

    def load_network(self, name):
        """Load network weights and training loss history."""
        filename = name+'_reservoir_size_'+str(self.net.reservoir_size)+'.net'
        self.net.load_state_dict(torch.load(filename))
        self.train_loss = np.load(name+'_training_loss.npy').tolist()
        self.val_loss = np.load(name+'_validation_loss.npy').tolist()


def progress(train_loss, val_loss):
    """Define progress bar description."""
    return "Train/Loss: {:.6f}  Val/Loss: {:.6f}".format(
        train_loss, val_loss)
