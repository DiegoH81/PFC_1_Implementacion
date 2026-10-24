import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.func import jvp
from pydmd import DMD
import numpy as np
import pandas as pd

import Equations as eq
import Painter


def derivate(net, s):
    f1 = lambda z: jvp(net, (z,), (torch.ones_like(z),))
    (f, df), (_, d2f) = jvp(f1, (s,), (torch.ones_like(s),))
    return f, df, d2f

class SEP_DeepONet(nn.Module):
    def __init__(self, branch_dim, trunk_dim, rank):
        super().__init__()
        self.latent_dim = branch_dim[-1]
        self.rank = rank
        
        trunk_dim_with_rank = trunk_dim[:-1] + [trunk_dim[-1] * rank]
        self.branch_net = self.make_mlp(branch_dim)
        self.trunk_t = self.make_mlp(trunk_dim_with_rank)
        self.trunk_x = self.make_mlp(trunk_dim_with_rank)
        self.trunk_y = self.make_mlp(trunk_dim_with_rank)
        self.trunk_alpha = self.make_mlp(trunk_dim_with_rank)

        self.output_bias = nn.Parameter(torch.zeros(1))
    
    def make_mlp(self, dims):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)
    
    def forward(self, init, x, y, t, alpha):
        f_init = self.branch_net(init)
        
        f_x = self.trunk_x(x).reshape(-1, self.latent_dim, self.rank)
        f_y = self.trunk_y(y).reshape(-1, self.latent_dim, self.rank)
        f_t = self.trunk_t(t).reshape(-1, self.latent_dim, self.rank)
        f_a = self.trunk_alpha(alpha).reshape(-1, self.latent_dim, self.rank)
        
        out = torch.einsum('bp,xpr,ypr,tpr,apr->bxyta',
                           f_init, f_x, f_y, f_t, f_a)

        return out + self.output_bias
    
    
if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    alphas = [0.1, 0.3, 1.0]
    n_samples = 3
    dim_malla = 101
    t_size = 101
    total_time = 1.0
    
    n_train = 25
    w_ic, w_bc, w_pde = 51, 51, 31
    
    latent_dim, rank, hidden = 50, 50, 50

    n_epochs = 100000
    log_every = 1000
    learning_rate = 1e-3
    
    print("Generating dataset")
    datasets = []
    for a in alphas:
        datasets.append(eq.HeatEquationDataset(equation_class = eq.HeatEquation2DConst,
                                               num_samples = n_samples, spatial_size = (dim_malla, dim_malla),
                                               total_time = total_time, temporal_size = t_size,
                                               alpha = a))
        
    system_painter = Painter.HeatEquationPainter()
    #system_painter.plot_solution(datasets[0][10], 10, "Testing draw")
    
    print("Processing test data")
    tests = []
    for a, dataset in zip(alphas, datasets):
        for sol in dataset:
            T0 = float(sol[0, 1, 1])
            sol_tensor = torch.tensor(sol, dtype = torch.float32, device = device)
            tests.append((T0, a, sol_tensor))

   