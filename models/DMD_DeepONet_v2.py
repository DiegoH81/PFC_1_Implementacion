import torch
import torch.nn as nn


class DMD_DeepONet_v2(nn.Module):
    def __init__(self, branch_dim, modes_dim, dynamics_dim, trunk_dim):
        super().__init__()
        p = branch_dim[-1]
        self.branch = self.make_mlp(branch_dim)
        self.branch_dmd_modes = self.make_mlp(modes_dim)
        self.branch_dmd_dynamics = self.make_mlp(dynamics_dim)
        self.trunk = self.make_mlp(trunk_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def make_mlp(self, dims):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, f, modes, dynamics, x):
        b0 = self.branch(f.flatten(1))
        y_dmd = (self.branch_dmd_modes(modes)
                 * self.branch_dmd_dynamics(dynamics)).sum(dim = 1)
        y_tr = self.trunk(x)
        return (b0 * y_dmd) @ y_tr.T + self.bias