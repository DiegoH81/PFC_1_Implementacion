import torch
import torch.nn as nn
from pydmd import DMD



class DMDPart:
    def __init__(self, data, rank: int):
        self.data = data
        self.rank = rank
        
    def _compute_dmd(self):
        snapshots = self.data.reshape(self.data.shape[0], -1).T
        dmd = DMD(svd_rank = self.rank)
        dmd.fit(snapshots)

        return dmd

    def method(self):
        dmd = self._compute_dmd()
        modes, dynamics = [], []
                
        sz = len(dmd.amplitudes)
        
        for i in range (sz):
            modes.append(dmd.modes.real[:, i])
            dynamics.append(dmd.dynamics.real[i])

        return [modes, dynamics]

class DMD_DeepONet(nn.Module):
    def __init__(self, branch_dim, modes_dim, dynamics_dim, trunk_dim, output_dim):
        super().__init__()
        
        self.branch = self.make_mlp(branch_dim)
        
        self.branch_dmd_modes = self.make_mlp(modes_dim)
        self.branch_dmd_dynamics = self.make_mlp(dynamics_dim)
        
        self.trunk = self.make_mlp(trunk_dim)
        
        self.final_linear = nn.Linear(branch_dim[-1], output_dim)
        
    def make_mlp(self, dims):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)
    
    def forward(self, f, modes, dynamics, x):
        y_dmd = self.branch_dmd_modes(modes) * self.branch_dmd_dynamics(dynamics)
        y_br = self.branch(f) * y_dmd
        y_tr = self.trunk(x)
        y_out = y_br @ y_tr
        return self.final_linear(y_out)  