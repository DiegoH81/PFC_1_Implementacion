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
    def __init__(self, branch_dim, modes_dim, dynamics_dim, trunk_dim):
        super().__init__()
        self.config = dict(branch_dim = branch_dim, modes_dim = modes_dim,
                           dynamics_dim = dynamics_dim, trunk_dim = trunk_dim)
        
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

    def forward(self, f, modes, dynamics, coords):
        b0 = self.branch(f.flatten(1))
        y_dmd = (self.branch_dmd_modes(modes)
                 * self.branch_dmd_dynamics(dynamics)).sum(dim = 1)
        y_tr = self.trunk(coords)
        return (b0 * y_dmd) @ y_tr.T + self.bias
    
    def save(self, path, epoch, optimizer, scheduler):
        torch.save({"epoch": epoch, "model": self.state_dict(), "config": self.config,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()}, path)

    @classmethod
    def load(cls, path, device = "cpu"):
        checkpoint = torch.load(path, map_location = device)
        net = cls(**checkpoint["config"]).to(device)
        net.load_state_dict(checkpoint["model"])
        return net.eval()