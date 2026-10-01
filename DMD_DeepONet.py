import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from pydmd import DMD


import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict


import Equations as eq
import Painter

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
    def __init__(self, branch_dim, branch_dmd_dim_modes, branch_dmd_dim_dynamics, trunk_dim, output_dim):
        super().__init__()
        
        self.branch = self.make_mlp(branch_dim)
        
        self.branch_dmd_modes = self.make_mlp(branch_dmd_dim_modes)
        self.branch_dmd_dynamics = self.make_mlp(branch_dmd_dim_dynamics)
        
        self.trunk = self.make_mlp(trunk_dim)
        
        self.final_linear = nn.Linear(branch_dim[-1], output_dim)
        
    def make_mlp(self, dims):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)
    
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    n_samples = 1000
    dim_malla = 10
    t_size = 50
    alpha = 0.5
    total_time = 0.05
    
    n_epochs = 150
    learning_rate = 1e-3
    
    print("Generating dataset")
    dataset = eq.HeatEquationDataset( equation_class = eq.HeatEquation2D,
                                     num_samples = n_samples, spatial_size=(dim_malla, dim_malla),
                                     total_time = total_time, temporal_size = t_size,
                                     alpha = alpha )

    system_painter = Painter.HeatEquationPainter()
    system_painter.plot_solution(dataset[10], 10, "Testing draw")
    
    
    print("Adding DMD")
    processed_data = []
    
    for sol in dataset:
        init_state = torch.tensor(sol[0], dtype = torch.float32).flatten()
        end_state = torch.tensor(sol[-1], dtype = torch.float32).flatten()
        
        sol_tensor = torch.tensor(sol, dtype = torch.float32)
        dmd_processor = DMDPart(sol, rank = 10)
        
        modes_list, dynamics_list = dmd_processor.method()
        
        modes_tensor = torch.tensor(np.array(modes_list), dtype = torch.float32).flatten()
        dynamics_tensor = torch.tensor(np.array(dynamics_list), dtype = torch.float32).flatten()
        
        processed_data.append({
            "init_state": init_state,
            "end_state": end_state,
            "modes": modes_tensor,
            "dynamics": dynamics_tensor
        })