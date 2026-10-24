import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from pydmd import DMD
import numpy as np
import pandas as pd

import Equations as eq
import Painter

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