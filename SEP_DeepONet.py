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
    
    def forward(self, initial_temp, t, x, y, alpha):
        p, r = self.latent_dim, self.rank
        num_t, num_x, num_y, num_alpha = len(t), len(x), len(y), len(alpha)

        y_br = self.branch_net(initial_temp)

        y_t = self.trunk_t(t).view(-1, r, p).permute(2, 0, 1)
        y_x = self.trunk_x(x).view(-1, r, p).permute(2, 0, 1)
        y_y = self.trunk_y(y).view(-1, r, p).permute(2, 0, 1)
        y_alpha = self.trunk_alpha(alpha).view(-1, r, p).permute(2, 0, 1)


        y_tx = (y_t[:, :, None, :] * y_x[:, None, :, :]).reshape(p, num_t * num_x, r)
        y_ya = (y_y[:, :, None, :] * y_alpha[:, None, :, :]).reshape(p, num_y * num_alpha, r)

        y_tr = y_tx @ y_ya.transpose(1, 2)
        y_tr = y_tr.reshape(p, -1)

        y_out = y_br @ y_tr
        return y_out.view(-1, num_t, num_x, num_y, num_alpha) + self.output_bias