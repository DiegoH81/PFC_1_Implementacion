import torch
import torch.nn as nn
from torch.func import jvp
import numpy as np
import pandas as pd



def derivate(net, s):
    tangent = torch.ones_like(s)
    
    def inner(z):
        return jvp(net, (z,), (tangent,))
    
    (f, df), (_, d2f) = jvp(inner, (s,), (tangent,))
    
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
    
    def contract(self, f_init, f_x, f_y, f_t, f_a):
        p = self.latent_dim
        r = self.rank
        
        f_x = f_x.reshape(-1, p, r)
        f_y = f_y.reshape(-1, p, r)
        f_t = f_t.reshape(-1, p, r)
        f_a = f_a.reshape(-1, p, r)

        bx  = torch.einsum('bp,xpr->bxpr', f_init, f_x)
        yta = torch.einsum('ypr,tpr,apr->ytapr', f_y, f_t, f_a)
        return torch.einsum('bxpr,ytapr->bxyta', bx, yta)

    def forward(self, init, x, y, t, alpha):
        return self.contract(self.branch_net(init), self.trunk_x(x), self.trunk_y(y),
                             self.trunk_t(t), self.trunk_alpha(alpha)) + self.output_bias