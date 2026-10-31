import torch
import torch.nn as nn
from torch.func import jvp
import numpy as np
import pandas as pd

import Equations as eq
import Painter

import os
os.makedirs("ckpt", exist_ok = True)


def derivate(net, s):
    tangent = torch.ones_like(s)
    
    def inner(z):
        return jvp(net, (z,), (tangent,))
    
    (f, df), (_, d2f) = jvp(inner, (s,), (tangent,))
    
    return f, df, d2f

def save_ckpt(path, epoch):
    torch.save({"epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": {"hidden": hidden, "latent_dim": latent_dim, "rank": rank}},
               path)

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
    p_ic, p_bc, p_res = 51, 51, 31
    
    latent_dim, rank, hidden = 50, 50, 50

    n_epochs = 150000
    log_every = 1000
    learning_rate = 1e-3
    
    print("Generating dataset")
    datasets = []
    for a in alphas:
        datasets.append(eq.HeatEquationDataset(equation_class = eq.HeatEquation2D,
                                               num_samples = n_samples, spatial_size = (dim_malla, dim_malla),
                                               total_time = total_time, temporal_size = t_size,
                                               alpha = a, mode = "CONST"))
        
    
    system_painter = Painter.HeatEquationPainter()
    #system_painter.plot_solution(datasets[0][10], 10, "Testing draw")
    
    print("Processing test data")
    tests = []
    for a, dataset in zip(alphas, datasets):
        for sol in dataset:
            T0 = float(sol[0, 1, 1])
            sol_tensor = torch.tensor(sol, dtype = torch.float32, device = device)
            tests.append((T0, a, sol_tensor))

    print("Building training points")
    lin = lambda n: torch.linspace(0, 1, n, device = device)[:, None]
    alpha_grid = lambda n: torch.logspace(-1, 0, n, device = device)[:, None]

    u_tr = lin(n_train)
    B = n_train

    x_ic = y_ic = lin(p_ic + 2)[1:-1]
    t_ic = torch.zeros(1, 1, device = device)
    a_ic = alpha_grid(p_ic)

    x_edge = torch.tensor([[0.0], [1.0]], device = device)
    x_bc = y_bc = t_bc = lin(p_bc)
    a_bc = alpha_grid(p_bc)

    x_r = y_r = t_r = lin(p_res)
    a_r = alpha_grid(p_res)

    x_te = y_te = t_te = lin(dim_malla)

    model = SEP_DeepONet(branch_dim = [1, hidden, hidden, hidden, hidden, hidden, latent_dim],
                         trunk_dim = [1, hidden, hidden, hidden, hidden, hidden, latent_dim],
                         rank = rank).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9**(1/2000))

    def test_error():
        errs = []
        with torch.no_grad():
            for T0, a, s in tests:
                u0 = torch.tensor([[T0]], device = device)
                a_t = torch.tensor([[a]], device = device)
                pred = model(u0, x_te, y_te, t_te, a_t)[0, :, :, :, 0].permute(2, 0, 1)
                errs.append(torch.linalg.norm(s - pred) / torch.linalg.norm(s))
        return torch.stack(errs).mean().item()

    train_losses, ic_losses, bc_losses, res_losses, val_losses, log_epochs = [], [], [], [], [], []
    
    best_val = float("inf")
    print("Starting training")
    for epoch in range(n_epochs):
        model.train()

        pred_ic = model(u_tr, x_ic, y_ic, t_ic, a_ic)
        loss_ic = ((pred_ic - u_tr.view(B, 1, 1, 1, 1)) ** 2).mean()

        loss_bc = 0.5 * (model(u_tr, x_edge, y_bc, t_bc, a_bc).pow(2).mean()
                         + model(u_tr, x_bc, x_edge, t_bc, a_bc).pow(2).mean())

        f_init = model.branch_net(u_tr)
        fx, _, fx2 = derivate(model.trunk_x, x_r)
        fy, _, fy2 = derivate(model.trunk_y, y_r)
        ft, ft1, _ = derivate(model.trunk_t, t_r)
        fa = model.trunk_alpha(a_r)
        fa_alpha = a_r * fa

        u_t = model.contract(f_init, fx, fy, ft1, fa)
        u_xx = model.contract(f_init, fx2, fy, ft, fa_alpha)
        u_yy = model.contract(f_init, fx, fy2, ft, fa_alpha)
        loss_res = ((u_t - u_xx - u_yy) ** 2).mean()

        loss = loss_ic + loss_bc + 4 * loss_res

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % log_every == 0:
            save_ckpt("ckpt/last.pt", epoch + 1)
            model.eval()
            
            val_loss = test_error()
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt("ckpt/best.pt", epoch + 1)

            log_epochs.append(epoch)
            train_losses.append(loss.item())
            ic_losses.append(loss_ic.item())
            bc_losses.append(loss_bc.item())
            res_losses.append(loss_res.item())
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{n_epochs} - train {loss.item():.4e} - ic {loss_ic.item():.4e} - "
                  f"bc {loss_bc.item():.4e} - res {loss_res.item():.4e} - L2 {val_loss:.4e}")
            
            pd.DataFrame({"Epoch": log_epochs, "Train Loss": train_losses, "IC Loss": ic_losses,
                          "BC Loss": bc_losses, "Res Loss": res_losses,
                          "Test L2 Error": val_losses}).to_csv("ckpt/log.csv", index = False)

    save_ckpt("ckpt/final.pt", n_epochs)
    table = pd.DataFrame({"Epoch": log_epochs,
                          "Train Loss": train_losses,
                          "IC Loss": ic_losses,
                          "BC Loss": bc_losses,
                          "Res Loss": res_losses,
                          "Test L2 Error": val_losses})
    table.to_csv("ckpt/log.csv", index = False)
    print(table.to_string(index = False))