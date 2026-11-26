import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from utils.Paths import COMP_WEIGHTS, COMP_LOGS, make_dirs
from utils.Metrics import rel_l2
from comparison.common_scenario import ALPHA_MIN, ALPHA_MAX, DIM_MESH, T_SIZE, TOTAL_TIME, get_val_set
from models.SEP_DeepONet import SEP_DeepONet, derivate


if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    make_dirs()
    
    name = "SEP_comp"
    
    n_train = 25
    p_ic, p_bc, p_res = 51, 51, 31
    
    latent_dim, rank, hidden = 50, 50, 50
    
    n_epochs = 150000
    log_every = 1000
    learning_rate = 1e-3
    
    print("Generating dataset")
    datasets = get_val_set()

    print("Processing validation data")
    vals = []
    for dataset in datasets.values():
        alpha = dataset.kwargs["alpha"]
        for sol in dataset:
            T0 = float(sol[0][1][1])
            sol_tensor = torch.tensor(sol, dtype = torch.float32, device = device)
            vals.append((T0, alpha, sol_tensor))
    
    print("Building training points")
    lin = lambda n: torch.linspace(0, 1, n, device = device)[:, None]
    alpha_grid = lambda n: torch.logspace(np.log10(ALPHA_MIN), np.log10(ALPHA_MAX), n, device = device)[:, None]

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

    x_te = y_te = lin(DIM_MESH)
    t_te = lin(T_SIZE)

    model = SEP_DeepONet(branch_dim = [1, hidden, hidden, hidden, hidden, hidden, latent_dim],
                         trunk_dim = [1, hidden, hidden, hidden, hidden, hidden, latent_dim],
                         rank = rank).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9**(1/2000))

    def predict(T0, a):
        u0 = torch.tensor([[T0]], device = device)
        a_t = torch.tensor([[a]], device = device)
        return model(u0, x_te, y_te, t_te, a_t)[0, :, :, :, 0].permute(2, 0, 1)

    def val_error():
        errs = []
        with torch.no_grad():
            for T0, a, s in vals:
                errs.append(rel_l2(predict(T0, a), s))
        return float(np.mean(errs))

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
        fa_alpha = TOTAL_TIME * a_r * fa

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
            model.save(f"{COMP_WEIGHTS}/{name}_last.pt", epoch + 1, optimizer, scheduler)
            model.eval()
            
            val_loss = val_error()
            if val_loss < best_val:
                best_val = val_loss
                model.save(f"{COMP_WEIGHTS}/{name}_best.pt", epoch + 1, optimizer, scheduler)

            log_epochs.append(epoch)
            train_losses.append(loss.item())
            ic_losses.append(loss_ic.item())
            bc_losses.append(loss_bc.item())
            res_losses.append(loss_res.item())
            val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{n_epochs} - train {loss.item():.4e} - ic {loss_ic.item():.4e} - "
                  f"bc {loss_bc.item():.4e} - res {loss_res.item():.4e} - L2 {val_loss:.4e}")
            
            pd.DataFrame({"Epoch": log_epochs, "Train Loss": train_losses, "IC Loss": ic_losses,
                          "BC Loss": bc_losses, "Res Loss": res_losses,
                          "Val L2 Error": val_losses}).to_csv(f"{COMP_LOGS}/log_{name}.csv", index = False)

    model.save(f"{COMP_WEIGHTS}/{name}_final.pt", n_epochs, optimizer, scheduler)
    
    table = pd.DataFrame({"Epoch": log_epochs,
                          "Train Loss": train_losses,
                          "IC Loss": ic_losses,
                          "BC Loss": bc_losses,
                          "Res Loss": res_losses,
                          "Val L2 Error": val_losses})
    table.to_csv(f"{COMP_LOGS}/log_{name}.csv", index = False)
    print(table.to_string(index = False))
