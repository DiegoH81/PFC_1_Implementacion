import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
import numpy as np
import pandas as pd

from models.SEP_DeepONet import derivate
from common import (ALPHA_MIN, ALPHA_MAX, DIM_MALLA, T_SIZE, TOTAL_TIME, OUT_DIR, SEP_CKPT,
                    SEP_HIDDEN, SEP_LATENT, SEP_RANK, build_sep,
                    get_test_set, get_val_set, rel_l2, field_metrics, report)


if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok = True)

    n_train = 25
    p_ic, p_bc, p_res = 51, 51, 31
    n_epochs = 150000
    log_every = 1000
    learning_rate = 1e-3

    def save_ckpt(path, epoch):
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "config": {"hidden": SEP_HIDDEN, "latent_dim": SEP_LATENT, "rank": SEP_RANK,
                               "total_time": TOTAL_TIME}}, path)

    vals = [(float(s["sol"][0, 1, 1]), s["alpha"], s["sol"].to(device)) for s in get_val_set()]

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

    x_te = y_te = lin(DIM_MALLA)
    t_te = lin(T_SIZE)

    model = build_sep(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9**(1/2000))

    def predict(T0, a):
        u0 = torch.tensor([[T0]], device = device)
        a_t = torch.tensor([[a]], device = device)
        return model(u0, x_te, y_te, t_te, a_t)[0, :, :, :, 0].permute(2, 0, 1)

    def val_error():
        with torch.no_grad():
            return float(np.mean([rel_l2(predict(T0, a), s) for T0, a, s in vals]))

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
            model.eval()
            val_loss = val_error()
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(SEP_CKPT, epoch + 1)

            log_epochs.append(epoch)
            train_losses.append(loss.item())
            ic_losses.append(loss_ic.item())
            bc_losses.append(loss_bc.item())
            res_losses.append(loss_res.item())
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{n_epochs} - train {loss.item():.4e} - ic {loss_ic.item():.4e} - "
                  f"bc {loss_bc.item():.4e} - res {loss_res.item():.4e} - val L2 {val_loss:.4e}")

            pd.DataFrame({"Epoch": log_epochs, "Train Loss": train_losses, "IC Loss": ic_losses,
                          "BC Loss": bc_losses, "Res Loss": res_losses,
                          "Val L2 Error": val_losses}).to_csv(f"{OUT_DIR}/log_SEP.csv", index = False)

    
    model.load_state_dict(torch.load(SEP_CKPT, map_location = device)["model"])
    model.eval()
    rows = []
    with torch.no_grad():
        for s in get_test_set():
            sol = s["sol"].to(device)
            T0 = float(sol[0, 1, 1])
            pred = predict(T0, s["alpha"])
            rows.append({"material": s["material"], "alpha": s["alpha"], "T0": T0,
                         **field_metrics(pred[-1].cpu(), sol[-1].cpu(), "final"),
                         **field_metrics(pred.cpu(), sol.cpu(), "full")})
    report(rows, "SEP")
