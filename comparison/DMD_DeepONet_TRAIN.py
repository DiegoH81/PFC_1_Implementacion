import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from common import (DIM_MALLA, OUT_DIR, DMD_CKPT, build_dmd, dmd_inputs,
                    make_solutions, sample_train_alphas, get_test_set, field_metrics, report)


if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok = True)

    n_train_alphas = 40
    n_per_alpha = 25
    n_epochs = 300
    log_every = 10
    learning_rate = 1e-3
    batch_size = 32

    print("Generating train dataset")
    samples = [dmd_inputs(sol)
               for a in sample_train_alphas(n_train_alphas, seed = 0)
               for sol in make_solutions(float(a), n_per_alpha)]

    full_ds = TensorDataset(*(torch.stack(x) for x in zip(*samples)))
    n_train = int(0.8 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [n_train, len(full_ds) - n_train])
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_ds, batch_size = batch_size, shuffle = False)

    net = build_dmd(device)

    xx = torch.linspace(0, 1, DIM_MALLA)
    gx, gy = torch.meshgrid(xx, xx, indexing = "ij")
    grid = torch.stack([gx.ravel(), gy.ravel()], dim = 1).to(device)

    def model(f, modes, dyn):
        return net(f, modes, dyn, grid).reshape(-1, DIM_MALLA, DIM_MALLA)

    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, factor = 0.5)

    log_epochs, train_losses, val_losses, val_rel = [], [], [], []
    best_val = float("inf")

    print("Starting training")
    for epoch in range(n_epochs):
        net.train()
        total = 0.0
        for f, modes, dyn, y in train_loader:
            f, modes, dyn, y = f.to(device), modes.to(device), dyn.to(device), y.to(device)
            optimizer.zero_grad()
            loss = ((model(f, modes, dyn) - y) ** 2).mean()
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_loss = total / len(train_loader)
        scheduler.step(train_loss)

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            net.eval()
            vtot, vrel, vcnt = 0.0, 0.0, 0
            with torch.no_grad():
                for f, modes, dyn, y in val_loader:
                    f, modes, dyn, y = f.to(device), modes.to(device), dyn.to(device), y.to(device)
                    pred = model(f, modes, dyn)
                    vtot += ((pred - y) ** 2).mean().item()
                    vrel += ((pred - y).flatten(1).norm(dim = 1) / y.flatten(1).norm(dim = 1)).sum().item()
                    vcnt += y.shape[0]
            v_rel = vrel / vcnt
            if v_rel < best_val:
                best_val = v_rel
                torch.save(net.state_dict(), DMD_CKPT)

            log_epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(vtot / len(val_loader))
            val_rel.append(v_rel)
            print(f"Epoch {epoch+1}/{n_epochs} - train {train_loss:.4e} - "
                  f"val {val_losses[-1]:.4e} - val L2 {v_rel:.4e}")

    pd.DataFrame({"Epoch": log_epochs, "Train Loss": train_losses, "Val Loss": val_losses,
                  "Val L2": val_rel}).to_csv(f"{OUT_DIR}/log_DMD.csv", index = False)

    # ---------------- evaluación por material ----------------
    net.load_state_dict(torch.load(DMD_CKPT, map_location = device))
    net.eval()
    rows = []
    with torch.no_grad():
        for s in get_test_set():
            f, m, d, y = dmd_inputs(s["sol"].numpy())
            pred = model(f.unsqueeze(0).to(device), m.unsqueeze(0).to(device), d.unsqueeze(0).to(device))[0].cpu()
            rows.append({"material": s["material"], "alpha": s["alpha"], "T0": float(f[1, 1]),
                         **field_metrics(pred, y, "final")})
    report(rows, "DMD")
