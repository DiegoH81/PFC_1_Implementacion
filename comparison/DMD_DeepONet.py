import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

from utils.Paths import COMP_WEIGHTS, COMP_LOGS, make_dirs
from comparison.common_scenario import ALPHA_MIN, ALPHA_MAX, DIM_MESH, T_SIZE, make_dataset, get_val_set
from models.DMD_DeepONet import DMD_DeepONet, get_relevant_data


def build_tensors(sols, k):
    processed_data = []
    
    for sol in sols:
        f, modes, dyn = get_relevant_data(sol, k)
        
        processed_data.append({ "init_state": torch.tensor(f, dtype = torch.float32),
                                "end_state": torch.tensor(sol[-1], dtype = torch.float32), 
                                "modes": torch.tensor(modes, dtype = torch.float32),
                                "dynamics": torch.tensor(dyn, dtype = torch.float32)
                              })
    
    init_all = torch.stack([d["init_state"] for d in processed_data])
    modes_all = torch.stack([d["modes"] for d in processed_data])
    dyn_all = torch.stack([d["dynamics"] for d in processed_data])
    end_all = torch.stack([d["end_state"] for d in processed_data])
    
    return init_all, modes_all, dyn_all, end_all

if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    make_dirs()
    
    name = "DMD"
    
    k = 30
    dim_malla = DIM_MESH
    hidden = 100
    
    n_alphas = 40
    n_per_alpha = 25
    
    n_epochs = 200
    log_every = 10
    learning_rate = 1e-3
    batch_size = 32
    
    print("Generating dataset")
    rng = np.random.default_rng(0)
    train_alphas = np.exp(rng.uniform(np.log(ALPHA_MIN), np.log(ALPHA_MAX), n_alphas))
    
    train_sols = []
    for a in train_alphas:
        dataset = make_dataset(float(a), n_per_alpha)
        for sol in dataset:
            train_sols.append(sol)
    
    val_sols = []
    for dataset in get_val_set().values():
        for sol in dataset:
            val_sols.append(sol)
    
    print("Adding DMD")
    init_all, modes_all, dyn_all, end_all = build_tensors(train_sols, k)
    train_ds = TensorDataset(init_all, modes_all, dyn_all, end_all)
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    
    f_val, modes_val, dyn_val, y_val = build_tensors(val_sols, k)
    f_val, modes_val, dyn_val, y_val = f_val.to(device), modes_val.to(device), dyn_val.to(device), y_val.to(device)
    
    model = DMD_DeepONet( branch_dim = [dim_malla ** 2, hidden, hidden],
                          modes_dim = [dim_malla ** 2, hidden, hidden],
                          dynamics_dim = [k, hidden, hidden],
                          trunk_dim = [2, hidden, hidden, hidden, hidden] ).to(device)

    xx = torch.linspace(0, 1, dim_malla)
    gx, gy = torch.meshgrid(xx, xx, indexing="ij")
    grid = torch.stack([gx.ravel(), gy.ravel()], dim=1).to(device)

    def predict(f, modes, dyn):
        return model(f, modes, dyn, grid).reshape(-1, dim_malla, dim_malla)

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 20, factor = 0.5, min_lr = 1e-6)

    best_val = float("inf")
    log_epochs = []
    train_losses = []
    val_losses = []
    relative_loss = []
    
    print("Starting training")
    for epoch in range(n_epochs):
        model.train()
        total = 0.0
        for f, modes, dyn, y in train_loader:
            f, modes, dyn, y = f.to(device), modes.to(device), dyn.to(device), y.to(device)

            optimizer.zero_grad()
            pred = predict(f, modes, dyn)
            loss = ((pred - y) ** 2).mean()
            loss.backward()
            optimizer.step()
            total += loss.item()

        train_loss = total / len(train_loader)
        scheduler.step(train_loss)

        # Val
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                pred = predict(f_val, modes_val, dyn_val)
                val_full = ((pred - y_val) ** 2).mean().item()
                val_rel = ((pred - y_val).flatten(1).norm(dim=1) / y_val.flatten(1).norm(dim=1)).mean().item()
            
            model.save(f"{COMP_WEIGHTS}/{name}_last.pt", epoch + 1, optimizer, scheduler)
            if val_rel < best_val:
                best_val = val_rel
                model.save(f"{COMP_WEIGHTS}/{name}_best.pt", epoch + 1, optimizer, scheduler)

            log_epochs.append(epoch + 1)
            train_losses.append(train_loss)
            val_losses.append(val_full)
            relative_loss.append(val_rel)
            
            table = pd.DataFrame({"Epoch": log_epochs,
                                  "Train Loss": train_losses,
                                  "Val Loss": val_losses,
                                  "Val L2": relative_loss})
            
            table.to_csv(f"{COMP_LOGS}/log_{name}.csv", index = False)
            print(f"Epoch {epoch + 1}/{n_epochs} - train {train_loss:.4e} - val {val_full:.4e} - val L2 {val_rel:.4e}")

    print(table.to_string(index = False))