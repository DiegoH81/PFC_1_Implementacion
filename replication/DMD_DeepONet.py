import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import utils.Equations as eq

from models.DMD_DeepONet import DMD_DeepONet, DMDPart


if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    n_samples = 1000
    dim_malla = 10
    t_size = 50
    alpha = 0.5
    total_time = 0.05
    
    n_epochs = 100
    log_every = 10
    learning_rate = 1e-3
    
    print("Generating dataset")
    dataset = eq.HeatEquationDataset( equation_class = eq.HeatEquation2D,
                                     num_samples = n_samples, spatial_size = (dim_malla, dim_malla),
                                     total_time = total_time, temporal_size = t_size,
                                     alpha = alpha, mode = "NORMAL" )


    print("Adding DMD")
    processed_data = []
    
    for sol in dataset:
        init_state = torch.tensor(sol[0], dtype = torch.float32)
        end_state = torch.tensor(sol[-1], dtype = torch.float32)
        
        sol_tensor = torch.tensor(sol, dtype = torch.float32)
        dmd_processor = DMDPart(sol, rank = 10)
        
        modes_list, dynamics_list = dmd_processor.method()
        
        modes_tensor = torch.tensor(np.array(modes_list), dtype = torch.float32)
        dynamics_tensor = torch.tensor(np.array(dynamics_list), dtype = torch.float32)
        
        processed_data.append({
            "init_state": init_state,
            "end_state": end_state,
            "modes": modes_tensor,
            "dynamics": dynamics_tensor
        })
        
    init_all = torch.stack([d["init_state"] for d in processed_data])
    end_all = torch.stack([d["end_state"]  for d in processed_data])
    modes_all = torch.stack([d["modes"]     for d in processed_data])
    dyn_all = torch.stack([d["dynamics"]  for d in processed_data])

    full_ds = TensorDataset(init_all, modes_all, dyn_all, end_all)
    n_train = int(0.8 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [n_train, len(full_ds) - n_train])

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_ds, batch_size = batch_size, shuffle = False)
    
    model = DMD_DeepONet( branch_dim = [dim_malla ** 2, 100, 100],
                          modes_dim = [dim_malla ** 2, 100, 100],
                          dynamics_dim = [t_size, 100, 100],
                          trunk_dim = [2, 100, 100] ).to(device)

    xx = torch.linspace(0, 1, dim_malla)
    gx, gy = torch.meshgrid(xx, xx, indexing="ij")
    grid = torch.stack([gx.ravel(), gy.ravel()], dim=1).to(device)

    def predict(f, modes, dyn):
        return model(f, modes, dyn, grid).reshape(-1, dim_malla, dim_malla)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, factor = 0.5)


    best_val = float("inf")
    train_losses = []
    
    only_1_losses = []
    test_losses = []
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
        if (epoch) % log_every == 0:
            print(f"epoch {epoch}")
            
            model.eval()
            with torch.no_grad():
                # 1 SAMPLE
                f1, m1, d1, y1 = val_ds[np.random.randint(len(val_ds))]
                pred1 = predict(f1.unsqueeze(0).to(device),
                                m1.unsqueeze(0).to(device),
                                d1.unsqueeze(0).to(device))
                test_1 = ((pred1 - y1.unsqueeze(0).to(device)) ** 2).mean().item()

                # all
                vtot = 0.0
                vrel = 0.0
                vcnt = 0

                for f, modes, dyn, y in val_loader:
                    f, modes, dyn, y = f.to(device), modes.to(device), dyn.to(device), y.to(device)
                    
                    pred = predict(f, modes, dyn)
                    vtot += ((pred - y) ** 2).mean().item()
                    vrel += ((pred - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)).sum().item()
                    vcnt += y.shape[0]
                    
                val_full = vtot / len(val_loader)
                val_rel = vrel / vcnt

            train_losses.append(train_loss)
            only_1_losses.append(test_1)
            test_losses.append(val_full)
            relative_loss.append(val_rel)

        
    idx = list(range(0, n_epochs, 10))
    table = pd.DataFrame({"Epoch": idx,
                          "Train Loss": train_losses,
                          "Test Loss": test_losses,
                          "Test 1 Loss": only_1_losses,
                          "L2 error: ": relative_loss})
    print(table.to_string(index = False))