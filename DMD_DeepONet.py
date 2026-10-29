import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from pydmd import DMD
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


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
    
    def forward(self, f, modes, dynamics, x):
        y_dmd = self.branch_dmd_modes(modes) * self.branch_dmd_dynamics(dynamics)
        y_br = self.branch(f) * y_dmd
        y_tr = self.trunk(x)
        y_out = y_br @ y_tr
        return self.final_linear(y_out)  
    
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

    system_painter = Painter.HeatEquationPainter()
    #system_painter.plot_solution(dataset[10], 10, "Testing draw")
    
    
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
    
    model = DMD_DeepONet( branch_dim = [dim_malla, 100, 100],
                          branch_dmd_dim_modes = [100, 100, 100],
                          branch_dmd_dim_dynamics = [50, 100, 100],
                          trunk_dim = [2, 100, 100], output_dim = dim_malla).to(device)

    xx = torch.linspace(0, 1, dim_malla)
    gx, gy = torch.meshgrid(xx, xx, indexing="ij")
    grid = torch.stack([gx.ravel(), gy.ravel()], dim=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, factor = 0.5)


    best_val = float("inf")
    train_losses, val_losses = [], []
    print("Starting training")
    for epoch in range(n_epochs):
        model.train()
        total = 0.0
        for f, modes, dyn, y in train_loader:
            f, modes, dyn, y = f.to(device), modes.to(device), dyn.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(f, modes, dyn, grid)
            loss = ((pred - y) ** 2).mean()
            loss.backward()
            optimizer.step()
            total += loss.item()

        train_loss = total / len(train_loader)
        scheduler.step(train_loss)

        # Val
        if (epoch + 1) % log_every == 0:
            model.eval()
            with torch.no_grad():
                vtot = 0.0
                for f, modes, dyn, y in val_loader:
                    f, modes, dyn, y = f.to(device), modes.to(device), dyn.to(device), y.to(device)
                    pred = model(f, modes, dyn, grid)
                    vtot += ((pred - y) ** 2).mean().item()

                val_loss = vtot / len(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"Epoch {epoch + 1}/{n_epochs} - train {train_loss:.4f} - val {val_loss:.4f}")
        
    idx = list(range(0, n_epochs, 10))
    table = pd.DataFrame({"Epoch": idx,
                          "Train Loss": train_losses,
                          "Test Loss": val_losses})
    print(table.to_string(index = False))