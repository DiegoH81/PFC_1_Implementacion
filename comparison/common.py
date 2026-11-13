import os
import numpy as np
import pandas as pd
import torch

import utils.Equations as eq
from models.DMD_DeepONet import DMDPart
from models.DMD_DeepONet_v2 import DMD_DeepONet_v2
from models.SEP_DeepONet import SEP_DeepONet

ALPHA_MIN, ALPHA_MAX = 0.01, 1.0
DIM_MALLA = 10
T_SIZE = 50
TOTAL_TIME = 0.1
MODE = "CONST"

L_PLACA = 0.10
T_END = 9.0
MATERIALS_MM2S = {
    "Cobre":        111.0,
    "Aluminio":      97.0,
    "Hierro":        23.0,
    "Aire (300 K)":  19.0,
    "Acero 1% C":    11.72,
    "Titanio":        9.3,
    "Inox 304":       4.2,
}
MATERIALS = {name: a * 1e-6 * T_END / (L_PLACA ** 2 * TOTAL_TIME) for name, a in MATERIALS_MM2S.items()}
N_TEST_PER_MATERIAL = 20
N_VAL = 50


OUT_DIR = "OUT_COMPARISON"
TEST_PATH = f"{OUT_DIR}/test_set.pt"
VAL_PATH = f"{OUT_DIR}/val_set.pt"
SEP_CKPT = f"{OUT_DIR}/SEP_best.pt"
DMD_CKPT = f"{OUT_DIR}/DMD_best.pt"
FIG_DIR = f"{OUT_DIR}/figs"


DMD_RANK = 10
SEP_HIDDEN, SEP_LATENT, SEP_RANK = 50, 50, 50


def build_dmd(device):
    n = DIM_MALLA ** 2
    return DMD_DeepONet_v2(branch_dim = [n, 100, 100], modes_dim = [n, 100, 100],
                           dynamics_dim = [T_SIZE, 100, 100],
                           trunk_dim = [2, 100, 100, 100, 100]).to(device)


def build_sep(device, hidden = SEP_HIDDEN, latent = SEP_LATENT, rank = SEP_RANK):
    h = hidden
    return SEP_DeepONet(branch_dim = [1, h, h, h, h, h, latent],
                        trunk_dim = [1, h, h, h, h, h, latent], rank = rank).to(device)


def load_dmd(device):
    net = build_dmd(device)
    net.load_state_dict(torch.load(DMD_CKPT, map_location = device))
    return net.eval()


def load_sep(device):
    ck = torch.load(SEP_CKPT, map_location = device)
    c = ck["config"]
    net = build_sep(device, c["hidden"], c["latent_dim"], c["rank"])
    net.load_state_dict(ck["model"])
    return net.eval()


def dmd_inputs(sol):
    modes, dyn = DMDPart(sol, rank = DMD_RANK).method()
    for _ in range(len(modes), DMD_RANK):
        modes.append(np.zeros_like(modes[0]))
        dyn.append(np.zeros_like(dyn[0]))
    return (torch.from_numpy(sol[0]),
            torch.tensor(np.array(modes), dtype = torch.float32),
            torch.tensor(np.array(dyn), dtype = torch.float32),
            torch.from_numpy(sol[-1]))


def make_solutions(alpha, n):
    ds = eq.HeatEquationDataset(equation_class = eq.HeatEquation2D, num_samples = n,
                                spatial_size = (DIM_MALLA, DIM_MALLA), total_time = TOTAL_TIME,
                                temporal_size = T_SIZE, alpha = alpha, mode = MODE)
    return [np.asarray(s, dtype = np.float32) for s in ds]


def sample_train_alphas(n, seed = 0):
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(np.log(ALPHA_MIN), np.log(ALPHA_MAX), n))


def _cached(path, seed, build):
    if os.path.exists(path):
        return torch.load(path)
    os.makedirs(OUT_DIR, exist_ok = True)
    np_state, torch_state = np.random.get_state(), torch.get_rng_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = build()
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    torch.save(data, path)
    return data


def get_test_set():
    def build():
        return [{"material": name, "alpha": float(a), "sol": torch.from_numpy(sol)}
                for name, a in MATERIALS.items() for sol in make_solutions(a, N_TEST_PER_MATERIAL)]
    return _cached(TEST_PATH, 2026, build)


def get_val_set():
    def build():
        return [{"alpha": float(a), "sol": torch.from_numpy(make_solutions(float(a), 1)[0])}
                for a in sample_train_alphas(N_VAL, seed = 777)]
    return _cached(VAL_PATH, 777, build)


# Metrics for eval
def rel_l2(pred, true):
    return (torch.linalg.norm(pred - true) / torch.linalg.norm(true)).item()


def field_metrics(pred, true, prefix):
    err = pred - true
    return {f"L2_{prefix}": rel_l2(pred, true),
            f"MSE_{prefix}": err.pow(2).mean().item(),
            f"Emax_{prefix}": err.abs().max().item()}


def report(rows, name):
    os.makedirs(OUT_DIR, exist_ok = True)
    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c not in ("material", "alpha", "T0")]
    order = list(MATERIALS.keys())
    summary = df.groupby("material")[cols].mean().reindex(order)
    summary.insert(0, "alpha", [MATERIALS[m] for m in order])
    summary.loc["Promedio"] = [np.nan] + list(df[cols].mean().values)

    pd.set_option("display.width", 200)
    print(f"\n===== {name} | Escenario A | {len(df)} trayectorias =====")
    print(summary.to_string(float_format = lambda v: f"{v:.4e}" if abs(v) < 1e-2 else f"{v:.4f}"))
    df.to_csv(f"{OUT_DIR}/results_{name}_detalle.csv", index = False)
    summary.to_csv(f"{OUT_DIR}/results_{name}_resumen.csv")
    return summary
