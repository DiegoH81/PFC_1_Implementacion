import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

from utils.Paths import COMP_WEIGHTS, COMP_FIGS, make_dirs
from utils.Painter import Panel, Grid
from utils.Metrics import get_all_metrics, delta_rel, summarize
from comparison.common_scenario import MATERIALS, MATERIALS_alpha, DIM_MESH, get_test_set
from models.SEP_DeepONet import SEP_DeepONet
from models.DMD_DeepONet import DMD_DeepONet, get_relevant_data


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    make_dirs()

    sep_name = "SEP"
    dmd_name = "DMD"

    sep = SEP_DeepONet.load(f"{COMP_WEIGHTS}/{sep_name}_best.pt", device)
    dmd = DMD_DeepONet.load(f"{COMP_WEIGHTS}/{dmd_name}_best.pt", device)
    
    k = dmd.config["dynamics_dim"][0]

    xs = torch.linspace(0, 1, DIM_MESH, device = device)[:, None]
    t_end = torch.ones(1, 1, device = device)
    
    gx, gy = torch.meshgrid(xs[:, 0], xs[:, 0], indexing = "ij")
    grid = torch.stack([gx.ravel(), gy.ravel()], dim = 1)


    def predict_sep(T0, alpha):
        u0 = torch.tensor([[T0]], dtype = torch.float32, device = device)
        a = torch.tensor([[alpha]], dtype = torch.float32, device = device)
        return sep(u0, xs, xs, t_end, a)[0, :, :, 0, 0]

    def predict_dmd(sol):
        f, modes, dyn = get_relevant_data(sol, k)
        f = torch.tensor(f, dtype = torch.float32, device = device)[None]
        modes = torch.tensor(modes, dtype = torch.float32, device = device)[None]
        dyn = torch.tensor(dyn, dtype = torch.float32, device = device)[None]
        return dmd(f, modes, dyn, grid).reshape(DIM_MESH, DIM_MESH)


    order = sorted(MATERIALS, key = MATERIALS.get)
    test = get_test_set()

    rows = []
    mean_T0 = {}
    mean_gt = {}
    mean_sep = {}
    mean_dmd = {}
    mean_err_sep = {}
    mean_err_dmd = {}

    for name in order:
        n = len(test[name])
        mean_T0[name] = 0.0
        mean_gt[name] = np.zeros((DIM_MESH, DIM_MESH))
        mean_sep[name] = np.zeros((DIM_MESH, DIM_MESH))
        mean_dmd[name] = np.zeros((DIM_MESH, DIM_MESH))
        mean_err_sep[name] = np.zeros((DIM_MESH, DIM_MESH))
        mean_err_dmd[name] = np.zeros((DIM_MESH, DIM_MESH))

        for sol in test[name]:
            T0 = float(sol[0][1][1])
            gt = torch.tensor(sol[-1], dtype = torch.float32, device = device)

            with torch.no_grad():
                pred_sep = predict_sep(T0, MATERIALS[name])
                pred_dmd = predict_dmd(sol)

            row = {"material": name}
            row.update(get_all_metrics(pred_sep, gt, "SEP", u_max = T0))
            row.update(get_all_metrics(pred_dmd, gt, "DMD", u_max = T0))
            rows.append(row)

            gt = gt.cpu().numpy()
            pred_sep = pred_sep.cpu().numpy()
            pred_dmd = pred_dmd.cpu().numpy()
            mean_T0[name] += T0 / n
            mean_gt[name] += gt / n
            mean_sep[name] += pred_sep / n
            mean_dmd[name] += pred_dmd / n
            mean_err_sep[name] += np.abs(pred_sep - gt) / n
            mean_err_dmd[name] += np.abs(pred_dmd - gt) / n

    
    # data tables
    df = pd.DataFrame(rows)
    cols = ["L2_SEP", "L2_DMD", "MSE_SEP", "MSE_DMD", "Emax_SEP", "Emax_DMD"]
    table = summarize(df, cols, order)
    table["dL2"] = delta_rel(table["L2_DMD"], table["L2_SEP"])
    table["dMSE"] = delta_rel(table["MSE_DMD"], table["MSE_SEP"])

    pd.set_option("display.width", 200)
    print(f"\nAvg of {len(df)} simulations")
    print(table.to_string(float_format = "{:.4e}".format))


    # figures
    col_labels = []
    for name in order:
        col_labels.append(f"{name}\n" + rf"$\alpha_f$ = {MATERIALS_alpha[name]:g} mm$^2$/s")

    maps = Grid(3, len(order), ["Ground truth", "SEP-DeepONet", "DMD-DeepONet"], col_labels, r"$u/T_0$")
    errors = Grid(2, len(order), ["SEP-DeepONet", "DMD-DeepONet"], col_labels, r"$|\hat{u} - u|/T_0$")

    err_max = 0.0
    for j, name in enumerate(order):
        T0 = mean_T0[name]
        maps.set_panel(0, j, Panel(mean_gt[name] / T0, vmin = 0.0, vmax = 1.0))
        maps.set_panel(1, j, Panel(mean_sep[name] / T0, vmin = 0.0, vmax = 1.0))
        maps.set_panel(2, j, Panel(mean_dmd[name] / T0, vmin = 0.0, vmax = 1.0))
        err_max = max(err_max, (mean_err_sep[name] / T0).max(), (mean_err_dmd[name] / T0).max())

    for j, name in enumerate(order):
        T0 = mean_T0[name]
        errors.set_panel(0, j, Panel(mean_err_sep[name] / T0, cmap = "viridis", vmin = 0.0, vmax = 0.1))
        errors.set_panel(1, j, Panel(mean_err_dmd[name] / T0, cmap = "viridis", vmin = 0.0, vmax = 0.1))

    maps.save(f"{COMP_FIGS}/heat_map")
    errors.save(f"{COMP_FIGS}/heat_error")
    print("Figs saved at", COMP_FIGS)
