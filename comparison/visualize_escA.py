import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

from common import (DIM_MALLA, T_SIZE, FIG_DIR, MATERIALS, MATERIALS_MM2S,
                    load_sep, load_dmd, dmd_inputs, get_test_set, field_metrics)
from utils.Painter import HeatEquationPainter

T0_OBJETIVO = 0.5


def latex_num(v):
    if abs(v) >= 1e-2:
        return f"{v:.4f}"
    mant, exp = f"{v:.2e}".split("e")
    return rf"${mant}\times10^{{{int(exp)}}}$"


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(FIG_DIR, exist_ok = True)

    sep, dmd = load_sep(device), load_dmd(device)
    painter = HeatEquationPainter()

    lin = lambda n: torch.linspace(0, 1, n, device = device)[:, None]
    xs, ts = lin(DIM_MALLA), lin(T_SIZE)
    xx = torch.linspace(0, 1, DIM_MALLA)
    gx, gy = torch.meshgrid(xx, xx, indexing = "ij")
    grid = torch.stack([gx.ravel(), gy.ravel()], dim = 1).to(device)

    def predict_final(sol, a):
        T0 = float(sol[0, 1, 1])
        with torch.no_grad():
            u0 = torch.tensor([[T0]], device = device)
            at = torch.tensor([[a]], device = device)
            sep_final = sep(u0, xs, xs, ts, at)[0, :, :, -1, 0].T.cpu().numpy()

            f, m, d, _ = dmd_inputs(sol)
            dmd_final = dmd(f.unsqueeze(0).to(device), m.unsqueeze(0).to(device),
                            d.unsqueeze(0).to(device), grid)
            dmd_final = dmd_final.reshape(DIM_MALLA, DIM_MALLA).cpu().numpy()
        return T0, sep_final, dmd_final

    test = get_test_set()
    order = list(MATERIALS.keys())

    
    metric_rows = []
    for s in test:
        sol = s["sol"].numpy()
        T0, sep_f, dmd_f = predict_final(sol, s["alpha"])
        gt = torch.from_numpy(sol[-1])
        msep = field_metrics(torch.from_numpy(sep_f), gt, "final")
        mdmd = field_metrics(torch.from_numpy(dmd_f), gt, "final")
        metric_rows.append({"material": s["material"], "alpha": s["alpha"], "T0": T0,
                            "L2_SEP": msep["L2_final"], "L2_DMD": mdmd["L2_final"],
                            "MSE_SEP": msep["MSE_final"], "MSE_DMD": mdmd["MSE_final"],
                            "Emax_SEP": msep["Emax_final"], "Emax_DMD": mdmd["Emax_final"]})

    dfm = pd.DataFrame(metric_rows)
    cols = ["L2_SEP", "L2_DMD", "MSE_SEP", "MSE_DMD", "Emax_SEP", "Emax_DMD"]
    mean_t = dfm.groupby("material")[cols].mean().reindex(order)
    mean_t.loc["Promedio"] = dfm[cols].mean()

    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.4e}" if abs(v) < 1e-2 else f"{v:.4f}"
    print(f"\n===== Tabla (MEDIA sobre {len(dfm)} trayectorias) =====")
    print(mean_t.to_string(float_format = fmt))


    dfm.to_csv(f"{FIG_DIR}/metricas_detalle.csv", index = False)
    mean_t.to_csv(f"{FIG_DIR}/metricas_media.csv")

    # Figs
    samples = []
    for name in order:
        cand = [s for s in test if s["material"] == name]
        s = min(cand, key = lambda s: abs(float(s["sol"][0, 1, 1]) - T0_OBJETIVO))
        sol = s["sol"].numpy()
        T0, sep_f, dmd_f = predict_final(sol, s["alpha"])
        samples.append(dict(name = name, alpha = s["alpha"], T0 = T0,
                            gt = sol[-1] / T0, sep = sep_f / T0, dmd = dmd_f / T0))

    titles = [f"{r['name']}\n" + rf"$\alpha$ = {r['alpha']:.3f}" for r in samples]
    labels = ["Ground truth", "SEP-DeepONet", "DMD-DeepONet"]

    painter.plot_comparison_grid(
        fields = [[r[k] for r in samples] for k in ("gt", "sep", "dmd")],
        row_labels = labels, col_titles = titles,
        vmin = 0.0, vmax = 1.0, cbar_label = r"$u / T_0$",
        save_path = f"{FIG_DIR}/escA_mapas")

    jc = DIM_MALLA // 2
    painter.plot_profiles(
        x = np.linspace(0, 1, DIM_MALLA),
        curves = [[r["gt"][jc, :], r["sep"][jc, :], r["dmd"][jc, :]] for r in samples],
        titles = [t.replace("\n", " ") for t in titles], labels = labels,
        styles = [dict(color = "k", marker = "o", ms = 3, lw = 1.8),
                  dict(color = "tab:blue", marker = "s", ms = 3),
                  dict(color = "tab:red", marker = "^", ms = 3, ls = "--")],
        xlabel = r"$x$", ylabel = r"$u / T_0$",
        save_path = f"{FIG_DIR}/escA_perfiles")

    print("\nSaved figs at", FIG_DIR)
    for r in samples:
        e_sep = field_metrics(torch.from_numpy(r["sep"]), torch.from_numpy(r["gt"]), "final")["L2_final"]
        e_dmd = field_metrics(torch.from_numpy(r["dmd"]), torch.from_numpy(r["gt"]), "final")["L2_final"]
        print(f"  {r['name']:14s} T0 = {r['T0']:.3f} | L2 SEP {e_sep:.4f} | L2 DMD {e_dmd:.4f}")
