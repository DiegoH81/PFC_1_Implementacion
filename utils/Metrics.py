import pandas as pd
import torch

def rel_l2(pred, true):
    return (torch.linalg.norm(pred - true) / torch.linalg.norm(true)).item()


def bound_violation(pred, u_min, u_max):
    above = (pred - u_max).clamp(min = 0)
    below = (u_min - pred).clamp(min = 0)
    return (above ** 2 + below ** 2).mean().item()

def get_all_metrics(pred, true, prefix, u_max = None, u_min = 0.0):
    err = pred - true
    metrics = {f"L2_{prefix}": rel_l2(pred, true),
            f"MSE_{prefix}": err.pow(2).mean().item()}
    
    if u_max is not None:
        metrics[f"Emax_{prefix}"] = bound_violation(pred, u_min, u_max)
        
    return metrics
    
def delta_rel(error, reference):
    return 100.0 * (error - reference) / reference

def summarize(df, cols, order):
    table = df.groupby("material")[cols].mean().reindex(order)
    table.loc["Promedio"] = df[cols].mean()
    return table