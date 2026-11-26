import os
import numpy as np
import pandas as pd
import torch
import utils.Equations as eq

from utils.Paths import COMP_DATA, make_dirs


# dom
ALPHA_MIN, ALPHA_MAX = 0.01, 1.0
TOTAL_TIME = 0.1

# mesh
DIM_MESH = 10
T_SIZE = 50

# alpha conv
LENGTH_PLATE = 0.10
TIME_END = 9.0
MATERIALS_alpha = { "Cobre": 111.0,
                    "Aluminio": 97.0,
                    "Hierro": 23.0,
                    "Aire (300 K)": 19.0,
                    "Acero 1% C": 11.72,
                    "Titanio": 9.3,
                    "Inox 304": 4.2 }



MATERIALS = {}
units_in_1_m = 10   # 0.10 = 1u 
t_units_in_1_s = 1/90.0  # 9 = 0.1t
for name, alpha_mm2s in MATERIALS_alpha.items():
    alpha_m2s = alpha_mm2s / 1000000.0
    MATERIALS[name] = alpha_m2s * ( units_in_1_m**2 / t_units_in_1_s)


# test
N_TEST_PER_MATERIAL = 20
NUM_VAL = 50
T0_MIN = 0.1
SEED_TEST = 2006

SOLVER_MODE = "CONST"


OUT_DIR = "OUT_COMPARISON"
TEST_PATH = f"{OUT_DIR}/test_set_T0min.pt"
VAL_PATH = f"{OUT_DIR}/val_set.pt"


def make_dataset(alpha, n, seed = None, accept = None):
    return eq.HeatEquationDataset(eq.HeatEquation2D, n, seed = seed, accept = accept,
                                  spatial_size = (DIM_MESH, DIM_MESH), total_time = TOTAL_TIME,
                                  temporal_size = T_SIZE, alpha = alpha, mode = SOLVER_MODE)



def has_enough_heat(sol):
    return sol[0][1][1] >= T0_MIN

def get_test_set():
    test = {}
    
    for i, (name, alpha) in enumerate(MATERIALS.items()):
        path = f"{COMP_DATA}/test_{i}.pkl"
        
        if os.path.exists(path):
            test[name] = eq.HeatEquationDataset.load(path)
            continue
        
        test[name] = make_dataset(alpha, N_TEST_PER_MATERIAL, SEED_TEST + i, accept = has_enough_heat)
        test[name].save(path)
    
    return test