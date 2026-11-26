import os

OUT_REPLICA = "OUTPUT_REPLICA"
REPLICA_WEIGHTS = f"{OUT_REPLICA}/weights"
REPLICA_LOGS = f"{OUT_REPLICA}/logs"


OUT_COMPARISON = "OUTPUT_COMPARISON"
COMP_DATA = f"{OUT_COMPARISON}/data"
COMP_WEIGHTS = f"{OUT_COMPARISON}/weights"
COMP_LOGS = f"{OUT_COMPARISON}/logs"
COMP_FIGS = f"{OUT_COMPARISON}/figs"


TEST_PATH = f"{COMP_DATA}/test_set_T0min.pt"
VAL_PATH = f"{COMP_DATA}/val_set.pt"
SEP_CKPT = f"{COMP_WEIGHTS}/SEP_best.pt"

def make_dirs():
    for d in (REPLICA_WEIGHTS, REPLICA_LOGS, COMP_DATA, COMP_WEIGHTS,
              COMP_LOGS, COMP_FIGS):
        os.makedirs(d, exist_ok = True)