import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pydmd import DMD
from tqdm import tqdm
from typing import Dict
from torchviz import make_dot

