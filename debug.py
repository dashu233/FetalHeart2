import numpy as np
import pickle
import math
import torch

with open('fold_list.pkl', 'rb') as f:
    dt = pickle.load(f)
    for key in dt:
        print(key,dt[key])