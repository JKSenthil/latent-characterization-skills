import os
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(PROJ_DIR, "plots")
WEIGHTS_DIR = os.path.join(PROJ_DIR, 'model_weights')
DATA_PATH = os.path.join(PROJ_DIR, 'data', 'big_traj_dict.pkl')
