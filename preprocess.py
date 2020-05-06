import numpy as np

from utils import load_trajectory

def preprocess(point=None, seq_len=32):
    """
    point :: trajectory for point to load from, 
        None -> all points
    seq_len :: sequence length of trajectories
    """
    traj_dict = load_trajectory()
    if point == None:
        trajectories = []
        for key in traj_dict:
            for t in traj_dict[key]:
                trajectories.append(t)
    else:
        trajectories = traj_dict[point]

    count = 0
    for traj in trajectories:
        if len(traj) <= seq_len:
            count += 1
    
    padded_trajectories = np.zeros((count, seq_len, 6))
    i = 0
    for traj in trajectories:
        if len(traj) <= seq_len:
            padded_trajectories[i, 0:len(traj),:] = traj
            for j in range(len(traj), seq_len):
                padded_trajectories[i, j,:] = traj[-1,:]
            i += 1
    return padded_trajectories

def normalize(padded_trajectories):
    """
    Normalizes state features to (0,1)
    """
    _min = np.min(padded_trajectories, axis=(0,1))
    normalized_padded_traj = padded_trajectories - _min
    _max = np.max(normalized_padded_traj, axis=(0,1))
    normalized_padded_traj /= _max
    return normalized_padded_traj, _max, _min