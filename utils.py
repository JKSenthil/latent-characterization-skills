import torch
import pickle
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_trajectory(path='data/big_traj_dict.pkl'):
    big_traj_dict, _ = pickle.load(open(path, "rb"))
    return big_traj_dict

def plot_trajectory_histogram(traj_dict, point=None, thres=1000):
    histogram = {}
    if point == None: # plot traj for all plots
        points = traj_dict.keys()
    else:
        points = [point]
    
    for point in points:
        data = traj_dict[point]
        for traj in data:
            if len(traj) < thres:
                if len(traj) not in histogram:
                    histogram[len(traj)] = 0
                histogram[len(traj)] += 1
    
    plt.figure()
    plt.xlabel("Trajectory length")
    plt.ylabel("Frequency")
    plt.bar(histogram.keys(), histogram.values(), 1.0, color='g')
    plt.savefig('histogram.png')
