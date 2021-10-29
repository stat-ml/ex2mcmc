import numpy as np
import sklearn.datasets
import os
import matplotlib.pyplot as plt
import datetime
import random
import itertools

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class PoolSet(Dataset):
    def __init__(self, p_x):
        ## input: torch.tensor (NOT CUDA TENSOR)
        self.len = len(p_x)
        self.x = p_x    ##[N, p]
    
    def __getitem__(self, index):
        return self.x[index]
    
    def __len__(self):
        return self.len

def prepare_swissroll_data(batch_size=1000):
    data = sklearn.datasets.make_swiss_roll(
                    n_samples=batch_size,
                    noise=0.25
                )[0]
    data = data.astype('float32')[:, [0, 2]]
    data /= 7.5 # stdev plus a little
    return data

def prepare_25gaussian_data(batch_size=1000, 
                            sigma=0.05,
                            random_seed=42):
    dataset = []
    for i in range(batch_size//25):
        for x in range(-2, 3):
            for y in range(-2, 3):
                point = np.random.randn(2)*sigma
                point[0] += x
                point[1] += y
                dataset.append(point)
    dataset = np.array(dataset, dtype=np.float32)
    np.random.seed(random_seed)
    random.seed(random_seed)
    np.random.shuffle(dataset)
    means = np.array(list(itertools.product(np.arange(-2,3), 
                                            repeat=2)), dtype=np.float64)
    #dataset /= 2.828 # stdev
    return dataset, means

def prepare_gaussians(num_samples_in_cluster, dim, 
                      num_gaussian_per_dim, coord_limits, 
                      sigma = 0.1, random_seed = 42):
    num_clusters = num_gaussian_per_dim ** dim
    num_samples = num_samples_in_cluster * num_clusters
    coords_per_dim = np.linspace(-coord_limits, 
                                 coord_limits, 
                                 num = num_gaussian_per_dim)
    copy_coords = list(np.tile(coords_per_dim, (dim, 1)))
    centers = np.array(np.meshgrid(*copy_coords)).T.reshape(-1, dim)
    dataset = sklearn.datasets.make_blobs(n_samples = num_samples, 
                                          n_features = dim, 
                                          centers = centers, 
                                          cluster_std = sigma,
                                          random_state = random_seed)[0]
    return dataset

def prepare_train_batches(dataset, batch_size):
    while True:
        for i in range(len(dataset) // batch_size):
            yield dataset[i * batch_size:(i + 1) * batch_size]

def prepare_dataloader(dataset, batch_size, random_seed=None):
    dataset = torch.FloatTensor(dataset) 
    poolset = PoolSet(dataset)
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)    
    dataloader = DataLoader(poolset, batch_size=batch_size, shuffle=True)
    return dataloader

def logging(path_to_logs, mode, train_dataset_size,
            batch_size, n_dim, n_layers_g, 
            n_layers_d, n_hid_g, n_hid_d, 
            n_out, loss_type, lr_init, 
            Lambda, num_epochs, k_g, k_d):
    f = open(path_to_logs, "w")
    f.write(f"Dataset = {mode}\n")
    f.write("Setup for training GANs:\n")
    f.write(f"Train dataset size = {train_dataset_size}\n")
    f.write(f"Batch size = {batch_size}\n")
    f.write(f"Hidden dim for prior of generator = {n_dim}\n")
    f.write(f"Number of hidden layers in generator = {n_layers_g}\n")
    f.write(f"Number of hidden layers in discriminator = {n_layers_d}\n")
    f.write(f"Number of hidden neurons in generator = {n_hid_g}\n")
    f.write(f"Number of hidden neurons in discriminator = {n_hid_d}\n")
    f.write(f"Dim of output for generator = {n_out}\n")
    f.write(f"Loss type = {loss_type}\n")
    f.write(f"Learning rate = {lr_init}\n")
    f.write(f"Lambda for gradient penalization = {Lambda}\n")
    f.write(f"Number of epochs = {num_epochs}\n")
    f.write(f"Number of generator learning passes = {k_g}\n")
    f.write(f"Number of discriminator learning passes = {k_d}\n")
    f.write(f"-------------------------------------------\n")
    f.close()
