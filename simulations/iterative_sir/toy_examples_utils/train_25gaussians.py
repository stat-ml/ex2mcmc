import datetime
import importlib
import os
import random
import time

import numpy as np
import sklearn.datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from gan_fc_models import (
    Discriminator_fc,
    Generator_fc,
    weights_init_1,
    weights_init_2,
)
from gan_train import train_gan
from matplotlib import pyplot as plt
from paths import path_to_save_local, path_to_save_remote, port_to_remote
from sklearn.preprocessing import StandardScaler
from torch import autograd
from torch.autograd import Variable

from toy_examples_utils import (
    logging,
    prepare_25gaussian_data,
    prepare_dataloader,
    prepare_train_batches,
)


train_jensen = False
if train_jensen:
    module_name = "params_25gaussians"
else:
    module_name = "params_25gaussians_wasserstein"

params_module = importlib.import_module(module_name)
random_seed = params_module.random_seed
batch_size = params_module.batch_size
train_dataset_size = params_module.train_dataset_size
sigma = params_module.sigma
n_dim = params_module.n_dim
n_layers_d = params_module.n_layers_d
n_layers_g = params_module.n_layers_g
n_hid_d = params_module.n_hid_d
n_hid_g = params_module.n_hid_g
n_out = params_module.n_out
normalize_to_0_1 = params_module.normalize_to_0_1
loss_type = params_module.loss_type
lr_init = params_module.lr_init
betas = params_module.betas
use_gradient_penalty = params_module.use_gradient_penalty
Lambda = params_module.Lambda
num_epochs = params_module.num_epochs
num_epoch_for_save = params_module.num_epoch_for_save
batch_size_sample = params_module.batch_size_sample
k_g = params_module.k_g
k_d = params_module.k_d
mode = params_module.mode
n_calib_pts = params_module.n_calib_pts
plot_mhgan = params_module.plot_mhgan
device = params_module.device


# torch.manual_seed(random_seed)
# np.random.seed(random_seed)
# random.seed(random_seed)

X_train, means = prepare_25gaussian_data(
    train_dataset_size,
    sigma,
    random_seed,
)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
# X_train_batches = prepare_train_batches(X_train, BATCH_SIZE)
train_dataloader = prepare_dataloader(
    X_train_std,
    batch_size,
    random_seed=random_seed,
)

G = Generator_fc(
    n_dim=n_dim,
    n_layers=n_layers_g,
    n_hid=n_hid_g,
    n_out=n_out,
    non_linear=nn.ReLU(),
    device=device,
).to(device)
D = Discriminator_fc(
    n_in=n_dim,
    n_layers=n_layers_d,
    n_hid=n_hid_d,
    non_linear=nn.ReLU(),
    device=device,
).to(device)
G.init_weights(weights_init_2, random_seed=random_seed)
D.init_weights(weights_init_2, random_seed=random_seed)

d_optimizer = torch.optim.Adam(D.parameters(), betas=betas, lr=lr_init)
#                               weight_decay = weight_decay)
g_optimizer = torch.optim.Adam(G.parameters(), betas=betas, lr=lr_init)
#                               weight_decay = weight_decay)

cur_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
new_dir = os.path.join(path_to_save_local, cur_time)
os.mkdir(new_dir)
path_to_plots = os.path.join(new_dir, "plots")
path_to_models = os.path.join(new_dir, "models")
os.mkdir(path_to_plots)
os.mkdir(path_to_models)
path_to_logs = os.path.join(new_dir, "logs.txt")

logging(
    path_to_logs=path_to_logs,
    mode=mode,
    train_dataset_size=train_dataset_size,
    batch_size=batch_size,
    n_dim=n_dim,
    n_layers_g=n_layers_g,
    n_layers_d=n_layers_d,
    n_hid_g=n_hid_g,
    n_hid_d=n_hid_d,
    n_out=n_out,
    loss_type=loss_type,
    lr_init=lr_init,
    Lambda=Lambda,
    num_epochs=num_epochs,
    k_g=k_g,
    k_d=k_d,
)

print("Start to train GAN")
train_gan(
    X_train=X_train,
    train_dataloader=train_dataloader,
    generator=G,
    g_optimizer=g_optimizer,
    discriminator=D,
    d_optimizer=d_optimizer,
    loss_type=loss_type,
    batch_size=batch_size,
    device=device,
    use_gradient_penalty=use_gradient_penalty,
    Lambda=Lambda,
    num_epochs=num_epochs,
    num_epoch_for_save=num_epoch_for_save,
    batch_size_sample=batch_size_sample,
    k_g=k_g,
    k_d=k_d,
    n_calib_pts=n_calib_pts,
    normalize_to_0_1=normalize_to_0_1,
    scaler=scaler,
    mode=mode,
    path_to_logs=path_to_logs,
    path_to_models=path_to_models,
    path_to_plots=path_to_plots,
    path_to_save_remote=path_to_save_remote,
    port_to_remote=port_to_remote,
    plot_mhgan=plot_mhgan,
)
