import torch


random_seed = 42

batch_size = 64
num_samples_in_cluster = 1000
dim = 5
num_gaussian_per_dim = 3
coord_limits = 2.0
sigma = 0.05
num_clusters = num_gaussian_per_dim ** dim
train_dataset_size = num_samples_in_cluster * num_clusters

n_dim = 5
n_layers_d = 3
n_layers_g = 3
n_hid_d = 512
n_hid_g = 256
n_out = 5

normalize_to_0_1 = True

# loss_type='Jensen_nonsaturing'
# loss_type='Jensen_minimax'
loss_type = "Wasserstein"

lr_init = 1e-4
betas = (0.5, 0.9)

use_gradient_penalty = True
Lambda = 0.1
num_epochs = 5000
num_epoch_for_save = 100
batch_size_sample = 10000
k_g = 1
k_d = 100
mode = "5d_gaussians"
proj_list = [[0, 1], [2, 3], [0, 4]]
n_calib_pts = 3 * batch_size_sample

plot_mhgan = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
