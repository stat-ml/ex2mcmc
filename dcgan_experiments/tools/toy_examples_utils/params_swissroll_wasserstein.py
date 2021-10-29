import torch

random_seed = 42

train_dataset_size = 64000
batch_size = 256    

n_dim = 2
n_layers_d = 3
n_layers_g = 3
n_hid_d = 256
n_hid_g = 128
n_out = 2

normalize_to_0_1 = True

#loss_type='Jensen_nonsaturing'
#loss_type='Jensen_minimax'
loss_type='Wasserstein'

lr_init = 1e-4
betas = (0.5, 0.9)

use_gradient_penalty = True
Lambda = 0.1
num_epochs = 5000
num_epoch_for_save = 100
batch_size_sample = 5000  
k_g = 1
k_d = 100
mode = 'swissroll'
n_calib_pts = 10000

plot_mhgan = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
