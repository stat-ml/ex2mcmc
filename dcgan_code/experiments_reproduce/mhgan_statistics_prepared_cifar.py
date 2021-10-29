import os
import sys

path_to_tools = '/home/daniil/pycharm_dir/gans_sampling'

api_path_cifar = os.path.join(path_to_tools, 'tools', 'cifar10_utils')
api_path_sampling = os.path.join(path_to_tools, 'tools', 'sampling_utils')
api_path_gan_metrics = os.path.join(path_to_tools, 'tools', 'gan_metrics')
models_cifar_scratch_path = os.path.join(path_to_tools, 'models', 'models_cifar10')

sys.path.append(api_path_cifar)
sys.path.append(api_path_sampling)
sys.path.append(api_path_gan_metrics)

import numpy as np
import random

import torch

from dcgan import (Discriminator_cifar10,
                   Generator_cifar10)

from params_cifar10 import args
from general_utils import Discriminator_logits

from metrics_utils import (calculate_images_statistics)

args.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = args.device

G = Generator_cifar10(ngpu=1)
D = Discriminator_cifar10(ngpu=1)

D.load_state_dict(torch.load(os.path.join(models_cifar_scratch_path, 'netD_epoch_199.pth')))
G.load_state_dict(torch.load(os.path.join(models_cifar_scratch_path, 'netG_epoch_199.pth')))

D_logits = Discriminator_logits(D, ngpu=1)

if torch.cuda.is_available():
    D = D.to(device).eval()
    G = G.to(device).eval()

G.z_dim = 100
G.device = device
z_dim = 100

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed)

n_steps = 640
path_to_save = f'/home/daniil/gans-mcmc/saved_numpy_arrays/mhgan_50k_{n_steps}_iters.npy'
download_data = np.load(path_to_save)

download_data_unsqueeze = np.expand_dims(download_data, axis=0)

method_name = 'mhgan_cifar10_statistics'
random_seed = 42
every_step = 1
use_generator = False
path_to_save_np = "/home/daniil/gans-mcmc/saved_numpy_arrays"
path_to_save = "/home/daniil/gans-mcmc/saved_numpy_arrays"
dataset = "cifar10"
calculate_is = True
calculate_msid_train = False
calculate_msid_test = False

batch_size = 50

mhgan_scores = calculate_images_statistics(z_agg_step = download_data_unsqueeze,
                                           G = G,
                                           device = device,
                                           batch_size = batch_size,
                                           path_to_save = path_to_save,
                                           path_to_save_np = path_to_save_np,
                                           method_name = method_name,
                                           random_seed = random_seed,
                                           every_step = every_step,
                                           use_generator = use_generator,
                                           dataset = dataset,
                                           calculate_is = calculate_is,
                                           calculate_msid_train = calculate_msid_train,
                                           calculate_msid_test = calculate_msid_test)