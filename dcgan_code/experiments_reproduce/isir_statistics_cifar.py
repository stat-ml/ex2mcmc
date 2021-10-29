import os
import sys
import torch

path_to_tools = '/home/daniil/pycharm_dir/gans_sampling'

api_path_cifar = os.path.join(path_to_tools, 'tools', 'cifar10_utils')
api_path_sampling = os.path.join(path_to_tools, 'tools', 'sampling_utils')
api_path_gan_metrics = os.path.join(path_to_tools, 'tools', 'gan_metrics')
models_cifar_scratch_path = os.path.join(path_to_tools, 'models', 'models_cifar10')

sys.path.append(api_path_cifar)
sys.path.append(api_path_sampling)
sys.path.append(api_path_gan_metrics)

from ebm_sampling import (load_data_from_batches)
from metrics_utils import (calculate_images_statistics)
from dcgan import (Generator_cifar10)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

G = Generator_cifar10(ngpu=1)

G.load_state_dict(torch.load(os.path.join(models_cifar_scratch_path, 'netG_epoch_199.pth')))

if torch.cuda.is_available():
    G = G.to(device).eval()

n = 50000
n_steps = 1000
N = 5
batch_size = 200

method_name = 'sir_dcgan_cifar_recalc_1000_steps'
path_to_save = '/home/daniil/gans-mcmc/saved_numpy_arrays'
file_name = f'{method_name}_N_{N}_nsteps_{n_steps}'

load_np = load_data_from_batches(n, batch_size,
                                 path_to_save, file_name)

batch_size = 50
random_seed = 42
path_to_save_np = '/home/daniil/gans-mcmc/saved_numpy_arrays'
path_to_save = path_to_save_np
method_name = 'sir_dcgan_cifar_recalc_stats_1000_steps'
every_step = 50
dataset = "cifar10"
calculate_is = True
calculate_msid_train = False
calculate_msid_test = False

sir_statistics = calculate_images_statistics(z_agg_step=load_np,
                                             G=G,
                                             device=device,
                                             batch_size=batch_size,
                                             path_to_save=path_to_save,
                                             path_to_save_np=path_to_save_np,
                                             method_name=method_name,
                                             random_seed=random_seed,
                                             every_step=every_step,
                                             dataset=dataset,
                                             calculate_is=calculate_is,
                                             calculate_msid_train=calculate_msid_train,
                                             calculate_msid_test=calculate_msid_test)