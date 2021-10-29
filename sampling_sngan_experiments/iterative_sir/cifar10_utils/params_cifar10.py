import sys

import torch
from general_utils import DotDict


sys.path.append("../sampling_utils")


args = DotDict()

args.dataset = "CIFAR10"
args.sn = True
args.z_dim = 128
args.random_seed = 42

args.epochs = 100
args.batch_size = 64
args.lr = 2e-4
args.g_iter = 1
args.d_iter = 5

args.mode = "train"
args.path_to_save_local = "/home/daniil/gans-mcmc/cifar10_experiment"
args.path_to_save_remote = "/media/Data/Archive/Common/Sirius2020/Sampling_control_optimization/plots_statml"
args.port_to_remote = 12345
args.data_root = "/home/daniil/gans-mcmc/cifar10/cifar_data"
args.nsamples = 64
args.inception_score = True
args.workers = 4
args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# args.model_load_path = '/home/daniil/gans-mcmc/cifar10_experiment/2021_01_26-01_40_45/models'
# args.pretrained_filename = '2021_01_26-04_17_40_models_epoch_31.pth'
args.model_load_path = None
args.pretrained_models = None
args.pretrained_opt_scheduls = None

args.log_step = 20
args.sample_step = 400
args.verbose = True
args.num_epoch_for_save = 10

args.m_g = 4
args.ngf = 512
args.ndf = 512
