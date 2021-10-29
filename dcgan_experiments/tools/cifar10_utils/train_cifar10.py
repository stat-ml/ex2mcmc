import os

from gans_sampling.tools.gan_metrics.dataloader import get_loader
from trainer import Trainer
from discriminator import Discriminator
from generator import Generator

import datetime
from params_cifar10 import args

cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
new_dir = os.path.join(args.path_to_save_local, cur_time)
os.mkdir(new_dir)
path_to_plots = os.path.join(new_dir, 'plots')
path_to_models = os.path.join(new_dir, 'models')
path_to_logs = os.path.join(new_dir, 'logs.txt')
os.mkdir(path_to_plots)
os.mkdir(path_to_models)
args.model_save_path = path_to_models
args.path_to_plots = path_to_plots
args.log_file = path_to_logs

train_loader, _ = get_loader(root = args.data_root,
                             dataset = args.dataset,
                             batch_size=args.batch_size,
                             num_workers=args.workers,
                             random_seed = args.random_seed)
G = Generator(args)
D = Discriminator(args)

trainer = Trainer(train_loader, G, D, args)

if args.verbose:
   trainer.show_current_model()

print("Start to train GAN")
trainer.train()
