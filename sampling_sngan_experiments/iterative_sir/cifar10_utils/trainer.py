import datetime
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from dataloader import GenDataset
from general_utils import init_params_xavier, print_network, to_np, to_var
from logger import Logger
from metrics import inception_score
from torch.autograd import Variable


sys.path.append("../sampling_utils")


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class Trainer:
    def __init__(self, train_loader, G, D, args):
        self.train_loader = train_loader
        self.G = G
        self.D = D
        self.args = args
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.nsamples = args.nsamples
        self.d_iter = args.d_iter
        self.g_iter = args.g_iter

        self.epoch = 0
        self.step = 0
        self.device = args.device

        self.logger = Logger(
            log_file=args.log_file,
            plot_dir=args.path_to_plots,
            port_to_remote=args.port_to_remote,
            path_to_save_remote=args.path_to_save_remote,
        )

        self.G_optimizer = optim.Adam(
            G.parameters(),
            self.lr,
            betas=(0.0, 0.9),
        )
        self.D_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, D.parameters()),
            self.lr,
            betas=(0.0, 0.9),
        )
        self.G_scheduler = optim.lr_scheduler.ExponentialLR(
            self.G_optimizer,
            gamma=0.99,
        )
        self.D_scheduler = optim.lr_scheduler.ExponentialLR(
            self.D_optimizer,
            gamma=0.99,
        )

        self.criterion = nn.BCEWithLogitsLoss()

        if (args.model_load_path is not None) and (
            args.pretrained_models is not None
        ):
            print("Start downloading pretrained models")
            trainer.load_models(args.pretrained_models)

        if (args.model_load_path is not None) and (
            args.pretrained_opt_scheduls is not None
        ):
            print("Start downloading pretrained optimizers")
            trainer.load_optimizers_schedulers(args.pretrained_opt_scheduls)

        self.G.device = self.device
        self.D.device = self.device

        if self.device is not None:
            self.G = self.G.to(self.device)
            self.D = self.D.to(self.device)

        if (args.random_seed is not None) and (
            args.pretrained_models is not None
        ):
            torch.manual_seed(args.random_seed)
            np.random.seed(args.random_seed)
            random.seed(args.random_seed)
            self.G.apply(init_params_xavier)

            torch.manual_seed(args.random_seed)
            np.random.seed(args.random_seed)
            random.seed(args.random_seed)
            self.D.apply(init_params_xavier)

        if args.random_seed is not None:
            torch.manual_seed(args.random_seed)
            np.random.seed(args.random_seed)
            random.seed(args.random_seed)
        self.fixed_z = to_var(
            torch.randn(self.nsamples, self.G.z_dim),
            device=self.device,
        )

    def train(self):
        self.sample()
        for self.epoch in range(1, self.args.epochs + 1):
            epoch_info = self.train_epoch()
            for k, v in epoch_info.items():
                self.logger.scalar_summary("loss__" + k, float(v), self.epoch)

            print(
                "Epoch: %3d | Step: %8d | " % (self.epoch, self.step)
                + " | ".join(f"{k}: {v:.5f}" for k, v in epoch_info.items()),
            )
            self.sample()
            self.G_scheduler.step()
            self.D_scheduler.step()

            if (self.epoch - 1) % self.args.num_epoch_for_save == 0:
                print("Start to save models")
                cur_time = datetime.datetime.now().strftime(
                    "%Y_%m_%d-%H_%M_%S",
                )
                filename_models = f"{cur_time}_models_epoch_{self.epoch}.pth"
                self.save_models(filename_models)
                filename_opt_sched = (
                    f"{cur_time}_opt_sched_epoch_{self.epoch}.pth"
                )
                self.save_optimizers_schedulers(filename_opt_sched)

                if self.args.inception_score:
                    print("Start to calculate Inception score")
                    score_mean, score_std = inception_score(
                        GenDataset(self.G, 50000),
                        self.device,
                        self.batch_size,
                        True,
                    )
                    print(
                        "Inception score at epoch {} with 50000 generated samples - Mean: {:.3f}, Std: {:.3f}".format(
                            self.epoch,
                            score_mean,
                            score_std,
                        ),
                    )
                    self.logger.scalar_summary(
                        "mean Inception score",
                        float(score_mean),
                        self.epoch,
                    )
                    self.logger.scalar_summary(
                        "std Inception score",
                        float(score_std),
                        self.epoch,
                    )

    def train_epoch(self):
        self.G.train()
        self.D.train()

        for i, (real_imgs, real_labels) in enumerate(self.train_loader):
            real_imgs, real_labels = to_var(real_imgs, self.device), to_var(
                real_labels,
                self.device,
            )
            self.step += 1

            for _ in range(self.d_iter):
                # Discriminator
                # V(D) = E[logD(x)] + E[log(1-D(G(z)))]
                self.D.zero_grad()
                z = to_var(
                    torch.randn(self.batch_size, self.G.z_dim),
                    self.device,
                )

                real_labels.fill_(1)
                real_labels = real_labels.float()

                d_loss_real = self.criterion(self.D(real_imgs), real_labels)
                fake_imgs = self.G(z).detach()
                fake_labels = real_labels.clone()
                fake_labels.fill_(0)
                d_loss_fake = self.criterion(self.D(fake_imgs), fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()

                self.D_optimizer.step()

            # Generator
            # V(G) = -E[log(D(G(z)))]
            for _ in range(self.g_iter):
                self.G.zero_grad()
                fake_imgs = self.G(z)
                fake_labels.fill_(1)

                g_loss = self.criterion(self.D(fake_imgs), fake_labels)
                g_loss.backward()

                self.G_optimizer.step()

            if self.step % self.args.log_step == 0:
                print(
                    "step: {}, d_loss: {:.5f}, g_loss: {:.5f}".format(
                        self.step,
                        d_loss.cpu().item(),
                        g_loss.cpu().item(),
                    ),
                )

            if self.step % self.args.sample_step == 0:
                samples = self.denorm(self.infer(self.nsamples))
                self.logger.images_summary(
                    "samples_unfixed",
                    samples,
                    self.step,
                )

        return {
            "d_loss_real": d_loss_real.cpu().item(),
            "d_loss_fake": d_loss_fake.cpu().item(),
            "d_loss": d_loss.cpu().item(),
            "g_loss": g_loss.cpu().item(),
        }

    def sample(self):
        self.G.eval()
        if not os.path.exists(self.args.path_to_plots):
            os.makedirs(self.args.path_to_plots)

        samples = self.denorm(self.G(self.fixed_z))
        self.logger.images_summary("samples_fixed", samples, self.step)

    def infer(self, nsamples):
        self.G.eval()
        z = to_var(torch.randn(nsamples, self.G.z_dim), self.device)
        return self.G(z)

    def denorm(self, x):
        # For fake data generated with tanh(x)
        x = (x + 1) / 2
        return x.clamp(0, 1)

    def show_current_model(self):
        print_network(self.G)
        print_network(self.D)

    def save_models(self, filename):
        torch.save(
            {"G": self.G.state_dict(), "D": self.D.state_dict()},
            os.path.join(self.args.model_save_path, filename),
        )

    def save_optimizers_schedulers(self, filename):
        torch.save(
            {
                "G_opt": self.G_optimizer.state_dict(),
                "D_opt": self.D_optimizer.state_dict(),
                "G_sched": self.G_scheduler.state_dict(),
                "D_sched": self.D_scheduler.state_dict(),
            },
            os.path.join(self.args.model_save_path, filename),
        )

    def load_models(self, filename):
        ckpt = torch.load(os.path.join(self.args.model_load_path, filename))
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])

    def load_optimizers_schedulers(self, filename):
        ckpt = torch.load(os.path.join(self.args.model_load_path, filename))
        self.G_optimizer.load_state_dict(ckpt["G_opt"])
        self.D_optimizer.load_state_dict(ckpt["D_opt"])
        self.G_scheduler.load_state_dict(ckpt["G_sched"])
        self.D_scheduler.load_state_dict(ckpt["D_sched"])
