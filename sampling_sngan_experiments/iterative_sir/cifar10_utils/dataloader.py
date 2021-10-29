import os
import random
import sys

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from general_utils import to_np, to_var


sys.path.append("../sampling_utils")


def get_loader(
    root,
    dataset="CIFAR10",
    batch_size=128,
    num_workers=4,
    random_seed=42,
):

    assert dataset in ["CIFAR10", "CIFAR100"]

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    train_loader, val_loader = (
        torch.utils.data.DataLoader(
            globals()[dataset](
                root=root,
                train=is_training,
                download=True,
            ).preprocess(),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        for is_training in [True, False]
    )
    return train_loader, val_loader


class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train, download=True):
        super().__init__(root, train=train, download=download)

    def preprocess(self):
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
        if self.train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ],
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ],
            )
        return self


class CIFAR100(CIFAR10):
    base_folder = "cifar-100-python"
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]


class GenDataset(torch.utils.data.Dataset):
    """Dataset for Generator"""

    def __init__(self, G, nsamples, pretrained=False, device=None, z_dim=None):
        self.G = G
        self.nsamples = nsamples
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ],
        )
        self.pretrained = pretrained
        self.device = device
        self.z_dim = z_dim

    def __getitem__(self, index):
        if self.pretrained:
            z = to_var(torch.randn(1, self.z_dim, 1, 1), self.device)
        else:
            z = to_var(torch.randn(1, self.G.z_dim), self.G.device)
        return self.transform(
            np.squeeze(to_np(self.denorm(self.G(z)).permute(0, 2, 3, 1))),
        )

    def __len__(self):
        return self.nsamples

    def denorm(self, x):
        # For fake data generated with tanh(x)
        x = (x + 1) / 2
        return x.clamp(0, 1)


class LatentFixDataset(torch.utils.data.Dataset):
    """Dataset for Generator"""

    def __init__(self, latent_arr, G, device, nsamples):
        self.latent_arr = latent_arr
        self.G = G
        self.nsamples = nsamples
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ],
        )
        self.device = device

    def __getitem__(self, index):
        z = to_var(self.latent_arr[index], self.device)
        return self.transform(
            np.squeeze(to_np(self.denorm(self.G(z)).permute(0, 2, 3, 1))),
        )

    def __len__(self):
        return self.nsamples

    def denorm(self, x):
        # For fake data generated with tanh(x)
        x = (x + 1) / 2
        return x.clamp(0, 1)
