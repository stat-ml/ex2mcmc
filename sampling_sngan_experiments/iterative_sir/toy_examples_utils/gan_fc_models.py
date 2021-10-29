import random

import numpy as np
import torch
import torch.nn as nn


# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)
device_default = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init_1(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_2(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        std_init = 0.8 * (2 / m.in_features) ** 0.5
        m.weight.data.normal_(0.0, std=std_init)
        m.bias.data.fill_(0)


class Generator_fc(nn.Module):
    def __init__(
        self,
        n_dim=2,
        n_layers=4,
        n_hid=100,
        n_out=2,
        non_linear=nn.ReLU(),
        device=device_default,
    ):
        super().__init__()
        self.non_linear = non_linear
        self.device = device
        self.n_hid = n_hid
        self.n_dim = n_dim
        self.n_out = n_out
        self.n_layers = n_layers
        layers = [nn.Linear(self.n_dim, self.n_hid), non_linear]
        for i in range(n_layers - 1):
            layers.extend([nn.Linear(n_hid, n_hid), non_linear])
        layers.append(nn.Linear(n_hid, n_out))

        self.layers = nn.Sequential(*layers)
        # for i in range(4):
        #    std_init = 0.8 * (2/self.layers[i].in_features)**0.5
        #    torch.nn.init.normal_(self.layers[i].weight, std = std_init)

    def make_hidden(self, batch_size, random_seed=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
        return torch.randn(batch_size, self.n_dim, device=self.device)

    def forward(self, z):
        z = self.layers.forward(z)
        return z

    def sampling(self, batch_size, random_seed=None):
        z = self.make_hidden(batch_size, random_seed=random_seed)
        # print(z.detach().cpu())
        return self.forward(z)

    def init_weights(self, init_fun=weights_init_1, random_seed=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
        self.apply(init_fun)


class Discriminator_fc(nn.Module):
    def __init__(
        self,
        n_in=2,
        n_layers=4,
        n_hid=100,
        non_linear=nn.ReLU(),
        device=device_default,
    ):
        super().__init__()
        self.non_linear = non_linear
        self.device = device
        self.n_hid = n_hid
        self.n_in = n_in
        layers = [nn.Linear(self.n_in, self.n_hid), non_linear]
        for i in range(n_layers - 1):
            layers.extend([nn.Linear(n_hid, n_hid), non_linear])
        layers.append(nn.Linear(n_hid, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        z = self.layers.forward(z)
        return z

    def init_weights(self, init_fun=weights_init_1, random_seed=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
        self.apply(init_fun)
