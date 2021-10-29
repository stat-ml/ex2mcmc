import random

import numpy as np
import torch


class DotConfig:
    """
    Simple wrapper for config
    allowing access with dot notation
    """

    def __init__(self, yaml):
        self._dict = dict(yaml)

    def __getattr__(self, k):
        v = self._dict[k]
        if isinstance(v, dict):
            return DotConfig(v)
        return v

    def items(self):
        return [(k, DotConfig(v)) for k, v in self._dict.items()]

    @property
    def dict(self):
        return dict(self._dict)


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
