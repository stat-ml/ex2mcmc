import copy
import hashlib
import os
import pickle
import random
import time
import uuid

import dnnlib
import metrics_stylegan2_ada.metric_utils as metric_utils
import numpy as np
import scipy.linalg
import torch
from metrics_stylegan2_ada.metric_utils import ProgressMonitor


_feature_detector_cache = dict()


class MetricOptions:
    def __init__(
        self,
        G=None,
        G_kwargs={},
        dataset_kwargs={},
        num_gpus=1,
        rank=0,
        device=None,
        progress=None,
        cache=True,
        latent_dataset=None,
    ):
        assert 0 <= rank < num_gpus
        self.G = G
        self.G_kwargs = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.num_gpus = num_gpus
        self.rank = rank
        self.device = (
            device if device is not None else torch.device("cuda", rank)
        )
        self.progress = (
            progress.sub()
            if progress is not None and rank == 0
            else ProgressMonitor()
        )
        self.cache = cache
        self.latent_dataset = latent_dataset


def get_feature_detector_name(url):
    return os.path.splitext(url.split("/")[-1])[0]


def get_feature_detector(
    url,
    device=torch.device("cpu"),
    num_gpus=1,
    rank=0,
    verbose=False,
):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = rank == 0
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier()  # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier()  # others follow
    return _feature_detector_cache[key]


class FeatureStats:
    def __init__(
        self,
        capture_all=False,
        capture_mean_cov=False,
        max_items=None,
    ):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros(
                [num_features, num_features],
                dtype=np.float64,
            )

    def is_full(self):
        return (self.max_items is not None) and (
            self.num_items >= self.max_items
        )

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (
            self.num_items + x.shape[0] > self.max_items
        ):
            if self.num_items >= self.max_items:
                return
            x = x[: self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1)  # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, "wb") as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, "rb") as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj


def compute_feature_stats_for_dataset(
    opts,
    detector_url,
    detector_kwargs,
    rel_lo=0,
    rel_hi=1,
    batch_size=64,
    data_loader_kwargs=None,
    max_items=None,
    **stats_kwargs,
):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(
            pin_memory=True,
            num_workers=3,
            prefetch_factor=2,
        )

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(
            dataset_kwargs=opts.dataset_kwargs,
            detector_url=detector_url,
            detector_kwargs=detector_kwargs,
            stats_kwargs=stats_kwargs,
        )
        md5 = hashlib.md5(repr(sorted(args.items())).encode("utf-8"))
        cache_tag = f"{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}"
        cache_file = dnnlib.make_cache_dir_path(
            "gan-metrics",
            cache_tag + ".pkl",
        )

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(
                flag,
                dtype=torch.float32,
                device=opts.device,
            )
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = float(flag.cpu()) != 0

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(
        tag="dataset features",
        num_items=num_items,
        rel_lo=rel_lo,
        rel_hi=rel_hi,
    )
    detector = get_feature_detector(
        url=detector_url,
        device=opts.device,
        num_gpus=opts.num_gpus,
        rank=opts.rank,
        verbose=progress.verbose,
    )

    # Main loop.
    item_subset = [
        (i * opts.num_gpus + opts.rank) % num_items
        for i in range((num_items - 1) // opts.num_gpus + 1)
    ]
    for images, _labels in torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=item_subset,
        batch_size=batch_size,
        **data_loader_kwargs,
    ):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images.to(opts.device), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + "." + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file)  # atomic
    return stats


def compute_feature_stats_for_generator(
    opts,
    detector_url,
    detector_kwargs,
    rel_lo=0,
    rel_hi=1,
    batch_size=64,
    batch_gen=None,
    jit=False,
    **stats_kwargs,
):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and load labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    # Image generation func.
    def run_generator(z, c):
        img = G(z=z, c=c, **opts.G_kwargs)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img

    # JIT.
    if jit:
        z = torch.zeros([batch_gen, G.z_dim], device=opts.device)
        c = torch.zeros([batch_gen, G.c_dim], device=opts.device)
        run_generator = torch.jit.trace(
            run_generator,
            [z, c],
            check_trace=False,
        )

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(
        tag="generator features",
        num_items=stats.max_items,
        rel_lo=rel_lo,
        rel_hi=rel_hi,
    )
    detector = get_feature_detector(
        url=detector_url,
        device=opts.device,
        num_gpus=opts.num_gpus,
        rank=opts.rank,
        verbose=progress.verbose,
    )

    # Main loop.
    num_imgs = 0
    while not stats.is_full():
        images = []
        for _i in range(batch_size // batch_gen):
            z = torch.randn([batch_gen, G.z_dim], device=opts.device)
            # print(f"z shape = {z.shape}")
            c = [
                dataset.get_label(np.random.randint(len(dataset)))
                for _i in range(batch_gen)
            ]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            images.append(run_generator(z, c))
        images = torch.cat(images)
        num_imgs += images.shape[0]
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # print(f"num imgs = {num_imgs}")
    return stats


def compute_mean_cov(x_arr):
    num_features = x_arr[0].shape[1]
    raw_mean = np.zeros([num_features], dtype=np.float64)
    raw_cov = np.zeros([num_features, num_features], dtype=np.float64)
    num_items = 0

    for x in x_arr:
        num_items += x.shape[0]
        x64 = x.astype(np.float64)
        raw_mean += x64.sum(axis=0)
        raw_cov += x64.T @ x64

    # print(f"len of split = {len(x_arr)}")
    # print(f"num_items = {num_items}")
    mean = raw_mean / num_items
    cov = raw_cov / num_items
    cov = cov - np.outer(mean, mean)
    return mean, cov


def compute_feature_stats_for_latent_dataset(
    opts,
    detector_url,
    detector_kwargs,
    latent_dataset,
    rel_lo=0,
    rel_hi=1,
    batch_size=64,
    batch_gen=None,
    jit=False,
    **stats_kwargs,
):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and load labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    # Image generation func.
    def run_generator(z, c):
        img = G(z=z, c=c, **opts.G_kwargs)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img

    # JIT.
    if jit:
        z = torch.zeros([batch_gen, G.z_dim], device=opts.device)
        c = torch.zeros([batch_gen, G.c_dim], device=opts.device)
        run_generator = torch.jit.trace(
            run_generator,
            [z, c],
            check_trace=False,
        )

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(
        tag="latent dataset features",
        num_items=stats.max_items,
        rel_lo=rel_lo,
        rel_hi=rel_hi,
    )
    detector = get_feature_detector(
        url=detector_url,
        device=opts.device,
        num_gpus=opts.num_gpus,
        rank=opts.rank,
        verbose=progress.verbose,
    )

    # Main loop.
    index_latent = 0
    while not stats.is_full():
        images = []
        for _i in range(batch_size // batch_gen):
            z = latent_dataset[
                (index_latent * batch_gen) : ((index_latent + 1) * batch_gen)
            ].to(opts.device)
            # print(f"z shape = {z.shape}")
            c = [
                dataset.get_label(np.random.randint(len(dataset)))
                for _i in range(batch_gen)
            ]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            images.append(run_generator(z, c))
            index_latent += 1
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    return stats


def compute_is_for_latent_dataset(opts, num_gen, num_splits, latent_dataset):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"
    detector_kwargs = dict(
        no_output_bias=True,
    )  # Match the original implementation by not applying bias in the softmax layer.

    gen_probs = compute_feature_stats_for_latent_dataset(
        opts=opts,
        detector_url=detector_url,
        detector_kwargs=detector_kwargs,
        latent_dataset=latent_dataset,
        capture_all=True,
        max_items=num_gen,
    ).get_all()

    if opts.rank != 0:
        return float("nan"), float("nan")

    scores = []
    for i in range(num_splits):
        part = gen_probs[
            i * num_gen // num_splits : (i + 1) * num_gen // num_splits
        ]
        kl = part * (
            np.log(part) - np.log(np.mean(part, axis=0, keepdims=True))
        )
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))


def compute_is(opts, num_gen, num_splits):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"
    detector_kwargs = dict(
        no_output_bias=True,
    )  # Match the original implementation by not applying bias in the softmax layer.

    gen_probs = compute_feature_stats_for_generator(
        opts=opts,
        detector_url=detector_url,
        detector_kwargs=detector_kwargs,
        capture_all=True,
        max_items=num_gen,
    ).get_all()

    if opts.rank != 0:
        return float("nan"), float("nan")

    scores = []
    for i in range(num_splits):
        part = gen_probs[
            i * num_gen // num_splits : (i + 1) * num_gen // num_splits
        ]
        kl = part * (
            np.log(part) - np.log(np.mean(part, axis=0, keepdims=True))
        )
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))


def calculate_activation_statistics(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_fid_for_latent_dataset(
    opts,
    max_real,
    num_gen,
    num_splits,
    latent_dataset,
):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"
    detector_kwargs = dict(
        return_features=True,
    )  # Return raw features before the softmax layer.

    data_loader_kwargs = dict(pin_memory=True, prefetch_factor=2)

    dataset_features = compute_feature_stats_for_dataset(
        opts=opts,
        detector_url=detector_url,
        detector_kwargs=detector_kwargs,
        rel_lo=0,
        rel_hi=0,
        capture_all=True,
        data_loader_kwargs=data_loader_kwargs,
        max_items=max_real,
    ).get_all()

    latent_dataset_features = compute_feature_stats_for_latent_dataset(
        opts=opts,
        detector_url=detector_url,
        detector_kwargs=detector_kwargs,
        latent_dataset=latent_dataset,
        rel_lo=0,
        rel_hi=1,
        capture_all=True,
        max_items=num_gen,
    ).get_all()

    if opts.rank != 0:
        return float("nan")

    # print(f"num samples = {num_samples}")

    scores = []
    for i in range(num_splits):
        print(f"split = {i}")
        rng1 = np.random.default_rng(i)
        rng2 = np.random.default_rng(i + num_splits)
        act1_bs = dataset_features[
            rng1.choice(
                dataset_features.shape[0],
                dataset_features.shape[0],
                replace=True,
            )
        ]
        act2_bs = latent_dataset_features[
            rng2.choice(
                latent_dataset_features.shape[0],
                latent_dataset_features.shape[0],
                replace=True,
            )
        ]

        mu_real_split, sigma_real_split = calculate_activation_statistics(
            act1_bs,
        )
        mu_gen_split, sigma_gen_split = calculate_activation_statistics(
            act2_bs,
        )

        m = np.square(mu_gen_split - mu_real_split).sum()
        s, _ = scipy.linalg.sqrtm(
            np.dot(sigma_gen_split, sigma_real_split),
            disp=False,
        )  # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen_split + sigma_real_split - s * 2))
        scores.append(float(fid))
    scores = np.array(scores)
    # print(f"scores = {scores}")

    return float(np.mean(scores)), float(np.std(scores))


def compute_fid_for_latent_dataset_no_std(
    opts,
    max_real,
    num_gen,
    latent_dataset,
):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"
    detector_kwargs = dict(
        return_features=True,
    )  # Return raw features before the softmax layer.

    mu_real, sigma_real = compute_feature_stats_for_dataset(
        opts=opts,
        detector_url=detector_url,
        detector_kwargs=detector_kwargs,
        rel_lo=0,
        rel_hi=0,
        capture_mean_cov=True,
        max_items=max_real,
    ).get_mean_cov()

    mu_gen, sigma_gen = compute_feature_stats_for_latent_dataset(
        opts=opts,
        detector_url=detector_url,
        detector_kwargs=detector_kwargs,
        latent_dataset=latent_dataset,
        rel_lo=0,
        rel_hi=1,
        capture_mean_cov=True,
        max_items=num_gen,
    ).get_mean_cov()

    if opts.rank != 0:
        return float("nan")

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(
        np.dot(sigma_gen, sigma_real),
        disp=False,
    )  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)


def compute_fid(opts, max_real, num_gen, num_splits):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"
    detector_kwargs = dict(
        return_features=True,
    )  # Return raw features before the softmax layer.

    mu_real, sigma_real = compute_feature_stats_for_dataset(
        opts=opts,
        detector_url=detector_url,
        detector_kwargs=detector_kwargs,
        rel_lo=0,
        rel_hi=0,
        capture_mean_cov=True,
        max_items=max_real,
    ).get_mean_cov()

    mu_gen, sigma_gen = compute_feature_stats_for_generator(
        opts=opts,
        detector_url=detector_url,
        detector_kwargs=detector_kwargs,
        rel_lo=0,
        rel_hi=1,
        capture_mean_cov=True,
        max_items=num_gen,
    ).get_mean_cov()

    if opts.rank != 0:
        return float("nan")

    num_samples = min(mu_real.shape[0], mu_gen.shape[0])
    scores = []
    for i in range(num_splits):
        mu_real_split = mu_real[
            i * num_samples // num_splits : (i + 1) * num_samples // num_splits
        ]
        sigma_real_split = sigma_real[
            i * num_samples // num_splits : (i + 1) * num_samples // num_splits
        ]
        mu_gen_split = mu_gen[
            i * num_samples // num_splits : (i + 1) * num_samples // num_splits
        ]
        sigma_gen_split = sigma_gen[
            i * num_samples // num_splits : (i + 1) * num_samples // num_splits
        ]
        m = np.square(mu_gen_split - mu_real_split).sum()
        s, _ = scipy.linalg.sqrtm(
            np.dot(sigma_gen_split, sigma_real_split),
            disp=False,
        )  # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen_split + sigma_real_split - s * 2))
        scores.append(float(fid))
    scores = np.array(scores)

    return float(np.mean(scores)), float(np.std(scores))


def fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    mean, std = compute_fid(opts, max_real=None, num_gen=50000, num_splits=10)
    return dict(fid50k_mean=mean, fid50k_std=std)


def fid50k_full_for_latent_dataset(opts, latent_dataset):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    mean, std = compute_fid_for_latent_dataset(
        opts,
        max_real=None,
        num_gen=50000,
        num_splits=10,
        latent_dataset=latent_dataset,
    )
    return dict(fid50k_latent_mean=mean, fid50k_latent_std=std)


def fid50k_full_for_latent_dataset_no_std(opts, latent_dataset):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    mean = compute_fid_for_latent_dataset_no_std(
        opts,
        max_real=None,
        num_gen=50000,
        latent_dataset=latent_dataset,
    )
    return dict(fid50k_latent_mean=mean)


def is50k(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    mean, std = compute_is(opts, num_gen=50000, num_splits=10)
    return dict(is50k_mean=mean, is50k_std=std)


def is50k_for_latent_dataset(opts, latent_dataset):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    mean, std = compute_is_for_latent_dataset(
        opts,
        num_gen=50000,
        num_splits=10,
        latent_dataset=latent_dataset,
    )
    return dict(is50k_latent_mean=mean, is50k_latent_std=std)


def calc_metric(
    metric,
    **kwargs,
):  # See metric_utils.MetricOptions for the full list of arguments.
    opts = MetricOptions(**kwargs)

    # Calculate.
    start_time = time.time()
    if metric == "fid50k_full":
        results = fid50k_full(opts)
    elif metric == "kid50k_full":
        results = kid50k_full(opts)
    elif metric == "is50k":
        results = is50k(opts)
    elif metric == "is50k_for_latent_dataset":
        results = is50k_for_latent_dataset(opts, opts.latent_dataset)
    elif metric == "fid50k_full_for_latent_dataset":
        results = fid50k_full_for_latent_dataset(opts, opts.latent_dataset)
    elif metric == "fid50k_full_for_latent_dataset_no_std":
        results = fid50k_full_for_latent_dataset_no_std(
            opts,
            opts.latent_dataset,
        )

    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results=dnnlib.EasyDict(results),
        metric=metric,
        total_time=total_time,
        total_time_str=dnnlib.util.format_time(total_time),
    )


def delete_nan_samples(z):
    return z[~np.isnan(z).any(axis=1)]


def pipeline_for_latent_dataset(
    G,
    device,
    data,
    load_batches,
    path_to_save_cifar10_np,
    method_name,
    every_step,
    calc_is=True,
):
    if calc_is:
        metrics = [
            "is50k_for_latent_dataset",
            "fid50k_full_for_latent_dataset",
        ]
        names_list = [
            "inception_scores_mean",
            "inception_scores_std",
            "fid_scores_mean_train",
            "fid_scores_std_train",
        ]
    else:
        metrics = ["fid50k_full_for_latent_dataset"]
        names_list = ["fid_scores_mean_train", "fid_scores_std_train"]
    network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl"
    gpus = 1
    verbose = True
    args = dnnlib.EasyDict(
        metrics=metrics,
        num_gpus=gpus,
        network_pkl=network_pkl,
        verbose=verbose,
    )
    args.G = G
    args.dataset_kwargs = dnnlib.EasyDict(
        class_name="training.dataset.ImageFolderDataset",
        path=data,
    )
    args.dataset_kwargs.resolution = args.G.img_resolution
    args.dataset_kwargs.use_labels = args.G.c_dim != 0
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    G = copy.deepcopy(args.G).eval().requires_grad_(False).to(device)

    inception_scores_mean = []
    inception_scores_std = []
    fid_scores_mean_train = []
    fid_scores_std_train = []

    rank = 0

    for i in range(len(load_batches)):
        print("------------------------------------")
        print(f"step = {i*every_step}")
        all_dicts = {}

        current_samples = load_batches[i]
        print(f"sample size = {current_samples.shape}")
        no_nans_samples = delete_nan_samples(current_samples)
        print(f"sample size after deleteting nans = {no_nans_samples.shape}")
        latent_arr = torch.FloatTensor(no_nans_samples)

        for metric in args.metrics:
            print(f"Calculating {metric}...")
            progress = metric_utils.ProgressMonitor(verbose=args.verbose)
            result_dict = calc_metric(
                metric=metric,
                G=G,
                dataset_kwargs=args.dataset_kwargs,
                num_gpus=args.num_gpus,
                rank=rank,
                device=device,
                progress=progress,
                latent_dataset=latent_arr,
            )
            all_dicts[metric] = result_dict

        if calc_is:
            inception_scores_mean.append(
                all_dicts["is50k_for_latent_dataset"]["results"][
                    "is50k_latent_mean"
                ],
            )
            inception_scores_std.append(
                all_dicts["is50k_for_latent_dataset"]["results"][
                    "is50k_latent_std"
                ],
            )
            print(
                f"{method_name} mean inception score = {inception_scores_mean[i]}, std inception score = {inception_scores_std[i]}",
            )

        fid_scores_mean_train.append(
            all_dicts["fid50k_full_for_latent_dataset"]["results"][
                "fid50k_latent_mean"
            ],
        )
        fid_scores_std_train.append(
            all_dicts["fid50k_full_for_latent_dataset"]["results"][
                "fid50k_latent_std"
            ],
        )

        print(
            f"FID score for train CIFAR10 with {method_name}: mean {fid_scores_mean_train[i]}, score {fid_scores_std_train[i]}",
        )

    if calc_is:
        arrays_list = [
            inception_scores_mean,
            inception_scores_std,
            fid_scores_mean_train,
            fid_scores_std_train,
        ]
    else:
        arrays_list = [fid_scores_mean_train, fid_scores_std_train]

    dict_results = {}
    for i in range(len(names_list)):
        cur_score_path = os.path.join(
            path_to_save_cifar10_np,
            f"{method_name}_{names_list[i]}.npy",
        )
        np.save(cur_score_path, np.array(arrays_list[i]))
        dict_results[names_list[i]] = np.array(arrays_list[i])

    return dict_results
