import numpy as np
import torch
import random
import time
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm

import sys
cwd = os.getcwd()
api_path_sampling = os.path.join(cwd, '..', 'sampling_utils')
api_path_gan_metrics = os.path.join(cwd, '..', 'gan_metrics')
sys.path.append(api_path_sampling)
sys.path.append(api_path_gan_metrics)

from metrics import inception_score
from fid_msid_scores import calculate_stat_given_paths
from general_utils import to_var, to_np


def unsqueeze_transform(z):
    return z.unsqueeze(-1).unsqueeze(-1)


class LatentFixDatasetCeleba(torch.utils.data.Dataset):
    """Dataset for Generator
    """

    def __init__(self, latent_arr, G, device, nsamples):
        self.latent_arr = latent_arr
        self.G = G
        self.nsamples = nsamples
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.device = device

    def __getitem__(self, index):
        z = to_var(self.latent_arr[index].unsqueeze(0), self.device)
        with torch.no_grad():
            image = self.G(z)
        clamp_image = self.clamp(image.permute(0, 2, 3, 1))
        np_image = to_np(clamp_image)
        squeeze_image = np.squeeze(np_image)

        return self.transform(squeeze_image)

    def __len__(self):
        return self.nsamples

    def clamp(self, x):
        return (x.clamp(-1, 1) + 1.) / 2.


def delete_nan_samples(z):
    return z[~np.isnan(z).any(axis=1)]


def save_images_for_fid_fix_latent(G,
                                   real_dataloader,
                                   name_fake_test,
                                   name_real_test,
                                   latent_arr,
                                   device,
                                   transformer,
                                   random_seed,
                                   normalize_imgs=False,
                                   use_clamp=False):
    fake_list = []
    real_list = []
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    max_num_imgs = latent_arr.shape[0]
    start_ind = 0

    for i, data_real in tqdm(enumerate(real_dataloader, 0)):
        batch_real = data_real[0]
        batch_size = batch_real.shape[0]

        if start_ind + batch_size < max_num_imgs:
            fixed_noise = latent_arr[start_ind:start_ind + batch_size].to(device)
            if transformer is not None:
                fake_images = transformer(G(fixed_noise))
            else:
                fake_images = G(fixed_noise)
            if use_clamp:
                clamp_fake_images = fake_images.clamp(-1, 1)
            else:
                clamp_fake_images = fake_images
            start_ind += batch_size

            if normalize_imgs:
                fake_norm_np = ((1. + clamp_fake_images) / 2).detach().cpu().numpy()
                real_norm_np = ((1. + batch_real) / 2).detach().cpu().numpy()
            else:
                fake_norm_np = clamp_fake_images.detach().cpu().numpy()
                real_norm_np = batch_real.detach().cpu().numpy()
            fake_list.append(fake_norm_np)
            real_list.append(real_norm_np)

        else:
            fixed_noise = latent_arr[start_ind:].to(device)
            size_noise = fixed_noise.shape[0]
            if transformer is not None:
                fake_images = transformer(G(fixed_noise))
            else:
                fake_images = G(fixed_noise)
            if use_clamp:
                clamp_fake_images = fake_images.clamp(-1, 1)
            else:
                clamp_fake_images = fake_images
            batch_real = batch_real[:size_noise]

            if normalize_imgs:
                fake_norm_np = ((1. + clamp_fake_images) / 2).detach().cpu().numpy()
                real_norm_np = ((1. + batch_real) / 2).detach().cpu().numpy()
            else:
                fake_norm_np = clamp_fake_images.detach().cpu().numpy()
                real_norm_np = batch_real.detach().cpu().numpy()
            fake_list.append(fake_norm_np)
            real_list.append(real_norm_np)
            break

    fake_np = np.concatenate(fake_list)
    real_np = np.concatenate(real_list)

    print(f"shape of generated images = {fake_np.shape}")
    print(f"shape of real images = {real_np.shape}")

    np.save(name_fake_test, fake_np)
    np.save(name_real_test, real_np)


def calculate_celeba_statistics(z_agg_step, G,
                                device, batch_size,
                                path_to_save_np,
                                method_name,
                                image_size,
                                dataroot,
                                calculate_is=True,
                                random_seed=42,
                                every_step=5,
                                normalize_imgs=False,
                                use_clamp=False,
                                z_transform=None
                                ):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if image_size is not None:
        train_dataset = dset.ImageFolder(root=dataroot,
                                         transform=transforms.Compose([
                                                   transforms.Resize(image_size),
                                                   transforms.CenterCrop(image_size),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                                        (0.5, 0.5, 0.5)),
                                         ]))
        transformer = transforms.Compose([transforms.Resize(image_size),
                                          transforms.CenterCrop(image_size)])
    else:
        train_dataset = dset.ImageFolder(root=dataroot,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5),
                                                                  (0.5, 0.5, 0.5)),
                                         ]))
        transformer = None
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)

    batch_size_resnet = batch_size
    dim_resnet = 2048
    model_type = 'inception'
    cuda = True
    inception_scores_mean = []
    inception_scores_std = []
    fid_scores_mean_train = []
    fid_scores_std_train = []
    num_steps = z_agg_step.shape[0]
    for i in range(num_steps):
        print("------------------------------------")
        print(f"step = { i *every_step}")
        current_samples = z_agg_step[i]
        print(f"sample size = {current_samples.shape}")
        no_nans_samples = delete_nan_samples(current_samples)
        print(f"sample size after deleteting nans = {no_nans_samples.shape}")
        nsamples = len(no_nans_samples)
        latent_arr = torch.FloatTensor(no_nans_samples)
        if z_transform is not None:
            latent_arr = z_transform(latent_arr)
        num_samples = latent_arr.shape[0]
        latent_dataset = LatentFixDatasetCeleba(latent_arr, G,
                                                device, nsamples)
        if calculate_is:
            print(f"start to calculate inception score over {num_samples} images...")
            start = time.time()
            (inception_score_mean,
             inception_score_std) = inception_score(latent_dataset,
                                                    device,
                                                    batch_size, True)
            ln = f"{method_name} mean IS = {inception_score_mean}, std IS = {inception_score_std}"
            print(ln)
            inception_scores_mean.append(inception_score_mean)
            inception_scores_std.append(inception_score_std)
            end = round(time.time() - start, 3)
            print(f"time for inception calculation = {end}s")
        print(f"start to calculate FID score for train Celeba over {num_samples} images...")
        start = time.time()
        name_fake_train = os.path.join(path_to_save_np,
                                       f"{method_name}_pretrained_fake_train_step_{i *every_step}.npy")
        name_real_train = os.path.join(path_to_save_np,
                                       f"{method_name}_pretrained_real_train_step_{i *every_step}.npy")
        save_images_for_fid_fix_latent(G,
                                       train_dataloader,
                                       name_fake_train,
                                       name_real_train,
                                       latent_arr,
                                       device,
                                       transformer,
                                       random_seed,
                                       normalize_imgs,
                                       use_clamp
                                       )
        paths_to_train_method = [name_real_train, name_fake_train]
        results_fid_train = calculate_stat_given_paths(paths_to_train_method,
                                                       batch_size_resnet,
                                                       cuda,
                                                       dim_resnet,
                                                       model_type=model_type,
                                                       metric='fid')
        results_fid_train = results_fid_train[0]

        mean_fid_train = results_fid_train[1]
        std_fid_train = results_fid_train[2]
        print(f"FID score for train Celeba with {method_name}: mean {mean_fid_train}, score {std_fid_train}")
        fid_scores_mean_train.append(mean_fid_train)
        fid_scores_std_train.append(std_fid_train)
        end = round(time.time() - start, 3)
        print(f"time for FID calculation on train = {end}s")

    fid_scores_mean_train = np.array(fid_scores_mean_train)
    fid_scores_std_train = np.array(fid_scores_std_train)
    names_list = ["fid_scores_mean_train", "fid_scores_std_train"]
    arrays_list = [fid_scores_mean_train, fid_scores_std_train]
    if calculate_is:
        inception_scores_mean = np.array(inception_scores_mean)
        inception_scores_std = np.array(inception_scores_std)
        names_list.extend(["inception_scores_mean", "inception_scores_std"])
        arrays_list.extend([inception_scores_mean, inception_scores_std])
    dict_results = {}
    for i in range(len(names_list)):
        cur_score_path = os.path.join(path_to_save_np,
                                      f"{method_name}_{names_list[i]}.npy")
        np.save(cur_score_path, arrays_list[i])
        dict_results[names_list[i]] = arrays_list[i]

    return dict_results
