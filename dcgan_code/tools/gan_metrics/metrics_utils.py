import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import time
import os
from tqdm import tqdm

import torchvision.datasets as dset
import torchvision.transforms as transforms

import sys

cwd = os.getcwd()
api_path_sampling = os.path.join(cwd, '..', 'sampling_utils')
api_path_gan_metrics = os.path.join(cwd, '..', 'gan_metrics')
sys.path.append(api_path_sampling)
sys.path.append(api_path_gan_metrics)

from metrics import inception_score
from fid_msid_scores import calculate_stat_given_paths
from dataloader import LatentFixDataset


def save_images_fix_latent(G,
                           real_dataloader,
                           name_fake_test,
                           name_real_test,
                           latent_arr,
                           device,
                           random_seed,
                           use_generator=True,
                           use_grayscale=False):
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

        if start_ind + batch_size <= max_num_imgs:
            fixed_noise = latent_arr[start_ind:start_ind + batch_size].to(device)
            if use_generator:
                fixed_noise = G(fixed_noise)
            fake_images = fixed_noise.clamp(-1, 1)
            start_ind += batch_size
            fake_norm = (1. + fake_images) / 2
            real_norm = (1. + batch_real) / 2
            if use_grayscale:
                fake_norm = fake_norm.repeat(1, 3, 1, 1)
                real_norm = real_norm.repeat(1, 3, 1, 1)
            fake_norm_np = fake_norm.detach().cpu().numpy()
            real_norm_np = real_norm.detach().cpu().numpy()
            fake_list.append(fake_norm_np)
            real_list.append(real_norm_np)

        else:
            fixed_noise = latent_arr[start_ind:].to(device)
            if use_generator:
                fixed_noise = G(fixed_noise)
            fake_images = fixed_noise.clamp(-1, 1)
            add_num_imgs = max_num_imgs - start_ind
            batch_real = batch_real[:add_num_imgs]

            fake_norm = (1. + fake_images) / 2
            real_norm = (1. + batch_real) / 2
            if use_grayscale:
                fake_norm = fake_norm.repeat(1, 3, 1, 1)
                real_norm = real_norm.repeat(1, 3, 1, 1)

            fake_norm_np = fake_norm.detach().cpu().numpy()
            real_norm_np = real_norm.detach().cpu().numpy()
            fake_list.append(fake_norm_np)
            real_list.append(real_norm_np)
            break

    fake_np = np.concatenate(fake_list)
    real_np = np.concatenate(real_list)

    print(f"shape of generated images = {fake_np.shape}")
    print(f"shape of real images = {real_np.shape}")

    np.save(name_fake_test, fake_np)
    np.save(name_real_test, real_np)


def delete_nan_samples(z):
    return z[~np.isnan(z).any(axis=1)]


def z_transform(z):
    return z.unsqueeze(-1).unsqueeze(-1)


def calculate_images_statistics(z_agg_step, G,
                                device, batch_size,
                                path_to_save,
                                path_to_save_np,
                                method_name,
                                random_seed=42,
                                every_step=50,
                                use_conditional_model=False,
                                use_generator=True,
                                dataset="cifar10",
                                calculate_is=True,
                                calculate_msid_train=False,
                                calculate_msid_test=False,
                                z_transform=z_transform,
                                **kwargs):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if dataset == "cifar10":
        test_dataset = dset.CIFAR10(root=path_to_save,
                                    train=False,
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.Resize(32),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5)),
                                    ]))
        train_dataset = dset.CIFAR10(root=path_to_save,
                                     train=True,
                                     download=True,
                                     transform=transforms.Compose([
                                         transforms.Resize(32),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5),
                                                              (0.5, 0.5, 0.5)),
                                     ]))
    elif dataset == "mnist":
        test_dataset = dset.MNIST(root=path_to_save,
                                  train=False,
                                  download=True,
                                  transform=transforms.Compose([
                                       transforms.Resize(28),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),
                                  ]))
        train_dataset = dset.MNIST(root=path_to_save,
                                   train=True,
                                   download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize(28),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),
                                   ]))
    elif dataset == "celeba":
        image_size = kwargs["image_size"]
        dataroot = kwargs["dataroot"]
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
    else:
        raise ValueError("We support now only CIFAR10 and MNIST datasets")

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4)

    batch_size_resnet = 50
    dim_resnet = 2048
    model_type = 'inception'
    cuda = True
    inception_scores_mean = []
    inception_scores_std = []
    fid_scores_mean_train = []
    fid_scores_mean_test = []
    fid_scores_std_train = []
    fid_scores_std_test = []
    msid_scores_mean_train = []
    msid_scores_mean_test = []
    msid_scores_std_train = []
    msid_scores_std_test = []
    num_steps = z_agg_step.shape[0]
    for i in range(num_steps):
        print("------------------------------------")
        print(f"step = {i * every_step}")
        current_samples = z_agg_step[i]
        print(f"sample size = {current_samples.shape}")
        if use_generator:
            no_nans_samples = delete_nan_samples(current_samples)
            print(f"sample size after deleteting nans = {no_nans_samples.shape}")
        else:
            no_nans_samples = current_samples
        nsamples = len(no_nans_samples)
        latent_arr = torch.FloatTensor(no_nans_samples)
        if not use_conditional_model:
            if use_generator:
                latent_arr_transform = z_transform(latent_arr).unsqueeze(dim=1)
            else:
                latent_arr_transform = latent_arr
        else:
            raise ValueError("We support now only not conditional model")
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        use_grayscale = False
        if dataset == "cifar10":
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            use_grayscale = False
        elif dataset == "mnist":
            mean = (0.5,)
            std = (0.5,)
            use_grayscale = True

        latent_dataset = LatentFixDataset(latent_arr_transform, G,
                                          device, nsamples, use_generator,
                                          mean=mean, std=std,
                                          use_grayscale=use_grayscale)
        if calculate_is:
            print("start to calculate inception score...")
            start = time.time()
            (inception_score_mean,
             inception_score_std) = inception_score(latent_dataset,
                                                    device,
                                                    batch_size, True)
            print(
                f"{method_name} mean inception score = {inception_score_mean}, "
                f"std inception score = {inception_score_std}"
            )
            end = round(time.time() - start, 3)
            print(f"time for inception calculation = {end}s")
        else:
            inception_score_mean = -1.0
            inception_score_std = -1.0
        inception_scores_mean.append(inception_score_mean)
        inception_scores_std.append(inception_score_std)
        if use_generator:
            latent_arr_transform = z_transform(latent_arr)
        else:
            latent_arr_transform = latent_arr
        print(f"start to calculate FID score for test {dataset}...")
        start = time.time()
        name_fake_test = os.path.join(path_to_save_np,
                                      f"{method_name}_pretrained_fake_test_step_{i * every_step}.npy")
        name_real_test = os.path.join(path_to_save_np,
                                      f"{method_name}_pretrained_real_test_step_{i * every_step}.npy")
        save_images_fix_latent(G,
                               test_dataloader,
                               name_fake_test,
                               name_real_test,
                               latent_arr_transform,
                               device,
                               random_seed,
                               use_generator,
                               use_grayscale=use_grayscale)
        paths_to_test_method = [name_real_test, name_fake_test]

        results_fid_test = calculate_stat_given_paths(paths_to_test_method,
                                                      batch_size_resnet,
                                                      cuda,
                                                      dim_resnet,
                                                      model_type=model_type,
                                                      metric='fid')
        results_fid_test = results_fid_test[0]

        mean_fid_test = results_fid_test[1]
        std_fid_test = results_fid_test[2]
        fid_scores_mean_test.append(mean_fid_test)
        fid_scores_std_test.append(std_fid_test)
        print(f"FID score for test {dataset} with {method_name}: mean {mean_fid_test}, std {std_fid_test}")
        end = round(time.time() - start, 3)

        print(f"time for FID calculation on test = {end}s")
        if calculate_msid_test:
            print(f"start to calculate MSID score for test {dataset}...")
            start = time.time()
            results_msid_test = calculate_stat_given_paths(paths_to_test_method,
                                                           batch_size_resnet,
                                                           cuda,
                                                           dim_resnet,
                                                           model_type=model_type,
                                                           metric='msid')
            results_msid_test = results_msid_test[0]
            mean_msid_test = results_msid_test[1]
            std_msid_test = results_msid_test[2]
            print(f"MSID score for test {dataset} with {method_name}: mean {mean_msid_test}, std {std_msid_test}")
            end = round(time.time() - start, 3)
            print(f"time for MSID calculation on test = {end}s")
        else:
            mean_msid_test = -1.0
            std_msid_test = -1.0
        msid_scores_mean_test.append(mean_msid_test)
        msid_scores_std_test.append(std_msid_test)

        print(f"start to calculate FID score for train {dataset}...")
        start = time.time()
        name_fake_train = os.path.join(path_to_save_np,
                                       f"{method_name}_pretrained_fake_train_step_{i * every_step}.npy")
        name_real_train = os.path.join(path_to_save_np,
                                       f"{method_name}_pretrained_real_train_step_{i * every_step}.npy")
        save_images_fix_latent(G,
                               train_dataloader,
                               name_fake_train,
                               name_real_train,
                               latent_arr_transform,
                               device,
                               random_seed,
                               use_generator,
                               use_grayscale=use_grayscale)
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
        print(f"FID score for train {dataset} with {method_name}: mean {mean_fid_train}, std {std_fid_train}")
        fid_scores_mean_train.append(mean_fid_train)
        fid_scores_std_train.append(std_fid_train)
        end = round(time.time() - start, 3)
        print(f"time for FID calculation on train = {end}s")

        if calculate_msid_train:
            print(f"start to calculate MSID score for train {dataset}...")
            start = time.time()
            results_msid_train = calculate_stat_given_paths(paths_to_train_method,
                                                            batch_size_resnet,
                                                            cuda,
                                                            dim_resnet,
                                                            model_type=model_type,
                                                            metric='msid')
            results_msid_train = results_msid_train[0]
            mean_msid_train = results_msid_train[1]
            std_msid_train = results_msid_train[2]

            print(f"MSID score for train {dataset} with {method_name}: mean {mean_msid_train}, std {std_msid_train}")
            end = round(time.time() - start, 3)
            print(f"time for MSID calculation on test = {end}s")
        else:
            mean_msid_train = -1.0
            std_msid_train = -1.0
        msid_scores_mean_train.append(mean_msid_train)
        msid_scores_std_train.append(std_msid_train)

    inception_scores_mean = np.array(inception_scores_mean)
    inception_scores_std = np.array(inception_scores_std)
    fid_scores_mean_train = np.array(fid_scores_mean_train)
    fid_scores_mean_test = np.array(fid_scores_mean_test)
    fid_scores_std_train = np.array(fid_scores_std_train)
    fid_scores_std_test = np.array(fid_scores_std_test)
    msid_scores_mean_train = np.array(msid_scores_mean_train)
    msid_scores_mean_test = np.array(msid_scores_mean_test)
    msid_scores_std_train = np.array(msid_scores_std_train)
    msid_scores_std_test = np.array(msid_scores_std_test)
    names_list = ["inception_scores_mean", "inception_scores_std",
                  "fid_scores_mean_train", "fid_scores_mean_test",
                  "fid_scores_std_train", "fid_scores_std_test",
                  "msid_scores_mean_train", "msid_scores_mean_test",
                  "msid_scores_std_train", "msid_scores_std_test"
                  ]
    arrays_list = [inception_scores_mean, inception_scores_std,
                   fid_scores_mean_train, fid_scores_mean_test,
                   fid_scores_std_train, fid_scores_std_test,
                   msid_scores_mean_train, msid_scores_mean_test,
                   msid_scores_std_train, msid_scores_std_test
                   ]
    dict_results = {}
    for i in range(len(names_list)):
        cur_score_path = os.path.join(path_to_save_np,
                                      f"{method_name}_{names_list[i]}.npy")
        np.save(cur_score_path, arrays_list[i])
        dict_results[names_list[i]] = arrays_list[i]

    return dict_results


def load_dict_stats(method_name, path_to_save):
    names_list = ["inception_scores_mean", "inception_scores_std",
                  "fid_scores_mean_train", "fid_scores_mean_test",
                  "fid_scores_std_train", "fid_scores_std_test",
                  "msid_scores_mean_train", "msid_scores_mean_test",
                  "msid_scores_std_train", "msid_scores_std_test"
                  ]

    dict_results = {}
    for i in range(len(names_list)):
        cur_score_path = os.path.join(path_to_save,
                                      f"{method_name}_{names_list[i]}.npy")
        arrays_list = np.load(cur_score_path)
        dict_results[names_list[i]] = arrays_list

    return dict_results


def plot_scores_dynamics(scores,
                         every_step,
                         method_name,
                         path_to_save,
                         start_plot=None,
                         fig_plot=None,
                         figsize=(12, 8),
                         stds_coef=[2.0, 2.0, 2.0, 1.0],
                         plot_is_fid_msid=[True, True, True, False],
                         file_names=None,
                         color_conf="C0",
                         color_mean="orange"):
    xlabels = ["IS", "FID", "FID", "MSID"]
    mean_names = ["inception_scores_mean", "fid_scores_mean_train",
                  "fid_scores_mean_test", "msid_scores_mean_test"]
    std_names = ["inception_scores_std", "fid_scores_std_train",
                 "fid_scores_std_test", "msid_scores_std_test"]
    titles = ["Inception score", "Frechet Inception distance",
              "Frechet Inception distance", "MSID score"]
    train_test_mode = ["", "on train", "on test", "on test"]

    if start_plot is None:
        num_plots = sum(plot_is_fid_msid)
        fig, axs = plt.subplots(nrows=num_plots, figsize=(figsize[0], num_plots * figsize[1]))

    else:
        axs = start_plot
        fig = fig_plot

    ind_plot = 0
    cur_file_names = []
    for i in range(4):
        if plot_is_fid_msid[i]:
            range_steps = [i * every_step for i in range(len(scores[mean_names[i]]))]
            axs[ind_plot].plot(range_steps, scores[mean_names[i]], c=color_mean,
                               label=f'mean {xlabels[i]} for {method_name} {train_test_mode[i]}')
            axs[ind_plot].fill_between(range_steps,
                                       scores[mean_names[i]] - stds_coef[i] * scores[std_names[i]],
                                       scores[mean_names[i]] + stds_coef[i] * scores[std_names[i]],
                                       color=color_conf,
                                       alpha=0.2,
                                       label=fr'Intervals for mean {xlabels[i]} {method_name}: mean $\pm \; {stds_coef[i]} \cdot$ std')
            if start_plot is None:
                axs[ind_plot].set_xlabel("number of steps")
                axs[ind_plot].set_ylabel(xlabels[i])
                axs[ind_plot].set_title(f"{titles[i]} dynamics for {method_name}")
                axs[ind_plot].grid(True)
                name_mean = f'{method_name}_{mean_names[i]}.pdf'
            else:
                title = axs[ind_plot].get_title()
                title += f", {method_name}"
                axs[ind_plot].set_title(title)
                name_mean = f"{method_name}_" + file_names[ind_plot]
            axs[ind_plot].legend()
            cur_file_names.append(name_mean)
            name_mean_path = os.path.join(path_to_save, name_mean)
            fig.savefig(name_mean_path)
            ind_plot += 1

    return axs, fig, cur_file_names


def plot_old_scores_dynamics(scores,
                         every_step, method_name,
                         figsize,
                         path_to_save,
                         grad_step,
                         eps_scale,
                         coef=2.0,
                         plot_is=True,
                         plot_fid_train=True,
                         plot_fid_test=True):
    if plot_is:
        plt.figure(figsize=figsize)
        plt.xlabel("number of steps")
        plt.ylabel("IS")
        range_steps = [i * every_step for i in range(len(scores["inception_scores_mean"]))]
        plt.plot(range_steps, scores["inception_scores_mean"], c='orange',
                 label=f'mean IS for {method_name}')
        plt.fill_between(range_steps,
                         scores["inception_scores_mean"] - coef * scores["inception_scores_std"],
                         scores["inception_scores_mean"] + coef * scores["inception_scores_std"],
                         color="C0",
                         alpha=0.2, label=fr'confidence interval for mean IS: mean $\pm \; 2 \cdot$ std')
        plt.title(f"Inception score dynamics for {method_name}")
        plt.legend()
        plt.grid(True)
        name_inception_mean = f'{method_name}_eps_{grad_step}_noise_scale_{eps_scale}_inception_mean.pdf'
        name_inception_mean_path = os.path.join(path_to_save, name_inception_mean)
        plt.savefig(name_inception_mean_path)
        plt.show()
    if plot_fid_train:
        range_steps = [i * every_step for i in range(len(scores["fid_scores_mean_train"]))]
        plt.figure(figsize=figsize)
        plt.xlabel("number of steps")
        plt.ylabel("FID")
        plt.plot(range_steps, scores["fid_scores_mean_train"], c='orange',
                 label=f'mean FID for {method_name} on train')
        plt.fill_between(range_steps,
                         scores["fid_scores_mean_train"] - coef * scores["fid_scores_std_train"],
                         scores["fid_scores_mean_train"] + coef * scores["fid_scores_std_train"],
                         color="C0",
                         alpha=0.2,
                         label=fr'confidence interval for mean FID on train: mean $\pm \; 2 \cdot$ std')
        plt.title(f"Frechet Inception distance dynamics for {method_name}")
        plt.legend()
        plt.grid(True)
        name_fid_mean = f'{method_name}_eps_{grad_step}_noise_scale_{eps_scale}_fid_mean_train.pdf'
        name_fid_mean_path = os.path.join(path_to_save, name_fid_mean)
        plt.savefig(name_fid_mean_path)
        plt.show()
    if plot_fid_test:
        range_steps = [i * every_step for i in range(len(scores["fid_scores_mean_test"]))]
        plt.figure(figsize=figsize)
        plt.xlabel("number of steps")
        plt.ylabel("FID")
        plt.plot(range_steps, scores["fid_scores_mean_test"], c='green',
                 label=f'mean FID for {method_name} on test')
        plt.fill_between(range_steps,
                         scores["fid_scores_mean_test"] - coef * scores["fid_scores_std_test"],
                         scores["fid_scores_mean_test"] + coef * scores["fid_scores_std_test"],
                         color="C2",
                         alpha=0.2,
                         label=fr'confidence interval for mean FID on test: mean $\pm \; 2 \cdot$ std')
        plt.title(f"Frechet Inception distance dynamics for {method_name}")
        plt.legend()
        plt.grid(True)
        name_fid_mean = f'{method_name}_eps_{grad_step}_noise_scale_{eps_scale}_fid_mean_test.pdf'
        name_fid_mean_path = os.path.join(path_to_save, name_fid_mean)
        plt.savefig(name_fid_mean_path)
        plt.show()
