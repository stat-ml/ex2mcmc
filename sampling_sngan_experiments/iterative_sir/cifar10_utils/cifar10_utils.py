import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from dataloader import LatentFixDataset
from fid_score import calculate_fid_given_paths
from metrics import inception_score
from tqdm import tqdm


cwd = os.getcwd()
api_path_sampling = os.path.join(cwd, "..", "sampling_utils")
api_path_gan_metrics = os.path.join(cwd, "..", "gan_metrics")
sys.path.append(api_path_sampling)
sys.path.append(api_path_gan_metrics)


def save_images_for_fid(
    G,
    real_dataloader,
    name_fake_test,
    name_real_test,
    z_dim,
    device,
    random_seed,
):
    fake_list = []
    real_list = []
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    for i, data_real in enumerate(real_dataloader, 0):
        batch_real = data_real[0]
        batch_size = batch_real.shape[0]
        fixed_noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_images = G(fixed_noise)
        fake_norm_np = ((1.0 + fake_images) / 2).detach().cpu().numpy()
        real_norm_np = ((1.0 + batch_real) / 2).detach().cpu().numpy()
        fake_list.append(fake_norm_np)
        real_list.append(real_norm_np)

    fake_np = np.concatenate(fake_list)
    real_np = np.concatenate(real_list)

    np.save(name_fake_test, fake_np)
    np.save(name_real_test, real_np)


def plot_images(images_torch, figsize=(10, 10)):
    batch_size_sample = images_torch.shape[0]
    numpy_images = images_torch.detach().cpu().numpy().transpose(0, 2, 3, 1)
    numpy_images = (numpy_images - numpy_images.min()) / (
        numpy_images.max() - numpy_images.min()
    )
    nrow = int(batch_size_sample ** 0.5)
    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(nrow, nrow)
    for k in range(batch_size_sample):
        i = k // nrow
        j = k % nrow
        # axes[i][j].imshow(np.clip(numpy_images[k], 0, 1))
        axes[i][j].imshow(numpy_images[k])
        axes[i][j].axis("off")
    plt.show()


def delete_local_files(path_to_save_cifar10, name_fake_test, name_real_test):
    downloaded_files = os.path.join(path_to_save_cifar10, "cifar-10*")
    cmdline1 = f"rm -r {downloaded_files}"
    os.system(cmdline1)
    cmdline2 = f"rm {name_fake_test} {name_real_test}"
    os.system(cmdline2)


def delete_nan_samples(z):
    return z[~np.isnan(z).any(axis=1)]


def z_transform(z):
    return z.unsqueeze(-1).unsqueeze(-1)


def save_images_for_fid_fix_latent(
    G,
    real_dataloader,
    name_fake_test,
    name_real_test,
    latent_arr,
    device,
    random_seed,
):
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
            fixed_noise = latent_arr[start_ind : start_ind + batch_size].to(
                device,
            )
            fake_images = G(fixed_noise).clamp(-1, 1)
            start_ind += batch_size

            fake_norm_np = ((1.0 + fake_images) / 2).detach().cpu().numpy()
            real_norm_np = ((1.0 + batch_real) / 2).detach().cpu().numpy()
            fake_list.append(fake_norm_np)
            real_list.append(real_norm_np)

        else:
            fixed_noise = latent_arr[start_ind:].to(device)
            fake_images = G(fixed_noise).clamp(-1, 1)
            add_num_imgs = max_num_imgs - start_ind
            batch_real = batch_real[:add_num_imgs]

            fake_norm_np = ((1.0 + fake_images) / 2).detach().cpu().numpy()
            real_norm_np = ((1.0 + batch_real) / 2).detach().cpu().numpy()
            fake_list.append(fake_norm_np)
            real_list.append(real_norm_np)
            break

    fake_np = np.concatenate(fake_list)
    real_np = np.concatenate(real_list)

    print(f"shape of generated images = {fake_np.shape}")
    print(f"shape of real images = {real_np.shape}")

    np.save(name_fake_test, fake_np)
    np.save(name_real_test, real_np)


def delete_saved_files_for_cifar10_statistics(
    z_agg_step,
    path_to_save_cifar10_np,
    method_name,
    every_step=50,
):
    num_steps = z_agg_step.shape[0]
    for i in range(num_steps):
        name_fake_test = os.path.join(
            path_to_save_cifar10_np,
            f"{method_name}_pretrained_fake_test_step_{i*every_step}.npy",
        )
        name_real_test = os.path.join(
            path_to_save_cifar10_np,
            f"{method_name}_pretrained_real_test_step_{i*every_step}.npy",
        )
        name_fake_train = os.path.join(
            path_to_save_cifar10_np,
            f"{method_name}_pretrained_fake_train_step_{i*every_step}.npy",
        )
        name_real_train = os.path.join(
            path_to_save_cifar10_np,
            f"{method_name}_pretrained_real_train_step_{i*every_step}.npy",
        )
        os.remove(name_fake_test)
        os.remove(name_real_test)
        os.remove(name_fake_train)
        os.remove(name_real_train)


def calculate_cifar10_statistics(
    z_agg_step,
    G,
    device,
    batch_size,
    path_to_save_cifar10,
    path_to_save_cifar10_np,
    method_name,
    random_seed=42,
    every_step=50,
    use_conditional_model=False,
):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    test_dataset = dset.CIFAR10(
        root=path_to_save_cifar10,
        train=False,
        download=False,
        transform=transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ],
        ),
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    train_dataset = dset.CIFAR10(
        root=path_to_save_cifar10,
        train=True,
        download=False,
        transform=transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ],
        ),
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    batch_size_resnet = 50
    dim_resnet = 2048
    model_type = "inception"
    cuda = True
    inception_scores_mean = []
    inception_scores_std = []
    fid_scores_mean_train = []
    fid_scores_mean_test = []
    fid_scores_std_train = []
    fid_scores_std_test = []
    num_steps = z_agg_step.shape[0]
    for i in range(num_steps):
        print("------------------------------------")
        print(f"step = {i*every_step}")
        current_samples = z_agg_step[i]
        print(f"sample size = {current_samples.shape}")
        no_nans_samples = delete_nan_samples(current_samples)
        print(f"sample size after deleteting nans = {no_nans_samples.shape}")
        nsamples = len(no_nans_samples)
        latent_arr = torch.FloatTensor(no_nans_samples)
        if not use_conditional_model:
            latent_arr_transform = z_transform(latent_arr).unsqueeze(dim=1)
            latent_dataset = LatentFixDataset(
                latent_arr_transform,
                G,
                device,
                nsamples,
            )

        else:
            do_smth = True
        print("start to calculate inception score...")
        start = time.time()
        (inception_score_mean, inception_score_std) = inception_score(
            latent_dataset,
            device,
            batch_size,
            True,
        )
        print(
            f"{method_name} mean inception score = {inception_score_mean}, std inception score = {inception_score_std}",
        )
        inception_scores_mean.append(inception_score_mean)
        inception_scores_std.append(inception_score_std)
        end = round(time.time() - start, 3)
        print(f"time for inception calculation = {end}s")
        latent_arr_transform = z_transform(latent_arr)
        print("start to calculate FID score for test CIFAR10...")
        start = time.time()
        name_fake_test = os.path.join(
            path_to_save_cifar10_np,
            f"{method_name}_pretrained_fake_test_step_{i*every_step}.npy",
        )
        name_real_test = os.path.join(
            path_to_save_cifar10_np,
            f"{method_name}_pretrained_real_test_step_{i*every_step}.npy",
        )
        save_images_for_fid_fix_latent(
            G,
            test_dataloader,
            name_fake_test,
            name_real_test,
            latent_arr_transform,
            device,
            random_seed,
        )
        paths_to_test_method = [name_real_test, name_fake_test]

        results_fid_test = calculate_fid_given_paths(
            paths_to_test_method,
            batch_size_resnet,
            cuda,
            dim_resnet,
            model_type=model_type,
        )
        results_fid_test = results_fid_test[0]

        mean_fid_test = results_fid_test[1]
        std_fid_test = results_fid_test[2]
        fid_scores_mean_test.append(mean_fid_test)
        fid_scores_std_test.append(std_fid_test)
        print(
            f"FID score for test CIFAR10 with {method_name}: mean {mean_fid_test}, score {std_fid_test}",
        )
        end = round(time.time() - start, 3)

        print(f"time for FID calculation on test = {end}s")
        print("start to calculate FID score for train CIFAR10...")
        start = time.time()
        name_fake_train = os.path.join(
            path_to_save_cifar10_np,
            f"{method_name}_pretrained_fake_train_step_{i*every_step}.npy",
        )
        name_real_train = os.path.join(
            path_to_save_cifar10_np,
            f"{method_name}_pretrained_real_train_step_{i*every_step}.npy",
        )
        save_images_for_fid_fix_latent(
            G,
            train_dataloader,
            name_fake_train,
            name_real_train,
            latent_arr_transform,
            device,
            random_seed,
        )
        paths_to_train_method = [name_real_train, name_fake_train]
        results_fid_train = calculate_fid_given_paths(
            paths_to_train_method,
            batch_size_resnet,
            cuda,
            dim_resnet,
            model_type=model_type,
        )
        results_fid_train = results_fid_train[0]

        mean_fid_train = results_fid_train[1]
        std_fid_train = results_fid_train[2]
        print(
            f"FID score for train CIFAR10 with {method_name}: mean {mean_fid_train}, score {std_fid_train}",
        )
        fid_scores_mean_train.append(mean_fid_train)
        fid_scores_std_train.append(std_fid_train)
        end = round(time.time() - start, 3)
        print(f"time for FID calculation on train = {end}s")

    inception_scores_mean = np.array(inception_scores_mean)
    inception_scores_std = np.array(inception_scores_std)
    fid_scores_mean_train = np.array(fid_scores_mean_train)
    fid_scores_mean_test = np.array(fid_scores_mean_test)
    fid_scores_std_train = np.array(fid_scores_std_train)
    fid_scores_std_test = np.array(fid_scores_std_test)
    names_list = [
        "inception_scores_mean",
        "inception_scores_std",
        "fid_scores_mean_train",
        "fid_scores_mean_test",
        "fid_scores_std_train",
        "fid_scores_std_test",
    ]
    arrays_list = [
        inception_scores_mean,
        inception_scores_std,
        fid_scores_mean_train,
        fid_scores_mean_test,
        fid_scores_std_train,
        fid_scores_std_test,
    ]
    dict_results = {}
    for i in range(len(names_list)):
        cur_score_path = os.path.join(
            path_to_save_cifar10_np,
            f"{method_name}_{names_list[i]}.npy",
        )
        np.save(cur_score_path, arrays_list[i])
        dict_results[names_list[i]] = arrays_list[i]

    return dict_results


def save_fid_inception_to_np(
    method_name,
    n_steps,
    grad_step,
    eps_scale,
    path_to_np_files,
    scores,
):
    for k in scores.keys():
        file_name = f"{method_name}_eps_{grad_step}_noise_scale_{eps_scale}_nsteps_{n_steps}_{k}.npy"
        path_to_save_file = os.path.join(path_to_np_files, file_name)
        np.save(path_to_save_file, scores[k])


def plot_scores_cifar10_dynamics(
    scores,
    every_step,
    method_name,
    figsize,
    path_to_save,
    grad_step,
    eps_scale,
    coef=2.0,
    plot_is=True,
    plot_fid_train=True,
    plot_fid_test=True,
):
    if plot_is:
        plt.figure(figsize=figsize)
        plt.xlabel("number of steps")
        plt.ylabel("IS")
        range_steps = [
            i * every_step for i in range(len(scores["inception_scores_mean"]))
        ]
        plt.plot(
            range_steps,
            scores["inception_scores_mean"],
            c="orange",
            label=f"mean IS for {method_name}",
        )
        plt.fill_between(
            range_steps,
            scores["inception_scores_mean"]
            - coef * scores["inception_scores_std"],
            scores["inception_scores_mean"]
            + coef * scores["inception_scores_std"],
            color="C0",
            alpha=0.2,
            label=fr"confidence interval for mean IS: mean $\pm \; 2 \cdot$ std",
        )
        plt.title(f"Inception score dynamics for {method_name}")
        plt.legend()
        plt.grid(True)
        name_inception_mean = f"{method_name}_eps_{grad_step}_noise_scale_{eps_scale}_inception_mean.pdf"
        name_inception_mean_path = os.path.join(
            path_to_save,
            name_inception_mean,
        )
        plt.savefig(name_inception_mean_path)
        plt.show()
    if plot_fid_train:
        range_steps = [
            i * every_step for i in range(len(scores["fid_scores_mean_train"]))
        ]
        plt.figure(figsize=figsize)
        plt.xlabel("number of steps")
        plt.ylabel("FID")
        plt.plot(
            range_steps,
            scores["fid_scores_mean_train"],
            c="orange",
            label=f"mean FID for {method_name} on train",
        )
        plt.fill_between(
            range_steps,
            scores["fid_scores_mean_train"]
            - coef * scores["fid_scores_std_train"],
            scores["fid_scores_mean_train"]
            + coef * scores["fid_scores_std_train"],
            color="C0",
            alpha=0.2,
            label=fr"confidence interval for mean FID on train: mean $\pm \; 2 \cdot$ std",
        )
        plt.title(f"Frechet Inception distance dynamics for {method_name}")
        plt.legend()
        plt.grid(True)
        name_fid_mean = f"{method_name}_eps_{grad_step}_noise_scale_{eps_scale}_fid_mean_train.pdf"
        name_fid_mean_path = os.path.join(path_to_save, name_fid_mean)
        plt.savefig(name_fid_mean_path)
        plt.show()
    if plot_fid_test:
        range_steps = [
            i * every_step for i in range(len(scores["fid_scores_mean_test"]))
        ]
        plt.figure(figsize=figsize)
        plt.xlabel("number of steps")
        plt.ylabel("FID")
        plt.plot(
            range_steps,
            scores["fid_scores_mean_test"],
            c="green",
            label=f"mean FID for {method_name} on test",
        )
        plt.fill_between(
            range_steps,
            scores["fid_scores_mean_test"]
            - coef * scores["fid_scores_std_test"],
            scores["fid_scores_mean_test"]
            + coef * scores["fid_scores_std_test"],
            color="C2",
            alpha=0.2,
            label=fr"confidence interval for mean FID on test: mean $\pm \; 2 \cdot$ std",
        )
        plt.title(f"Frechet Inception distance dynamics for {method_name}")
        plt.legend()
        plt.grid(True)
        name_fid_mean = f"{method_name}_eps_{grad_step}_noise_scale_{eps_scale}_fid_mean_test.pdf"
        name_fid_mean_path = os.path.join(path_to_save, name_fid_mean)
        plt.savefig(name_fid_mean_path)
        plt.show()
