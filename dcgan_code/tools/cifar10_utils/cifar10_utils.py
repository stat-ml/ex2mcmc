import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import os


def save_images_for_fid(G,
                        real_dataloader,
                        name_fake_test,
                        name_real_test,
                        z_dim,
                        device,
                        random_seed):
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
        fake_norm_np = ((1. + fake_images) / 2).detach().cpu().numpy()
        real_norm_np = ((1. + batch_real) / 2).detach().cpu().numpy()
        fake_list.append(fake_norm_np)
        real_list.append(real_norm_np)

    fake_np = np.concatenate(fake_list)
    real_np = np.concatenate(real_list)

    np.save(name_fake_test, fake_np)
    np.save(name_real_test, real_np)


def plot_images(images_torch, figsize=(10, 10)):
    batch_size_sample = images_torch.shape[0]
    numpy_images = images_torch.detach().cpu().numpy().transpose(0, 2, 3, 1)
    numpy_images = (numpy_images - numpy_images.min()) / (numpy_images.max() - numpy_images.min())
    nrow = int(batch_size_sample ** 0.5)
    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(nrow, nrow)
    for k in range(batch_size_sample):
        i = k // nrow
        j = k % nrow
        # axes[i][j].imshow(np.clip(numpy_images[k], 0, 1))
        axes[i][j].imshow(numpy_images[k])
        axes[i][j].axis('off')
    plt.show()


def delete_local_files(path_to_save_cifar10,
                       name_fake_test,
                       name_real_test):
    downloaded_files = os.path.join(path_to_save_cifar10, 'cifar-10*')
    cmdline1 = f'rm -r {downloaded_files}'
    os.system(cmdline1)
    cmdline2 = f'rm {name_fake_test} {name_real_test}'
    os.system(cmdline2)


def delete_saved_files_for_cifar10_statistics(z_agg_step,
                                              path_to_save_cifar10_np,
                                              method_name,
                                              every_step=50):
    num_steps = z_agg_step.shape[0]
    for i in range(num_steps):
        name_fake_test = os.path.join(path_to_save_cifar10_np,
                                      f"{method_name}_pretrained_fake_test_step_{i * every_step}.npy")
        name_real_test = os.path.join(path_to_save_cifar10_np,
                                      f"{method_name}_pretrained_real_test_step_{i * every_step}.npy")
        name_fake_train = os.path.join(path_to_save_cifar10_np,
                                       f"{method_name}_pretrained_fake_train_step_{i * every_step}.npy")
        name_real_train = os.path.join(path_to_save_cifar10_np,
                                       f"{method_name}_pretrained_real_train_step_{i * every_step}.npy")
        os.remove(name_fake_test)
        os.remove(name_real_test)
        os.remove(name_fake_train)
        os.remove(name_real_train)


def save_fid_inception_to_np(method_name, n_steps,
                             grad_step,
                             eps_scale,
                             path_to_np_files,
                             scores):
    for k in scores.keys():
        file_name = f'{method_name}_eps_{grad_step}_noise_scale_{eps_scale}_nsteps_{n_steps}_{k}.npy'
        path_to_save_file = os.path.join(path_to_np_files, file_name)
        np.save(path_to_save_file, scores[k])
