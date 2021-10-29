import numpy as np
import random
import torch, torch.nn as nn
from torch.autograd import Variable
from torch import autograd
import time
import datetime
import os

import sys
sys.path.append("../sampling_utils")

from toy_examples_utils import prepare_train_batches
from visualization import epoch_visualization

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device_default = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_gradient_penalty(D, real_data, fake_data, batch_size, Lambda = 0.1,
                          device = device_default):
    #print(real_data.shape)
    #print(fake_data.shape)
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size()).to(device)

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, 
                                     requires_grad = True)
    discriminator_interpolates = D(interpolates)
    ones = torch.ones(discriminator_interpolates.size()).to(device)
    gradients = autograd.grad(outputs = discriminator_interpolates, 
                              inputs = interpolates,
                              grad_outputs = ones,
                              create_graph = True, 
                              retain_graph = True, 
                              only_inputs = True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean() * Lambda
    return gradient_penalty
    
def d_loss(fake, real, loss_type = 'Jensen_nonsaturing', criterion=None):
    if loss_type == 'Hinge':
        max_module = nn.ReLU()
        return max_module(1.0 - real).mean() + max_module(1.0 + fake).mean()
      
    elif loss_type == 'Wasserstein':
        return fake.mean() - real.mean()

    elif (loss_type.split('_')[0] == 'Jensen') and (criterion is not None):
        one_b = torch.ones_like(fake).to(fake.device)
        d_real_loss = criterion(real, one_b)
        d_fake_loss = criterion(fake, 1.-one_b)
        return d_real_loss + d_fake_loss         

        #fake_sigmoid = fake.sigmoid()
        #real_sigmoid = real.sigmoid()
        #return -torch.log(1. - fake_sigmoid).mean() - torch.log(real_sigmoid).mean()
        
    else:
       raise TypeError('Unknown loss type')

def g_loss(fake, loss_type = 'Jensen_nonsaturing', criterion=None):
    if loss_type in ['Hinge', 'Wasserstein']:
       return -fake.mean()
    elif (loss_type == 'Jensen_minimax') and (criterion is not None):
       #fake_sigmoid = fake.sigmoid()
       #return torch.log(1. - fake_sigmoid).mean()
       zeros = torch.zeros_like(fake).to(fake.device)
       return -criterion(fake, zeros)
    elif (loss_type == 'Jensen_nonsaturing') and (criterion is not None):
       #fake_sigmoid = fake.sigmoid()
       #return torch.log(1. - fake_sigmoid).mean()
       ones = torch.ones_like(fake).to(fake.device)
       return criterion(fake, ones)
    else:
       raise TypeError('Unknown loss type')

def train_gan(X_train,
              train_dataloader, 
              generator, g_optimizer, 
              discriminator, d_optimizer,
              loss_type = 'Jensen_nonsaturing',
              batch_size = 256,
              device = device_default,
              use_gradient_penalty = True,
              Lambda = 0.1,
              num_epochs = 200, 
              num_epoch_for_save = 10,
              batch_size_sample = 5000,
              k_g = 1,
              k_d = 10,
              n_calib_pts = 10000,
              normalize_to_0_1 = True,
              scaler = None,
              mode = '25_gaussians',
              path_to_logs = None,
              path_to_models = None,
              path_to_plots = None,
              path_to_save_remote = None,
              port_to_remote = None,
              proj_list = None,
              plot_mhgan = False):

    generator_loss_arr = []
    generator_mean_loss_arr = []
    discriminator_loss_arr = []
    discriminator_mean_loss_arr = []
    #one = torch.tensor(1, dtype = torch.float).to(device)
    if (loss_type.split('_')[0] == 'Jensen') and (normalize_to_0_1):
       criterion = nn.BCEWithLogitsLoss()
    elif (loss_type.split('_')[0] == 'Jensen') and (not normalize_to_0_1):
       criterion = nn.BCELoss()
    else:
       criterion = None

    try:
        for epoch in range(num_epochs):
            print(f"Start epoch = {epoch}")

            for p in discriminator.parameters():  # reset requires_grad
                p.requires_grad = True
            
            current_d_step = 1
            start_time = time.time()
            # Optimize D
            # discriminator.train(True)
            # generator.train(False)
            for data in train_dataloader:
                cur_batch_size = data.shape[0]
                discriminator.zero_grad()
                real_data = autograd.Variable(data).to(device)
                noise = generator.make_hidden(batch_size=cur_batch_size)
                noise = autograd.Variable(noise).to(device)
                fake_data = generator(noise)
                

                d_real_data = discriminator(real_data)
                d_fake_data = discriminator(fake_data)
                discriminator_loss = d_loss(d_fake_data, 
                                            d_real_data, 
                                            loss_type = loss_type,
                                            criterion = criterion)
                #print("OK")
                discriminator_loss.backward()

                if (use_gradient_penalty) and (Lambda > 0):
                    gradient_penalty = calc_gradient_penalty(discriminator, 
                                                             real_data.data, 
                                                             fake_data.data, 
                                                             cur_batch_size,
                                                             Lambda)
                    gradient_penalty.backward()
                    discriminator_loss += gradient_penalty
                d_optimizer.step()
                discriminator_loss_arr.append(discriminator_loss.data.cpu().item())

                if current_d_step < k_d:
                    current_d_step += 1
                    continue
                else:
                    current_d_step = 1

                 #discriminator.train(False)
                 #generator.train(True)
                 # Optimize G
                for p in discriminator.parameters():  # to avoid computation
                    p.requires_grad = False

                for _ in range(k_g):
                    g_optimizer.zero_grad()

                    # Do an update
                    noise = generator.make_hidden(batch_size=cur_batch_size)
                    noise = autograd.Variable(noise).to(device)
                    fake_data = generator(noise)
                    d_fake_data = discriminator(fake_data)

                    generator_loss = g_loss(d_fake_data, 
                                            loss_type = loss_type,
                                            criterion = criterion)
                    generator_loss.backward()
                    g_optimizer.step()
                    generator_loss_arr.append(generator_loss.data.cpu().item())
           
            end_time = time.time()
            calc_time = end_time - start_time
            discriminator_mean_loss_arr.append(np.mean(discriminator_loss_arr[-k_d :]))
            generator_mean_loss_arr.append(np.mean(generator_loss_arr[-k_g :]))
            time_msg = "Epoch {} of {} took {:.3f}s\n".format(epoch + 1, num_epochs, calc_time)
            discriminator_msg = "Discriminator last mean loss: \t{:.6f}\n".format(discriminator_mean_loss_arr[-1])
            generator_msg = "Generator last mean loss: \t{:.6f}\n".format(generator_mean_loss_arr[-1])
            print(time_msg)
            print(discriminator_msg)
            print(generator_msg)
            if path_to_logs is not None:
               f = open(path_to_logs, "a")
               f.write(time_msg)
               f.write(discriminator_msg)
               f.write(generator_msg)
               f.close()
 
            if epoch % num_epoch_for_save == 0:
               # Visualize
               epoch_visualization(X_train=X_train, 
                                   generator=generator, 
                                   discriminator=discriminator,
                                   use_gradient_penalty=use_gradient_penalty, 
                                   discriminator_mean_loss_arr=discriminator_mean_loss_arr, 
                                   epoch=epoch, 
                                   Lambda=Lambda,
                                   generator_mean_loss_arr=generator_mean_loss_arr, 
                                   path_to_save=path_to_plots,
                                   batch_size_sample=batch_size_sample,
                                   loss_type=loss_type,
                                   mode=mode,
                                   scaler=scaler,
                                   proj_list=proj_list,
                                   port_to_remote=port_to_remote, 
                                   path_to_save_remote=path_to_save_remote,
                                   n_calib_pts=n_calib_pts,
                                   normalize_to_0_1=normalize_to_0_1,
                                   plot_mhgan = plot_mhgan)
               if path_to_models is not None:
                  cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

                  discriminator_model_name = cur_time + '_discriminator.pth'
                  generator_model_name = cur_time + '_generator.pth'
                  path_to_discriminator = os.path.join(path_to_models, discriminator_model_name)
                  path_to_generator = os.path.join(path_to_models, generator_model_name)
              
                  torch.save(discriminator.state_dict(), path_to_discriminator)
                  torch.save(generator.state_dict(), path_to_generator)

                  discriminator_optimizer_name = cur_time + '_opt_discriminator.pth'
                  generator_optimizer_name = cur_time + '_opt_generator.pth'
                  path_to_opt_discriminator = os.path.join(path_to_models, discriminator_optimizer_name)
                  path_to_opt_generator = os.path.join(path_to_models, generator_optimizer_name)

                  torch.save(d_optimizer.state_dict(), path_to_opt_discriminator)
                  torch.save(g_optimizer.state_dict(), path_to_opt_generator)
                
    except KeyboardInterrupt:
        pass
