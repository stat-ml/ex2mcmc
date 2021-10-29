"""
Code adapted from "Your GAN is Secretly an Energy-based Model and You Should use Discriminator Driven Latent Sampling".

@article{DBLP:journals/corr/abs-2003-06060,
  author    = {Tong Che and
               Ruixiang Zhang and
               Jascha Sohl{-}Dickstein and
               Hugo Larochelle and
               Liam Paull and
               Yuan Cao and
               Yoshua Bengio},
  title     = {Your {GAN} is Secretly an Energy-based Model and You Should use Discriminator
               Driven Latent Sampling},
  journal   = {CoRR},
  volume    = {abs/2003.06060},
  year      = {2020},
  url       = {https://arxiv.org/abs/2003.06060},
  eprinttype = {arXiv},
  eprint    = {2003.06060},
  timestamp = {Tue, 17 Mar 2020 14:18:27 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2003-06060.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

import os, sys, time
import shutil
import yaml

from matplotlib import pyplot as plt
import argparse
import chainer
import numpy as np
from chainer import training
import chainer.functions as F
from chainer import Variable
from chainer.training import extension
from chainer.training import extensions
from chainer import reporter as reporter_module

import source.yaml_utils as yaml_utils
from updater import loss_dcgan_dis, loss_dcgan_gen
from source.miscs.random_samples import sample_continuous, sample_categorical
from evaluation import sample_generate, sample_generate_conditional, sample_generate_light, calc_inception


class FineTuneUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.loss_type = kwargs.pop('loss_type')
        self.conditional = kwargs.pop('conditional')
        self.n_gen_samples = kwargs.pop('n_gen_samples')
        if self.loss_type == 'dcgan':
            self.loss_gen = loss_dcgan_gen
            self.loss_dis = loss_dcgan_dis
        else:
            raise ValueError("We have to fine-tune the discriminator with the dcgan loss")
        super(FineTuneUpdater, self).__init__(*args, **kwargs)

    def _generete_samples(self, n_gen_samples=None):
        if n_gen_samples is None:
            n_gen_samples = self.n_gen_samples
        gen = self.models['gen']
        if self.conditional:
            y = sample_categorical(gen.n_classes, n_gen_samples, xp=gen.xp)
        else:
            y = None
        x_fake = gen(n_gen_samples, y=y)
        return x_fake, y

    def get_batch(self, xp):
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x = []
        y = []
        for j in range(batchsize):
            x.append(np.asarray(batch[j][0]).astype("f"))
            y.append(np.asarray(batch[j][1]).astype(np.int32))
        x_real = Variable(xp.asarray(x))
        y_real = Variable(xp.asarray(y, dtype=xp.int32)) if self.conditional else None
        return x_real, y_real

    def update_core(self):
        gen = self.models['gen']
        dis = self.models['dis']
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = gen.xp

        x_real, y_real = self.get_batch(xp)
        batchsize = len(x_real)
        dis_real = dis(x_real, y=y_real)
        x_fake, y_fake = self._generete_samples(n_gen_samples=batchsize)
        dis_fake = dis(x_fake, y=y_fake)
        x_fake.unchain_backward()

        loss_dis = self.loss_dis(dis_fake=dis_fake, dis_real=dis_real)
        dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()
        acc_real = (dis_real.array > 0.).mean()
        acc_fake = (dis_fake.array < 0.).mean()
        chainer.reporter.report({'loss_dis': loss_dis})
        chainer.reporter.report({'acc_real': acc_real})
        chainer.reporter.report({'acc_fake': acc_fake})


class DiscriminatorEvaluator(chainer.training.extensions.Evaluator):
    def __init__(self, *args, **kwargs):
        self.conditional = kwargs.pop('conditional')
        super(DiscriminatorEvaluator, self).__init__(*args, **kwargs)
    
    def get_batch(self, batch, xp):
        batchsize = len(batch)
        x = []
        y = []
        for j in range(batchsize):
            x.append(np.asarray(batch[j][0]).astype("f"))
            y.append(np.asarray(batch[j][1]).astype(np.int32))
        x_real = Variable(xp.asarray(x))
        y_real = Variable(xp.asarray(y, dtype=xp.int32)) if self.conditional else None
        return x_real, y_real

    def eval_acc(self, d_real, d_fake):
        return (d_real.array > 0.).mean(), (d_fake.array < 0.).mean()

    def evaluate(self):
        gen = self.get_target('gen')
        dis = self.get_target('dis')
        its = self.get_all_iterators()
        xp = gen.xp
        for it_name, it in its.items():
            if hasattr(it, 'reset'):
                it.reset()

        x_fake = gen(1000, y=None)
        print(dis(x_fake).mean())
        d = F.reshape(dis(x_fake), (-1,)).data.tolist()
        plt.hist(d, bins=20)
        plt.savefig('cal/dis_hist.png')
        plt.close()

        batch = its['val'].next()
        x = []
        y = []
        for j in range(len(batch)):
            x.append(np.asarray(batch[j][0]).astype("f"))
            y.append(np.asarray(batch[j][1]).astype(np.int32))
        x_real = Variable(xp.asarray(x))
        d = F.reshape(dis(x_real), (-1,)).data.tolist()
        plt.hist(d, bins=20)
        plt.savefig('cal/dis_hist2.png')
        plt.close()

        summary = chainer.DictSummary()
        with chainer.no_backprop_mode():
            observation = {}
            with reporter_module.report_scope(observation):
                for it_name, it in its.items():
                    for batch in it:
                        x_real, y_real = self.get_batch(batch, xp)
                        d_real = dis(x_real, y=y_real)
                        x_fake = gen(x_real.shape[0]) #, return_y=True)
                        y_fake = None # xp.zeros()
                        d_fake = dis(x_fake, y_fake)
                        acc_real, acc_fake = self.eval_acc(d_real, d_fake)
                        summary.add({'{}_acc_real'.format(it_name): acc_real,
                                     '{}_acc_fake'.format(it_name): acc_fake})
        return summary.compute_mean()


def load_models(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    dis_conf = config.models['binary_discriminator'] #'binary_discriminator']
    dis = yaml_utils.load_model(dis_conf['fn'], dis_conf['name'], dis_conf['args'])
    return gen, dis

def create_result_dir(result_dir, config_path, config):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    def copy_to_result_dir(fn, result_dir):
        bfn = os.path.basename(fn)
        shutil.copy(fn, '{}/{}'.format(result_dir, bfn))

    copy_to_result_dir(config_path, result_dir)
    copy_to_result_dir(
        config.models['generator']['fn'], result_dir)
    copy_to_result_dir(
        config.models['binary_discriminator']['fn'], result_dir)
    copy_to_result_dir(
        config.dataset['dataset_fn'], result_dir)
    

def make_optimizer(model, alpha=0.0002, beta1=0., beta2=0.9):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml', help='path to config file')
    parser.add_argument('--gpu', type=int, default=0, help='index of gpu to be used')
    parser.add_argument('--data_dir', type=str, default='./data/imagenet')
    parser.add_argument('--results_dir', type=str, default='./results/gans',
                        help='directory to save the results to')
    parser.add_argument('--inception_model_path', type=str, default='./datasets/inception_model',
                        help='path to the inception model')
    parser.add_argument('--snapshot', type=str, default='',
                        help='path to the snapshot')
    parser.add_argument('--loaderjob', type=int,
                        help='number of parallel data loading processes')
    parser.add_argument('--gen_ckpt', type=str,
                        help='path to the saved generator snapshot model file to load')
    parser.add_argument('--dis_ckpt', type=str,
                        help='path to the saved discriminator snapshot model file to load')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    chainer.cuda.get_device_from_id(args.gpu).use()

    # Dataset
    if config['dataset'][
        'dataset_name'] != 'CIFAR10Dataset':  # Cifar10 dataset handler does not take "root" as argument.
        config['dataset']['args']['root'] = args.data_dir
    train_dataset = yaml_utils.load_dataset(config)
    val_dataset = yaml_utils.load_dataset(config, test=True)

    # Iterator
    train_iterator = chainer.iterators.MultiprocessIterator(
        train_dataset, config.batchsize, n_processes=args.loaderjob)
    eval_train_iterator = chainer.iterators.MultiprocessIterator(
        train_dataset, config.batchsize, repeat=False, n_processes=args.loaderjob)
    val_iterator = chainer.iterators.MultiprocessIterator(
        val_dataset, config.batchsize, repeat=False, n_processes=args.loaderjob)

    #Models
    gen, dis = load_models(config)
    ignore_params = []
    for l_name in config['models']['binary_discriminator']['extra_layer_names']:
        ignore_params.extend(['{}{}'.format(l_name, k) for k, l in getattr(dis, l_name).namedparams()])
    ignore_params.append('fc2/u')
    ignore_params.append('fc1/u')
    ignore_params.append('fc3/u')
    print(ignore_params)
    chainer.serializers.load_npz(args.gen_ckpt, gen)
    chainer.serializers.load_npz(args.dis_ckpt, dis, ignore_names=ignore_params)
    gen.to_gpu(device=args.gpu)
    dis.to_gpu(device=args.gpu)
    models = {"gen": gen, "dis": dis}

    dis.disable_update()
    dis.fc1.enable_update()
    dis.fc2.enable_update()
    dis.fc3.enable_update()
    
    # Optimizer
    opt_dis = make_optimizer(
        dis, alpha=config.adam['alpha'], beta1=config.adam['beta1'], beta2=config.adam['beta2'])
    opts = {"opt_dis": opt_dis}

    # Updater
    kwargs = config.updater['args'] if 'args' in config.updater else {}
    kwargs.update({
        'models': models,
        'iterator': train_iterator,
        'optimizer': opts,
    })
    updater = FineTuneUpdater(**kwargs)

    dis.disable_update()
    dis.fc1.enable_update()
    dis.fc2.enable_update()
    dis.fc3.enable_update()

    # Trainer
    out = args.results_dir
    create_result_dir(out, args.config_path, config)
    trainer = training.Trainer(updater, (config.iteration, 'iteration'), out=out)
    report_keys = ["loss_dis", "loss_gen", "acc_real", "acc_fake", "train_acc_real", "train_acc_fake", "val_acc_real", "val_acc_fake"]

    # Set up logging
    trainer.extend(extensions.snapshot(), trigger=(config.snapshot_interval, 'iteration'))    
    trainer.extend(extensions.snapshot_object(
        dis, dis.__class__.__name__ + '_{.updater.iteration}.npz'), trigger=(config.snapshot_interval, 'iteration'))
    trainer.extend(extensions.LogReport(keys=report_keys,
                                        trigger=(config.display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(config.display_interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=config.progressbar_interval))
    # ext_opt_dis = extensions.LinearShift('alpha', (config.adam['alpha'], 0.),
    #                                      (config.iteration_decay_start, config.iteration), opt_dis)
    trainer.extend(DiscriminatorEvaluator(iterator={'val': val_iterator},
                                          target={'gen': gen, 'dis': dis},
                                          conditional=updater.conditional))
    #Evaluator
    if args.test:
        evaluator = DiscriminatorEvaluator(iterator={'train': eval_train_iterator, 'val': val_iterator},
                                           target={'gen': gen, 'dis': dis},
                                           conditional=updater.conditional)
        ret = evaluator()
        print(ret)
        return

    if args.snapshot:
        print("Resume training with snapshot:{}".format(args.snapshot))
        chainer.serializers.load_npz(args.snapshot, trainer)

    # Run the training
    print("start training")
    trainer.run()


if __name__ == '__main__':
    main()