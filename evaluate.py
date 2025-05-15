# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from time import time
import torchvision


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.amp import autocast

from model import AutoEncoder
import utils
import datasets
from train import test, init_processes, test_vae_fid

import networkx as nx
from itertools import combinations

from evaulate_concepts import get_dag, sample_constant_noise

def set_bn(model, bn_eval_mode, num_samples=1, t=1.0, iter=100):
    if bn_eval_mode:
        model.eval()
    else:
        model.train()
        with autocast("cuda"):
            for i in range(iter):
                if i % 10 == 0:
                    print('setting BN statistics iter %d out of %d' % (i+1, iter))
                model.module.sample(num_samples, t)
        model.eval()


def main(rank, eval_args):

    # ensures that weight initializations are all the same
    print(f"Setting up process for rank {rank}")
    setup(rank, eval_args.world_size)

    logging = utils.Logger(eval_args.local_rank, eval_args.save)

    # load a checkpoint
    logging.info('loading the model at:')
    logging.info(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location='cuda', weights_only=False)
    args = checkpoint['args']

    args.num_process_per_node = eval_args.world_size
    args.world_size = eval_args.world_size
    args.distributed = eval_args.world_size > 1

    if not hasattr(args, 'ada_groups'):
        logging.info('old model, no ada groups was found.')
        args.ada_groups = False

    if not hasattr(args, 'min_groups_per_scale'):
        logging.info('old model, no min_groups_per_scale was found.')
        args.min_groups_per_scale = 1

    if not hasattr(args, 'num_mixture_dec'):
        logging.info('old model, no num_mixture_dec was found.')
        args.num_mixture_dec = 10

    if eval_args.batch_size > 0:
        args.batch_size = eval_args.batch_size

    logging.info('loaded the model at epoch %d', checkpoint['epoch'])
    arch_instance = utils.get_arch_cells(args.arch_instance)
    uncomp_model = AutoEncoder(args, None, arch_instance)
    uncomp_model = uncomp_model.to(rank)
    ddp_model = DDP(uncomp_model, device_ids=[rank], output_device=rank)
    model = ddp_model #torch.compile(ddp_model)
    # Loading is not strict because of self.weight_normalized in Conv2D class in neural_operations. This variable
    # is only used for computing the spectral normalization and it is safe not to load it. Some of our earlier models
    # did not have this variable.
    model.load_state_dict(checkpoint['state_dict'])

    logging.info('args = %s', args)
    logging.info('num conv layers: %d', len(model.module.all_conv_layers))
    logging.info('param size = %fM ', utils.count_parameters_in_M(model.module))

    if eval_args.eval_mode == 'evaluate':
        # load train valid queue
        args.data = eval_args.data
        train_queue, valid_queue, num_classes = datasets.get_loaders(args)

        if eval_args.eval_on_train:
            logging.info('Using the training data for eval.')
            valid_queue = train_queue

        # get number of bits
        num_output = utils.num_output(args.dataset)
        bpd_coeff = 1. / np.log(2.) / num_output

        valid_neg_log_p, valid_nelbo = test(valid_queue, model, num_samples=eval_args.num_iw_samples, args=args, logging=logging)
        logging.info('final valid nelbo %f', valid_nelbo)
        logging.info('final valid neg log p %f', valid_neg_log_p)
        logging.info('final valid nelbo in bpd %f', valid_nelbo * bpd_coeff)
        logging.info('final valid neg log p in bpd %f', valid_neg_log_p * bpd_coeff)
    elif eval_args.eval_mode == 'evaluate_fid':
        bn_eval_mode = not eval_args.readjust_bn
        set_bn(model, bn_eval_mode, num_samples=2, t=eval_args.temp, iter=500)
        args.fid_dir = eval_args.fid_dir
        args.num_process_per_node, args.num_proc_node = eval_args.world_size, 1   # evaluate only one 1 node
        fid = test_vae_fid(model.module, args, total_fid_samples=50000)
        logging.info('fid is %f' % fid)
    elif eval_args.eval_mode == 'dag':
        model = model.eval()
        with torch.no_grad():
            get_dag(logging, model, args, eval_args)
    elif eval_args.eval_mode == 'sample_constant_noise':
        model = model.eval()
        with torch.no_grad():

            bn_eval_mode = not eval_args.readjust_bn
            set_bn(model, bn_eval_mode, num_samples=16, t=eval_args.temp, iter=500)

            num_iter = 100
            for ind in range(num_iter):     # sampling is repeated.
                image_name = 'constant_noise_gpu_%d_samples_%d' % (eval_args.local_rank, ind)
                sample_constant_noise(logging, model, args, eval_args, image_name=image_name, temp=eval_args.temp)
    elif eval_args.eval_mode.startswith('reconstruction'):
        num_samples = 4

        args.global_rank = rank
        args.local_rank = rank
        args.world_size = eval_args.world_size
        args.distributed = eval_args.world_size > 1
        args.batch_size = num_samples

        train_queue, valid_queue, num_classes = datasets.get_loaders(args)
        if eval_args.eval_mode == 'reconstruction_train':
            logging.info('Using the training data for eval.')
            queue = train_queue
        else:
            logging.info('Using the validation data for eval.')
            queue = valid_queue

        model = model.eval()

        with torch.no_grad():

            bn_eval_mode = not eval_args.readjust_bn
            set_bn(model, bn_eval_mode, num_samples=16, t=eval_args.temp, iter=500)
            
            total_samples = 500 // eval_args.world_size          # num images per gpu
            num_iter = int(np.ceil(total_samples / num_samples))   # num iterations per gpu

            for ind in range(num_iter):     # sampling is repeated.
                torch.cuda.synchronize()
                start = time()
                with autocast("cuda"):
                    x = next(iter(queue))
                    if not isinstance(x, torch.Tensor):
                        label = x[1]
                        x = x[0]
                    else:
                        label = None
                    x = x.to(rank)

                    logits, log_q, log_p, kl_all, kl_diag = model(x, batch_label=label)

                    if label is not None:
                        logging.info('label: %s', label)
                    output = model.module.decoder_output(logits)

                    x_img = x[:num_samples]
                    output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.sample()
                    output_img = output_img[:num_samples]

                    x_img = x_img.permute(0, 2, 3, 1)
                    output_img = output_img.permute(0, 2, 3, 1)
                    
                    x_img = x_img.cpu().numpy()
                    output_img = output_img.cpu().numpy()

                    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples, 2))
                    for i in range(num_samples):
                        axes[0, i].imshow(x_img[i])
                        axes[0, i].axis('off')
                        axes[1, i].imshow(output_img[i])
                        axes[1, i].axis('off')
                    plt.tight_layout()
                    plt.subplots_adjust(wspace=0, hspace=0)
                    plt.margins(0, 0)
                    plt.savefig(os.path.join(eval_args.save, 'gpu_%d_samples_%d.png' % (eval_args.local_rank, ind)))
                    plt.close(fig)

                    logging.info('Saved at: %s', os.path.join(eval_args.save, 'gpu_%d_samples_%d.png' % (eval_args.local_rank, ind)))

    else:
        bn_eval_mode = not eval_args.readjust_bn
        total_samples = 5000 // eval_args.world_size          # num images per gpu
        num_samples = 16                                      # sampling batch size
        num_iter = int(np.ceil(total_samples / num_samples))   # num iterations per gpu

        if args.arch_flag == 'concepts':
            if eval_args.eval_mode == 'sample_combo':
                concepts = (c for c in args.concepts if c != "obs")
                combos = list(combo for combo in combinations(concepts, 2))
                logging.info('combos: %s', combos)
            else:
                combos = [[c] for c in args.concepts]

        with torch.no_grad():
            n = int(np.floor(np.sqrt(num_samples)))
            set_bn(model, bn_eval_mode, num_samples=16, t=eval_args.temp, iter=500)
            for ind in range(num_iter):     # sampling is repeated.
                torch.cuda.synchronize()
                start = time()
                with autocast("cuda"):
                    if args.arch_flag == 'concepts':
                        combo = combos[ind % len(combos)]
                        logging.info('combo: %s', combo)
                        logits = model.module.sample(num_samples, eval_args.temp, batch_label=combo)
                    else:
                        logits = model.module.sample(num_samples, eval_args.temp)
                output = model.module.decoder_output(logits)
                output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) \
                    else output.sample()
                torch.cuda.synchronize()
                end = time()
                logging.info('sampling time per batch: %0.3f sec', (end - start))

                visualize = False
                if visualize:
                    output_tiled = utils.tile_image(output_img, n).cpu().numpy().transpose(1, 2, 0)
                    output_tiled = np.asarray(output_tiled * 255, dtype=np.uint8)
                    output_tiled = np.squeeze(output_tiled)

                    plt.imshow(output_tiled)
                    plt.show()
                else:
                    file_path = os.path.join(eval_args.save, 'gpu_%d_samples_%d.npz' % (eval_args.local_rank, ind))
                    np.savez_compressed(file_path, samples=output_img.cpu().numpy())

                    nrows = int(np.ceil(np.sqrt(num_samples)))
                    grid = torchvision.utils.make_grid(output_img.cpu(), nrow=nrows, normalize=True)
                    torchvision.utils.save_image(grid, file_path.replace('.npz', '.png'))

                    
                    logging.info('Saved at: {}'.format(file_path))
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('encoder decoder examiner')
    # experimental results
    parser.add_argument('--checkpoint', type=str, default='/tmp/expr/checkpoint.pt',
                        help='location of the checkpoint')
    parser.add_argument('--save', type=str, default='/tmp/expr',
                        help='location of the checkpoint')
    parser.add_argument('--eval_mode', type=str, default='sample', \
                        choices=['sample', 'sample_combo', 'evaluate', 'evaluate_fid', 'dag', 'sample_constant_noise',
                                 'reconstruction_train', 'reconstruction_test'],
                        help='evaluation mode. you can choose between sample or evaluate.')
    parser.add_argument('--eval_on_train', action='store_true', default=False,
                        help='Settings this to true will evaluate the model on training data.')
    parser.add_argument('--data', type=str, default='/tmp/data',
                        help='location of the data corpus')
    parser.add_argument('--readjust_bn', action='store_true', default=False,
                        help='adding this flag will enable readjusting BN statistics.')
    parser.add_argument('--temp', type=float, default=0.7,
                        help='The temperature used for sampling.')
    parser.add_argument('--num_iw_samples', type=int, default=1000,
                        help='The number of IW samples used in test_ll mode.')
    parser.add_argument('--fid_dir', type=str, default='/tmp/fid-stats',
                        help='path to directory where fid related files are stored')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Batch size used during evaluation. If set to zero, training batch size is used.')
    # DDP.
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process')
    parser.add_argument('--world_size', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')

    args = parser.parse_args()
    utils.create_exp_dir(args.save)

    size = args.world_size

    mp.spawn(
        main,
        args=(args,),
        nprocs=size,
        join=True
    )

def setup(rank, world_size):
    # initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()
