import os
import argparse

from torch.backends import cudnn
from utils.utils import *
import random
import torch
import numpy as np

from solver_OT import Solver
from utils.logger import get_logger


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':

        if not os.path.exists(config.results):
            os.makedirs(config.results)
        logger = get_logger(config.results, __name__, str(config.dataset) + '_{}_{}_{}_{}_{}.log'
                             .format(config.k, config.num_proto, config.len_map, config.mask_ratio, config.anomaly_ratio))
        logger.info(config)

        solver.test(logger)

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1) #1
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/MSL')
    parser.add_argument('--dataset', type=str, default='MSL')
    parser.add_argument('--input_c', type=int, default=55)
    parser.add_argument('--output_c', type=int, default=55)
    parser.add_argument('--model_save_path', type=str, default='model_params')
    parser.add_argument('--results', type=str, default='./results')

    parser.add_argument('--mask_ratio', type=float, default=0.05)
    parser.add_argument('--k', type=float, default=2)
    parser.add_argument('--num_proto', type=int, default=8)
    parser.add_argument('--len_map', type=int, default=8)
    parser.add_argument('--anomaly_ratio', type=float, default=0.5)

    parser.add_argument('--win_size', type=int, default=100)#context length
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--e_layers', type=int, default=3)#encoder layers
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)

    config = parser.parse_args()


    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    main(config)
