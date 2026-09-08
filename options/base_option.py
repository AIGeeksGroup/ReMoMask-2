import argparse
import os
import torch

class BaseOptions:

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default='trans')
        self.parser.add_argument('--vq_name', type=str, default='rvq')
        self.parser.add_argument('--gpu_id', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
        self.parser.add_argument('--dataset_name', type=str, default='humanml3d', choices=['humanml3d', 'kit'])
        self.parser.add_argument('--checkpoints_dir', type=str, default='./logs')
        self.parser.add_argument('--latent_dim', type=int, default=512)
        self.parser.add_argument('--n_heads', type=int, default=8)
        self.parser.add_argument('--n_layers', type=int, default=8)
        self.parser.add_argument('--ff_size', type=int, default=1024)
        self.parser.add_argument('--dropout', type=float, default=0.2)
        self.parser.add_argument('--attnj', action='store_true')
        self.parser.add_argument('--attnt', action='store_true')
        self.parser.add_argument('--max_motion_length', type=int, default=196)
        self.parser.add_argument('--unit_length', type=int, default=4)
        self.parser.add_argument('--force_mask', action='store_true')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.is_train = self.is_train
        if self.opt.gpu_id != -1:
            torch.cuda.set_device(self.opt.gpu_id)
        args = vars(self.opt)
        is_main = int(os.environ.get('RANK', '0')) == 0
        if is_main:
            print('------------ Options -------------')
            for (k, v) in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')
        if self.is_train and is_main:
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.name)
            os.makedirs(expr_dir, exist_ok=True)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for (k, v) in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
