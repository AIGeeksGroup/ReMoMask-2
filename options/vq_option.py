import argparse
import os
import torch

def arg_parse(is_train=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_name', type=str, default='humanml3d', choices=['humanml3d', 'kit'])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--max_epoch', default=50, type=int)
    parser.add_argument('--warm_up_iter', default=2000, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--milestones', default=[150000, 250000], nargs='+', type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--commit', type=float, default=0.02)
    parser.add_argument('--loss_vel', type=float, default=0.5)
    parser.add_argument('--recons_loss', type=str, default='l1_smooth')
    parser.add_argument('--code_dim1d', type=int, default=512)
    parser.add_argument('--nb_code1d', type=int, default=512)
    parser.add_argument('--code_dim2d', type=int, default=512)
    parser.add_argument('--nb_code2d', type=int, default=512)
    parser.add_argument('--mu', type=float, default=0.99)
    parser.add_argument('--down_t', type=int, default=2)
    parser.add_argument('--stride_t', type=int, default=2)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--dilation_growth_rate', type=int, default=3)
    parser.add_argument('--vq_act', type=str, default='relu', choices=['relu', 'silu', 'gelu'])
    parser.add_argument('--vq_norm', type=str, default=None, choices=['GN'])
    parser.add_argument('--num_quantizers', type=int, default=3)
    parser.add_argument('--shared_codebook', action='store_true')
    parser.add_argument('--quantize_dropout_prob', type=float, default=0.2)
    parser.add_argument('--ext', type=str, default='default')
    parser.add_argument('--name', type=str, default='pretrain_vq')
    parser.add_argument('--is_continue', action='store_true')
    parser.add_argument('--checkpoints_dir', type=str, default='./logs')
    parser.add_argument('--log_every', default=10, type=int)
    parser.add_argument('--save_latest', default=500, type=int)
    parser.add_argument('--eval_every_e', default=1, type=int)
    parser.add_argument('--feat_bias', type=float, default=5)
    parser.add_argument('--which_epoch', type=str, default='net_best_fid.tar')
    parser.add_argument('--seed', default=3407, type=int)
    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu_id)
    args = vars(opt)
    print('------------ Options -------------')
    for (k, v) in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    opt.is_train = is_train
    if is_train:
        expr_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for (k, v) in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
    return opt
