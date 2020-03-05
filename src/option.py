# -*- coding: utf-8 -*-
import argparse
import os

args = argparse.ArgumentParser(description='The option of FundusTool')

# Dataset option
args.add_argument('--task', type=str, default='optic', help='fundus|vessel|optic')
args.add_argument('--dataset', type=str, default='ORIGA', help='EyePac')
args.add_argument('--data_dir', type=str, default='../data', help='')
args.add_argument('--prepare', action='store_true', help='Use detector or not')
args.add_argument('--seed', type=int, default=0, help='')
args.add_argument('--n_threads', type=int, default=32, help='')
# Processing option
args.add_argument('--n_colors', type=int, default=3, help='')
args.add_argument('--n_classes', type=int, default=5, help='')
args.add_argument('--size', type=int, default=256, help='')
args.add_argument('--stage', type=int, default=0, help='')
args.add_argument('--decay_rate', type=float, default=1.0, help='')

# Training option
args.add_argument('--model', type=str, default='unet', help='Network')
args.add_argument('--resume', action='store_true', help='resume')
args.add_argument('--test', action='store_true', help='Test')
args.add_argument('--regression', action='store_true', help='regerssion mode')
args.add_argument('--gpu', type=str, default='0, 1', help='GPU index')
args.add_argument('--metric', type=str, default='val_kappa', help='Metric')
args.add_argument('--loss', type=str, default='CE', help='Loss')
args.add_argument('--save', type=str, default='../model', help='root path of model')
args.add_argument('--batch_size', type=int, default=4, help='')
args.add_argument('--epochs', type=int, default=200, help='')
args.add_argument('--warm_epochs', type=int, default=5, help='')
args.add_argument('--weight_decay', type=int, default=0.0005, help='')
args.add_argument('--gaps', type=int, default=10, help='')
args.add_argument('--lr', type=float, default=3e-3, help='learning rate')

args = args.parse_args()

def get_template(args):
    # Get the template of the dataset
    args.model_path = os.path.join(args.save, args.task, args.dataset + '_{}'.format(args.size), args.model)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    args.checkpoint = args.model_path + '/' + 'checkpoints/'
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    args.data_dir = os.path.join(args.data_dir, args.task, args.dataset)
    if args.task == 'fundus':
        print('use fundus statistic')
        args.mean = [108.64628601 / 255, 75.86886597 / 255, 54.34005737 / 255]
        args.std = [70.53946096 / 255, 51.71475228 / 255, 43.03428563 / 255]
    else:
        args.mean = [0.5, 0.5, 0.5]
        args.std = [0.5, 0.5, 0.5]
    return args

args = get_template(args)

