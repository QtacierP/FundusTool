# -*- coding: utf-8 -*-
import argparse
import os

args = argparse.ArgumentParser(description='The option of FundusTool')

# Dataset option
args.add_argument('--task', type=str, default='fundus', help='fundus|vessel|optic')
args.add_argument('--dataset', type=str, default='', help='EyePac')
args.add_argument('--data_dir', type=str, default='../data', help='')
args.add_argument('--prepare', action='store_true', help='Use detector or not')
args.add_argument('--seed', type=int, default=0, help='')

# Processing option
args.add_argument('--n_colors', type=int, default=3, help='')
args.add_argument('--n_classes', type=int, default=5, help='')
args.add_argument('--size', type=int, default=256, help='')

# Training option
args.add_argument('--model', type=str, default='InceptionV3', help='Network')
args.add_argument('--resume', action='store_true', help='resume')
args.add_argument('--test', action='store_true', help='Test')
args.add_argument('--gpu', type=str, default='0, 1', help='GPU index')
args.add_argument('--metric', type=str, default='val_cohen_kappa', help='Metric')
args.add_argument('--save', type=str, default='../model', help='root path of model')
args.add_argument('--batch_size', type=int, default=4, help='')
args.add_argument('--epochs', type=int, default=200, help='')
args.add_argument('--gaps', type=int, default=10, help='')
args.add_argument('--lr', type=float, default=1e-5, help='learning rate')

args = args.parse_args()

def get_template(args):
    # Get the template of the dataset


    args.model_path = os.path.join(args.save, args.task, args.dataset + '_{}'.format(args.size), args.model)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    args.checkpoint = args.model_path + '/' + 'checkpoints/'
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    return args

args = get_template(args)

