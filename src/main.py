import tensorflow as tf
from option import args
from model import get_model
import os
from data import get_dataloder
import numpy as np

# Activate eager mode to use multiple GPU
tf.compat.v1.disable_eager_execution()





def main():
    print('[GPU INDEX] : ', args.gpu)
    args.n_gpus = len(args.gpu.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    data_loader = get_dataloder(args)
    train_dataloader, val_dataloader, test_dataloader = data_loader.load()
    model = get_model(args)
    if not args.test:
        model.train(train_dataloader, val_dataloader)
    model.test(test_dataloader)

if __name__ == '__main__':
    main()