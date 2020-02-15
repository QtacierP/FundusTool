
from option import args
from model import get_model
import os
from data import get_dataloder



def main():
    print('[GPU INDEX] : ', args.gpu)
    args.n_gpus = len(args.gpu.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    data_loader = get_dataloder(args)
    train_data, val_data, test_data = data_loader.load()
    model = get_model(args)
    if not args.test:
        model.train(train_data, val_data)
    model.test(os.path.join(args.data_dir, args.task))  # We only care the Bad2Good

if __name__ == '__main__':
    main()