
from option import args
from model import get_model
import os



def main():
    print('[GPU INDEX] : ', args.gpu)
    args.n_gpus = len(args.gpu.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    model = get_model(args)
    if not args.test:
        model.train(os.path.join(args.data_dir, args.task))
    model.test(os.path.join(args.data_dir, args.task))  # We only care the Bad2Good

if __name__ == '__main__':
    main()