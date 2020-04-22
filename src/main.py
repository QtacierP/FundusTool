import tensorflow as tf
from option import args
from model import get_model
import os
from data import get_dataloder
import numpy as np




def test_f():
    from model.backbone import resnet50_backbone
    import torch
    base_model = resnet50_backbone(3, size=args.size)
    f = torch.randn(1, args.n_colors, args.size, args.size).cuda()
    model = list(base_model.children())[:-3]  # From head to the last bottleneck
    tail = list(list(base_model.children())[-3].children())[:-2]
    model += tail
    model = torch.nn.Sequential(*model).cuda()
    print(model(f).shape)



def main():
    print('[GPU INDEX] : ', args.gpu)
    args.n_gpus = len(args.gpu.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    data_loader = get_dataloder(args)
    train_dataloader, val_dataloader, test_dataloader = data_loader.load()
    model = get_model(args)
    result = None
    if not args.test:
        model.train(train_dataloader, val_dataloader)
    if args.test_dir == '':
        result = model.test(test_dataloader)
        print(result)
        f = open(os.path.join(args.model_path, 'result.txt'), 'w')
        f.write(result)
        f.close()

    else:
        result = model.test(args.test_dir, args.out_dir)
        print(result)



if __name__ == '__main__':
    main()