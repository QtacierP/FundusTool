import tensorflow as tf
from option import args, get_template
from model import get_model
import os
from data import get_dataloder
import numpy as np
from utils import get_statics
from copy import deepcopy


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


def compare():
    print('[GPU INDEX] : ', args.gpu)
    args.n_gpus = len(args.gpu.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.test = True
    args.task = 'optic' # Only support optic now
    train_dataloader, val_dataloader, test_dataloader = get_dataloder(args).load()
    unet = get_model(args)
    e_args = deepcopy(args)
    e_args.dataset = 'enhanced_{}'.format(args.dataset)
    e_args.data_dir = '../data'
    e_args = get_template(e_args)
    e_train_dataloader, e_val_dataloader, e_test_dataloader = get_dataloder(e_args).load()
    e_unet = get_model(e_args)
    print(e_unet.args.dataset)
    preds, gts, imgs = unet.test(test_dataloader, eval=True)
    preds = np.asarray(preds)
    gts = np.asarray(gts)
    e_preds, e_gts, e_imgs = e_unet.test(e_test_dataloader, eval=True)
    e_preds = np.asarray(e_preds)
    e_gts = np.asarray(e_gts)
    dices = get_statics(args, preds, gts, imgs)
    e_dices = get_statics(e_args, e_preds, e_gts, e_imgs)



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
        f.write(str(result))
        f.close()

    else:
        result = model.test(args.test_dir, args.out_dir)
        print(result)



if __name__ == '__main__':
    if not args.compare:
        main()
    else:
        compare()