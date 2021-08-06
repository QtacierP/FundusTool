
from option import args, get_template
from model import get_model
import os
from data import get_dataloder
import numpy as np
from utils import get_statics, get_compare, get_p_value, divide_score
from copy import deepcopy





def compare(divide=False):
    print('[GPU INDEX] : ', args.gpu)
    args.n_gpus = len(args.gpu.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.test = True
    args.task = 'optic' # Only support optic now
    train_dataloader, val_dataloader, test_dataloader = get_dataloder(args).load()
    e_args = deepcopy(args)
    e_args.dataset = 'enhanced_{}'.format(args.dataset)
    e_args.data_dir = '../data'
    e_args = get_template(e_args)
    e_train_dataloader, e_val_dataloader, e_test_dataloader = get_dataloder(e_args).load()
    e_unet = get_model(e_args)
    unet = get_model(args)
    preds, gts, imgs = unet.test(test_dataloader, eval=True)
    preds = np.asarray(preds)
    gts = np.asarray(gts)
    e_preds, e_gts, e_imgs = e_unet.test(e_test_dataloader, eval=True)
    e_preds = np.asarray(e_preds)
    e_gts = np.asarray(e_gts)
    dices = get_statics(args, preds, gts, imgs)
    e_dices = get_statics(e_args, e_preds, e_gts, e_imgs)
    if divide:
        img_list, e_img_list, dice_list, e_dice_list, \
        pred_list, e_pred_list, gt_list = divide_score(args, imgs,
        e_imgs, dices, e_dices, preds, e_preds, gts)
        grades = ['good', 'mid', 'bad']
        for i in range(3):
            imgs = img_list[i]
            print(len(imgs))
            e_imgs = e_img_list[i]
            dices = dice_list[i]
            e_dices = e_dice_list[i]
            preds = pred_list[i]
            e_preds = e_pred_list[i]
            gts = gt_list[i]
            e_gts = gts
            diffs = get_compare(args, imgs, e_imgs, preds, e_preds,
                                gts, dices, e_dices, grade=grades[i])
            z = get_p_value(dices, e_dices)
            print(z)
    else:
        diffs = get_compare(args, imgs, e_imgs, preds, e_preds, gts, dices, e_dices)
        z = get_p_value(dices, e_dices)
        print(z)

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
        result, preds, gts = model.test(test_dataloader)
        print(result)
        f = open(os.path.join(args.model_path, 'result.txt'), 'w')
        f.write(str(result))
        f.close()
        
    else:
        result, preds, gts = model.test(args.test_dir, args.out_dir)
        print(result)




if __name__ == '__main__':
    if not args.compare:
        main()
    else:
        compare()