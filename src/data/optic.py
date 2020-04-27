from data.common import AbstractDataLoader, ORIGIADataset, crop_OD, \
    REFUGEDataset
import os
from torchvision import datasets, transforms
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, \
    ConcatDataset, random_split
import torch
from option import args
from PIL.Image import NEAREST
from model.backbone import UNet
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
import random
from PIL import Image


# This is the optic cup & disc segmentation part.
# Stage 0: Coarse segmentation: For OD
# Stage 1: Fine segmentation: Crop OD firstly, then segment OC & OD

class MyDataLoader(AbstractDataLoader):
    def __init__(self, args):
        super(MyDataLoader, self).__init__(args)

    def load(self):
        train_preprocess = transforms.Compose([
            transforms.Resize((self.args.size, self.args.size), interpolation=NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.args.mean,
                                 self.args.std),
        ])

        train_gt_preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.args.size, self.args.size), interpolation=NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        test_preprocess = transforms.Compose([
            transforms.Resize((self.args.size, self.args.size), interpolation=NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(self.args.mean,
                                 self.args.std),
        ])

        test_gt_preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.args.size, self.args.size), interpolation=NEAREST),
        ])
        if self.args.dataset == 'ORIGA' or self.args.dataset == 'enhanced_ORIGA':
            if self.args.test:
                train_preprocess = test_preprocess
                train_gt_preprocess = test_gt_preprocess
            dataset = ORIGIADataset(args, self.args.data_dir,
                                    train_preprocess, train_gt_preprocess, self.args.stage)
            N = dataset.__len__()
            random.seed(0)
            torch.manual_seed(0)
            train_dataset, val_dataset, test_dataset = \
                random_split(dataset, [int(0.7 * N), int(0.1 * N), N - int(0.7 * N) - int(0.1 * N)])

        elif self.args.dataset == 'REFUGE' or self.args.dataset == 'enhanced_REFUGE':

            train_dataset = REFUGEDataset(self.args, self.args.data_dir,
                                          train_preprocess, train_gt_preprocess,
                                          stage=self.args.stage, task='train')

            val_dataset = REFUGEDataset(self.args, self.args.data_dir,
                                          test_preprocess, test_gt_preprocess,
                                          stage=self.args.stage, task='val')

            test_dataset = REFUGEDataset(self.args, self.args.data_dir,
                                          test_preprocess, test_gt_preprocess,
                                          stage=self.args.stage, task='test')
        else:
            raise NotImplementedError('{} dataset not implemented yet'
                                      .format(self.args.dataset))

        print('[Train]: ', train_dataset.__len__())
        print('[Val]: ', val_dataset.__len__())
        print('[Test]: ', test_dataset.__len__())
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size,
                                      shuffle=True, num_workers=self.args.n_threads)
        val_dataloader = DataLoader(val_dataset, batch_size=self.args.batch_size,
                                    shuffle=False, num_workers=self.args.n_threads)
        test_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size,
                                     shuffle=False, num_workers=self.args.n_threads)
        return train_dataloader, val_dataloader, test_dataloader


    def prepare(self):
        if self.args.dataset == 'ORIGA'  or self.args.dataset == 'enhanced_ORIGA':
            self.prepare_origa()
        elif self.args.dataset == 'REFUGE' or self.args.dataset == 'enhanced_REFUGE':
            self.prepare_refuge()

    def  prepare_refuge(self):
        if self.args.stage == 0:
           return
        out_img_path = os.path.join(self.args.data_dir, 'images_1')
        out_gt_path = os.path.join(self.args.data_dir, 'gts_1')
        tasks = ['train', 'val', 'test']
        for task in tasks:
            task_out_img_path = os.path.join(out_img_path, task)
            task_out_gt_path = os.path.join(out_gt_path, task)
            if not os.path.exists(task_out_img_path):
                os.makedirs(task_out_img_path)
            if not os.path.exists(task_out_gt_path):
                os.makedirs(task_out_gt_path)
            model = UNet(num_classes=2, n_colors=3).cuda()
            model.load_state_dict(torch.load('../model/optic/REFUGE_512_0/unet'
                                             '/unet_last.pt'))
            torch.set_grad_enabled(False)
            model.eval()
            if self.args.n_gpus:
                model = torch.nn.DataParallel(model)
            train_preprocess = transforms.Compose([
                transforms.Resize((512, 512), interpolation=NEAREST),
                transforms.ToTensor(),
                transforms.Normalize(self.args.mean,
                                     self.args.std),
            ])
            train_gt_preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512), interpolation=NEAREST),
                transforms.ToTensor(),
            ])
            dataset = REFUGEDataset(args, self.args.data_dir,
                                    train_preprocess,
                                    train_gt_preprocess, task=task, stage=0, need_name=True)
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                    shuffle=False, num_workers=self.args.n_threads)
            for data in tqdm(dataloader):
                imgs, gts, names, ori_imgs, ori_gts = data
                imgs = imgs.cuda()
                ori_imgs = ori_imgs.cpu().numpy()
                preds = model(imgs)
                preds = torch.softmax(preds, dim=1)
                preds = preds.detach().permute(0, 2, 3, 1).cpu().numpy()
                ori_gts = ori_gts.detach().cpu().numpy()
                for i in range(preds.shape[0]):
                    pred = preds[i, ...]
                    ori_gt = ori_gts[i, ...]
                    ori_img = ori_imgs[i, ...]
                    disc_map = pred[..., 1]
                    mini_img, mini_gt = crop_OD(disc_map, ori_img, ori_gt)
                    name = names[i].split('/')[-1]
                    file = os.path.join(task_out_img_path, name)
                    mini_img = mini_img.astype(np.uint8)
                    mini_gt = mini_gt.astype(np.uint8)
                    plt.imsave(file, mini_img)
                    name = name.split('.')[0] + '.bmp'
                    mini_gt = np.squeeze(mini_gt, axis=-1).astype('uint8')
                    mini_gt = np.squeeze(mini_gt, axis=-1)
                    file = os.path.join(task_out_gt_path, name)
                    mini_gt = Image.fromarray(mini_gt)
                    mini_gt.save(file)
        torch.set_grad_enabled(True)


    def prepare_origa(self):
        if self.args.stage == 0:
            return
        data_path = os.path.join(self.args.data_dir, 'images_0')
        out_img_path = os.path.join(self.args.data_dir, 'images_1')
        out_gt_path = os.path.join(self.args.data_dir, 'gts_1')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        model = UNet(num_classes=2, n_colors=3).cuda()
        model.load_state_dict(torch.load('../model/optic/ORIGA_512_0/unet'
                                         '/unet_best.pt'))
        torch.set_grad_enabled(False)
        model.eval()
        if self.args.n_gpus:
            model = torch.nn.DataParallel(model)
        train_preprocess = transforms.Compose([
            transforms.Resize((512, 512), interpolation=NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(self.args.mean,
                                 self.args.std),
        ])
        train_gt_preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512), interpolation=NEAREST),
            transforms.ToTensor(),
        ])
        dataset = ORIGIADataset(args, self.args.data_dir,
                                    train_preprocess,
                                 train_gt_preprocess, 0, need_name=True)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                      shuffle=False, num_workers=self.args.n_threads)
        for data in tqdm(dataloader):
            imgs, gts, names, ori_imgs, ori_gts = data
            imgs = imgs.cuda()
            ori_imgs = ori_imgs.cpu().numpy()
            preds = model(imgs)
            preds = preds.detach().permute(0, 2, 3, 1).cpu().numpy()
            ori_gts = ori_gts.detach().cpu().numpy()
            for i in range(preds.shape[0]):
                pred = preds[i, ...]
                ori_gt = ori_gts[i, ...]
                ori_img = ori_imgs[i, ...]
                mini_img, mini_gt = crop_OD(pred, ori_img, ori_gt)
                name = names[i].split('/')[-1]
                file = os.path.join(out_img_path, name)
                mini_img = mini_img.astype(np.uint8)
                mini_gt = mini_gt.astype(np.uint8)
                plt.imsave(file, mini_img)
                name = name.split('.')[0] + '.mat'
                mini_gt = np.squeeze(mini_gt, axis=-1)
                file = os.path.join(out_gt_path, name)
                savemat(file, {'mask': mini_gt})
        torch.set_grad_enabled(True)





