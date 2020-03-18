from data.common import AbstractDataLoader, ORIGIADataset
import os
from torchvision import datasets, transforms
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset, random_split
import torch
from option import args
from PIL.Image import NEAREST

# This is the optic cup & disc segmentation part.
# Stage 0: Coarse segmentation: For OD
# Stage 1: Fine segmentation: Crop OD firstly, then segment OC & OD

class MyDataLoader(AbstractDataLoader):
    def __init__(self, args):
        super(MyDataLoader, self).__init__(args)

    def prepare(self):
        pass

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

        if self.args.dataset == 'ORIGA':
            dataset = ORIGIADataset(args, self.args.data_dir,
                                    train_preprocess, train_gt_preprocess, self.args.stage)
        else:
            raise NotImplementedError('{} dataset not implemented yet'
                                      .format(self.args.dataset))
        N = dataset.__len__()
        train_dataset, val_dataset, test_dataset = \
            random_split(dataset, [int(0.7 * N), int(0.1 * N), N - int(0.7 * N) - int(0.1 * N)])
        print('[Train]: ', train_dataset.__len__())
        print('[Val]: ', val_dataset.__len__())
        print('[Test]: ', test_dataset.__len__())
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size,
                                      shuffle=True, num_workers=self.args.n_threads)
        val_dataloader = DataLoader(val_dataset, batch_size=self.args.batch_size,
                                    shuffle=False, num_workers=self.args.n_threads)
        test_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size,
                                     shuffle=False, num_workers=self.args.n_threads)
        '''import numpy as np
        from utils import UnNormalize
        from matplotlib.pyplot import show, imshow
        for batch in train_dataloader:
            imgs, labels = batch[0], \
                           batch[1]
            for i in range(imgs.shape[0]):
                img = imgs[i, ...]
                label = labels[i, ...]
                img = UnNormalize(self.args.mean,
                                  self.args.std)(img).permute(1, 2, 0).cpu().numpy()

                label = label.cpu().numpy()
                label = (label * 255).astype(np.uint8)
                img = (img * 255).astype(np.uint8)
                label = np.squeeze(label, axis=-1)
                print(np.unique(label))
                imshow(img)
                show()
                print(np.sum(label) / 255 / (label.shape[0] * label.shape[1]))
                imshow(label)
                show()'''
        return train_dataloader, val_dataloader, test_dataloader


