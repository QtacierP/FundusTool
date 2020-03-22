from data.common import AbstractDataLoader, DRIVEDataset
import os
from torchvision import datasets, transforms
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset, random_split
import torch
from option import args
from PIL.Image import NEAREST

# This is the vessel segmentation part
# DRIVE/STARE Dataset

class MyDataLoader(AbstractDataLoader):
    def __init__(self, args):
        super(MyDataLoader, self).__init__(args)

    def prepare(self):
        pass

    def load(self):

        train_preprocess = transforms.Compose([
            transforms.Resize((self.args.size, self.args.size), interpolation=NEAREST),
            transforms.RandomCrop((self.args.crop_size, self.args.crop_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.args.mean,
                                 self.args.std),
        ])
        train_gt_preprocess = transforms.Compose([
            transforms.Resize((self.args.size, self.args.size), interpolation=NEAREST),
            transforms.RandomCrop((self.args.crop_size, self.args.crop_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

        test_preprocess = transforms.Compose([
            transforms.Resize((self.args.size, self.args.size), interpolation=NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(self.args.mean,
                                 self.args.std),

        ])

        test_gt_preprocess = transforms.Compose([
            transforms.Resize((self.args.size, self.args.size), interpolation=NEAREST),
            transforms.ToTensor(),
        ])

        if self.args.dataset == 'DRIVE':
            train_dataset = DRIVEDataset(args, os.path.join(self.args.data_dir, 'train'),
                                    train_preprocess, train_gt_preprocess, step=self.args.step)
            val_dataset = DRIVEDataset(args, os.path.join(self.args.data_dir, 'validate'),
                                         test_preprocess, test_gt_preprocess)
            test_dataset = DRIVEDataset(args, os.path.join(self.args.data_dir, 'test'),
                                         test_preprocess, test_gt_preprocess)

        else:
            raise NotImplementedError('{} dataset not implemented yet'
                                      .format(self.args.dataset))
        print('[Train]: ', train_dataset.__len__() * self.args.step)
        print('[Val]: ', val_dataset.__len__())
        print('[Test]: ', test_dataset.__len__())
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size,
                                      shuffle=True, num_workers=self.args.n_threads)
        val_dataloader = DataLoader(val_dataset, batch_size=self.args.test_batch_size,
                                    shuffle=False, num_workers=self.args.n_threads)
        test_dataloader = DataLoader(test_dataset, batch_size=self.args.test_batch_size,
                                     shuffle=False, num_workers=self.args.n_threads)

        return train_dataloader, val_dataloader, test_dataloader


