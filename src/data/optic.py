from data.common import AbstractDataLoader, ORIGIADataset
import os
from torchvision import datasets, transforms
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset, random_split
import torch
from option import args

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
            transforms.Resize((self.args.size, self.args.size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.args.mean,
                                 self.args.std),
        ])
        train_gt_preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.args.size, self.args.size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

        if self.args.dataset == 'ORIGA':
            dataset = ORIGIADataset(self.args.data_dir,
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


        return train_dataloader, val_dataloader, test_dataloader