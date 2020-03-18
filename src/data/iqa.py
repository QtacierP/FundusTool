from data.common import AbstractDataLoader, \
    ScheduledWeightedSampler, PeculiarSampler, \
    make_weights_for_balanced_classes, KrizhevskyColorAugmentation, IQADataset
import os
from torchvision import datasets, transforms
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
import torch
from option import args

class MyDataLoader(AbstractDataLoader):
    def __init__(self, args):
        super(MyDataLoader, self).__init__(args)

    def prepare(self):
            if self.args.dataset == 'EyeQ':
                quality_list = [0, 1, 2]
            else:
                quality_list = [0, 2]
            train_csv = os.path.join(self.args.data_dir, 'Label_EyeQ_train.csv')
            test_csv = os.path.join(self.args.data_dir, 'Label_EyeQ_test.csv')
            train_table = pd.read_csv(train_csv)
            root_path = '../data/fundus/EyePac'
            for idx in tqdm(range(train_table.shape[0])):
                quality = train_table['quality'][idx]
                if quality not in quality_list:
                    continue
                img_name = train_table['image'][idx]
                flag = True
                for task in ['train', 'val']:
                    for c in range(5):
                        c = str(c)
                        img_path = os.path.join(root_path, task, c, img_name)
                        if os.path.exists(img_path):
                            out_dir = os.path.join(self.args.data_dir, task, str(quality))
                            if not os.path.exists(out_dir):
                                os.makedirs(out_dir)
                            os.system('cp {} {}'.format(img_path, out_dir))
                            flag = False
                            continue
                if flag:
                    print('{} Not found at all !'.format(img_name))

            test_table = pd.read_csv(test_csv)
            root_path = '../data/fundus/EyePac'
            for idx in tqdm(range(test_table.shape[0])):
                quality = test_table['quality'][idx]
                if quality not in quality_list:
                    continue
                img_name = test_table['image'][idx]
                for task in ['test']:
                    for c in range(5):
                        c = str(c)
                        img_path = os.path.join(root_path, task, c, img_name)
                        if os.path.exists(img_path):
                            out_dir = os.path.join(self.args.data_dir, task, str(quality))
                            if not os.path.exists(out_dir):
                                os.makedirs(out_dir)
                            os.system('cp {} {}'.format(img_path, out_dir))


    def load(self):
        train_path = os.path.join(self.args.data_dir, 'train')
        test_path = os.path.join(self.args.data_dir, 'test')
        val_path = os.path.join(self.args.data_dir, 'val')
        # Compile Pre-processing
        train_preprocess = transforms.Compose([
            transforms.Resize((self.args.size, self.args.size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.args.mean,
                                 self.args.std),
        ])
        test_preprocess = transforms.Compose([
            transforms.Resize((self.args.size, self.args.size)),
            transforms.ToTensor(),
            transforms.Normalize(self.args.mean,
                                 self.args.std),
        ])

        # Compile Dataset
        train_dataset = IQADataset(args, train_path, train_preprocess)
        test_dataset = IQADataset(args, test_path, test_preprocess)
        val_dataset = IQADataset(args, val_path, test_preprocess)

        print('[Train]: ', train_dataset.__len__())
        print('[Val]: ', val_dataset.__len__())
        print('[Test]: ', test_dataset.__len__())

        weights, weights_per_class = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes))
        #print('Use sample weights')
        weights = torch.DoubleTensor(weights)
        # Compile Sampler
        args.weight = torch.FloatTensor(weights_per_class)
        # This sampler only works for
        weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=True)
        #weighted_sampler = None
        #train_targets = [sampler[1] for sampler in train_dataset.samples]
        #weighted_sampler = ScheduledWeightedSampler(len(train_dataset), train_targets, True)
        # Compile DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size,
                            sampler=weighted_sampler, num_workers=self.args.n_threads)
        val_dataloader = DataLoader(val_dataset, batch_size=self.args.batch_size,
                                      shuffle=True, num_workers=self.args.n_threads)
        test_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size,
                                    shuffle=False, num_workers=self.args.n_threads)
        return train_dataloader, val_dataloader, test_dataloader




