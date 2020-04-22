from data.common import AbstractDataLoader, \
    ScheduledWeightedSampler, PeculiarSampler, \
    make_weights_for_balanced_classes, KrizhevskyColorAugmentation
import os
from torchvision import datasets, transforms
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from data.common import crop_image, crop_and_save
import torch
from option import args
from matplotlib.pyplot import imread, imsave
import multiprocessing



class MyDataLoader(AbstractDataLoader):
    def __init__(self, args):
        super(MyDataLoader, self).__init__(args)

    def prepare(self):
        if self.args.dataset == 'onlygood' or self.args.dataset == 'notbad':
            # We need to extract data from fundus
            if self.args.dataset == 'onlygood':
                quality_list = [0]
            else:
                quality_list = [0, 1]
            train_csv = os.path.join(self.args.data_dir, 'Label_EyeQ_train.csv')
            test_csv = os.path.join(self.args.data_dir, 'Label_EyeQ_test.csv')
            train_table = pd.read_csv(train_csv)
            root_path = '../data/fundus/EyePac'
            for idx in tqdm(range(train_table.shape[0])):
                print(int(train_table['quality'][idx]))
                if int(train_table['quality'][idx]) not in quality_list:
                    continue
                img_name = train_table['image'][idx]
                for task in ['train', 'val']:
                    for c in range(5):
                        c = str(c)
                        img_path = os.path.join(root_path, task, c, img_name)
                        if os.path.exists(img_path):
                            to_path = os.path.join(self.args.data_dir, task, c)
                            print(to_path)
                            if not os.path.exists(to_path):
                                os.makedirs(to_path)
                            os.system('cp {} {}'.format(img_path, to_path))
                            print('cp {} {}'.format(img_path, to_path))
            test_table = pd.read_csv(test_csv)
            for idx in tqdm(range(test_table.shape[0])):
                if int(test_table['quality'][idx]) not in quality_list:
                    continue
                img_name = test_table['image'][idx]
                for task in ['test']:
                    for c in range(5):
                        c = str(c)
                        img_path = os.path.join(root_path, task, c, img_name) + '/'
                        if os.path.exists(img_path):
                            to_path = os.path.join(self.args.data_dir, task, c)
                            if not os.path.exists(to_path):
                                os.makedirs(to_path)
                            os.system('cp -r {} {}'.format(img_path, to_path))
                            print('cp -r {} {}'.format(img_path, to_path))
        if self.args.dataset == 'extreme':
            root_path = '../data/fundus/EyePac'
            for task in ['train', 'val', 'test']:
                for c in [0, 4]:
                    c = str(c)
                    img_path = os.path.join(root_path, task, c)
                    to_path = os.path.join(self.args.data_dir, task)
                    if not os.path.exists(to_path):
                        os.makedirs(to_path)
                    os.system('cp -r {} {}'.format(img_path, to_path))
                    print('cp  -r {} {}'.format(img_path, to_path))
        if self.args.dataset == 'EyePac':
            print('Prepare for EyePac')
            root_path = '../data/fundus/kaggle'
            aimed_path = '../data/fundus/EyePac_512'
            target_path = '../data/fundus/EyePac'
            for task in ['train', 'val', 'test']:
                for c in range(5):
                    aimed_task_path = os.path.join(aimed_path, task, str(c))
                    aimed_list = sorted(os.listdir(aimed_task_path))
                    if task == 'val':
                        current = 'train'
                    else:
                        current = task
                    current_task_path = os.path.join(root_path, current)
                    current_list = sorted(os.listdir(current_task_path))
                    root_task_path = os.path.join(target_path, task, str(c))
                    if not os.path.exists(root_task_path):
                        os.makedirs(root_task_path)
                    p = multiprocessing.Pool(self.args.n_threads)
                    for i in tqdm(range(0, len(current_list))):
                        current_file = current_list[i]
                        p.apply_async(crop_and_save, args=(current_file, aimed_list, root_task_path, current_task_path))
                    print('Waiting for all subprocesses done...')
                    p.close()
                    p.join()
                    print('All subprocesses done.')
        if self.args.dataset == 'clean':
            pre_train = '../model/iqa/EyeQ_512_512/resnet50/resnet50_best.pt'
            from model.backbone import resnet50_backbone
            from data.common import TestSet
            import numpy as np
            import pickle
            import torch.nn as nn
            model = resnet50_backbone(3)
            model.load_state_dict(torch.load(pre_train))
            if self.args.n_gpus > 1:
                model = nn.DataParallel(model)
            model.eval()
            torch.set_grad_enabled(False)
            root_path = '../data/fundus/EyePac_512'
            target_root_path = '../data/fundus/clean'
            if not os.path.exists(target_root_path):
                os.makedirs(target_root_path)
            f = open(os.path.join(target_root_path, 'logits'), 'wb')
            logits = {}
            for task in ['train', 'val']:
                for c in range(5):
                    c = str(c)
                    task_path = os.path.join(root_path, task, c)
                    target_path = os.path.join(target_root_path, task, c)
                    if not os.path.exists(target_path):
                        os.makedirs(target_path)
                    transformation = transforms.Compose([
                        transforms.Resize((self.args.size, self.args.size)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5],
                                             [0.5, 0.5, 0.5]),
                    ])
                    dataset = TestSet(self.args, task_path, transformation)
                    test_dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                                 shuffle=False, num_workers=self.args.n_threads)
                    for test_data in tqdm(test_dataloader):
                        x, files = test_data
                        x = x.cuda()
                        scores = model(x)
                        scores = scores.detach().cpu().numpy()
                        for i, score in enumerate(scores):
                            max_id = np.max(score)
                            file_name = files[i]
                            full_file_name = os.path.join(task_path, file_name)
                            if max_id < 2:
                                os.system('cp -r {} {}'.format(full_file_name, target_path))
                                print('cp -r {} {}'.format(full_file_name, target_path))
                            file_name = os.path.join(task, c, file_name)
                            logits[file_name] = score
            pickle.dump(logits, f, -1)
            f.close()




















    def load(self):
        train_path = os.path.join(self.args.data_dir, 'train')
        test_path = os.path.join(self.args.data_dir, 'test')
        val_path = os.path.join(self.args.data_dir, 'val')
        # Compile Pre-processing
        train_preprocess = transforms.Compose([
            transforms.RandomResizedCrop(
                size= self.args.size,
                scale=(1 / 1.15, 1.15),
                ratio=(0.7561, 1.3225)
            ),
            transforms.RandomAffine(
                degrees=(-180, 180),
                translate=(40 / self.args.size, 40 / self.args.size),
                scale=None,
                shear=None
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.args.mean,
                                 self.args.std),
            KrizhevskyColorAugmentation(sigma=0.5)
        ])

        test_preprocess = transforms.Compose([
            transforms.Resize((self.args.size, self.args.size)),
            transforms.ToTensor(),
            transforms.Normalize(self.args.mean,
                                 self.args.std),
        ])

        # Compile Dataset
        train_dataset = datasets.ImageFolder(train_path, train_preprocess)
        test_dataset = datasets.ImageFolder(test_path, test_preprocess)
        val_dataset = datasets.ImageFolder(val_path, test_preprocess)
        weights, weights_per_class  = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes))
        print('Use sample weights')
        #weights= torch.DoubleTensor(weights)
        # Compile Sampler
        print('[Train]: ', train_dataset.__len__())
        print('[Val]: ', val_dataset.__len__())
        print('[Test]: ', test_dataset.__len__())
        # This sampler only works for
        args.weight = torch.FloatTensor(weights_per_class)
        #weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=False)
        #weighted_sampler = None
        train_targets = [sampler[1] for sampler in train_dataset.samples]
        weighted_sampler = ScheduledWeightedSampler(len(train_dataset), train_targets, True)
        # Compile DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size,
                            sampler=weighted_sampler, num_workers=self.args.n_threads)
        val_dataloader = DataLoader(val_dataset, batch_size=self.args.batch_size,
                                      shuffle=True, num_workers=self.args.n_threads)
        test_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size,
                                    shuffle=False, num_workers=self.args.n_threads)
        return train_dataloader, val_dataloader, test_dataloader




