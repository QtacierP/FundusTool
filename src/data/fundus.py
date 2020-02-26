from data.common import AbstractDataLoader, ScheduledWeightedSampler, PeculiarSampler, make_weights_for_balanced_classes
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from torchvision import datasets, transforms
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader


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
                    for c in range(1, 6):
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
                    for c in range(1, 6):
                        c = str(c)
                        img_path = os.path.join(root_path, task, c, img_name)
                        if os.path.exists(img_path):
                            to_path = os.path.join(self.args.data_dir, task, c)
                            if not os.path.exists(to_path):
                                os.makedirs(to_path)
                            os.system('cp {} {}'.format(img_path, to_path))
                            print('cp {} {}'.format(img_path, to_path))
    def load(self):
        train_path = os.path.join(self.args.data_dir, 'train')
        test_path = os.path.join(self.args.data_dir, 'test')
        val_path = os.path.join(self.args.data_dir, 'val')

        # Compile Pre-processing
        train_preprocess = transforms.Compose([

            transforms.RandomResizedCrop(
                size=self.args.size,
                scale=(1 / 1.15, 1.15),
                ratio=(0.7561, 1.3225)
            ),
            transforms.RandomAffine(
                degrees=(-180, 180),
                translate=(40 / 448, 40 / 448),
                scale=None,
                shear=None
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.args.mean,
                                 self.args.std)
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
        weights = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes))
        print('Use sample weights')
        print(weights)
        # Compile Sampler
        train_targets = [sampler[1] for sampler in train_dataset.samples]
        # This sampler only works for
        weighted_sampler = PeculiarSampler(len(train_dataset), train_targets, self.args.batch_size, weights, False)
        #weighted_sampler = None
        # Compile DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size,
                            sampler=weighted_sampler, num_workers=self.args.n_threads)
        val_dataloader = DataLoader(val_dataset, batch_size=self.args.batch_size,
                                      shuffle=True, num_workers=self.args.n_threads)
        test_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size,
                                    shuffle=False, num_workers=self.args.n_threads)
        return train_dataloader, val_dataloader, test_dataloader




