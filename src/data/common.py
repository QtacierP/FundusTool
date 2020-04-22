import torch
from torch.utils.data.sampler import Sampler
import random
import os
import glob
from scipy.io import loadmat
import numpy as np
import skimage.io
import torchvision
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
import cv2
from PIL import Image, ImageFilter
from matplotlib.pyplot import imsave, imread
from skimage.measure import label, regionprops
import scipy
import matplotlib.pyplot as plt


BALANCE_WEIGHTS = torch.tensor([1.3609453700116234, 14.378223495702006,
                                6.637566137566138, 40.235967926689575,
                                49.612994350282484], dtype=torch.double)
FINAL_WEIGHTS = torch.as_tensor([1, 2, 2, 2, 2], dtype=torch.double)

# for color augmentation, computed by origin author
U = torch.tensor([[-0.56543481, 0.71983482, 0.40240142],
                  [-0.5989477, -0.02304967, -0.80036049],
                  [-0.56694071, -0.6935729, 0.44423429]], dtype=torch.float32)
EV = torch.tensor([1.65513492, 0.48450358, 0.1565086], dtype=torch.float32)

class AbstractDataLoader():
    def __init__(self, args):
        self.args = args
        if args.prepare:
            self.prepare()

    def prepare(self):
        # Prepare your dataset here
        pass

    def load(self):
        # Load your dataset
        pass

class IQADataset(torchvision.datasets.DatasetFolder):
    def __init__(self, args, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        self.args = args
        super(IQADataset, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = skimage.io.imread(path)
        if self.args.n_colors == 1:
            '''sample = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE()
            sample = clahe.apply(sample)'''
            sample = 0.25 * sample[..., 0] + 0.75 * sample[..., 1]

        sample = Image.fromarray(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class ORIGIADataset(torch.utils.data.Dataset):
    def __init__(self,args, root, transforms, gt_transforms, stage=0, need_name=False):
        super(ORIGIADataset, self).__init__()
        self.root = root
        self.args = args
        self.transforms = transforms
        self.gt_transforms = gt_transforms
        self.imgs_path =  os.path.join(root, 'images_{}'.format(stage))
        self.labels_path = os.path.join(root, 'gts_{}'.format(stage))
        self.imgs_list =  sorted(glob.glob(os.path.join(self.imgs_path, '*.jpg')))
        self.labels_list =  sorted(glob.glob(os.path.join(self.labels_path, '*.mat')))
        self.need_name = need_name

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        name = self.imgs_list[index]
        ori_img = Image.open(name)
        ori_label = loadmat(self.labels_list[index])['mask']
        # Keep the same transformation
        seed = np.random.randint(999999999)
        random.seed(seed)
        # TODO: There exists a bug, when you use ToTensor in transform.
        # TODO: The value will be changed in label
        label = np.asarray(self.gt_transforms(ori_label))
        #label = np.expand_dims(label, axis=-1)
        label = torch.from_numpy(label)

        if self.args.n_classes == 2:  # Convert to binary map
            label = torch.where(label > 0,
                                torch.ones_like(label), torch.zeros_like(label))
        random.seed(seed)
        img = self.transforms(ori_img)
        if not self.need_name:
            return img, label
        else:
            ori_img = cv2.resize(np.asarray(ori_img), (3072, 2048))
            ori_label = cv2.resize(ori_label, (3072, 2048))
            ori_label = np.expand_dims(ori_label, axis=-1)
            return img, label, name, ori_img, ori_label

class REFUGEDataset(torch.utils.data.Dataset):
    def __init__(self, args, root, transforms, gt_transforms,
                 task, stage=0, need_name=False):
        super(REFUGEDataset, self).__init__()
        self.args = args
        self.root = root
        self.transforms = transforms
        self.gt_transforms = gt_transforms
        self.need_name = need_name
        self.imgs_path = os.path.join(root,  'images_{}'.format(stage), task)
        self.labels_path = os.path.join(root, 'gts_{}'.format(stage), task)
        self.imgs_list = sorted(glob.glob(os.path.join(self.imgs_path, '*.jpg')))
        self.labels_list = sorted(glob.glob(os.path.join(self.labels_path, '*.bmp')))

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        name = self.imgs_list[index]
        ori_img = Image.open(name)
        ori_label = Image.open(self.labels_list[index])
        # Keep the same transformation
        seed = np.random.randint(999999999)
        random.seed(seed)
        label = self.gt_transforms(ori_label)
        if self.args.n_classes == 2:  # Convert to binary map
            label = torch.where(label > 0,
                                torch.ones_like(label), torch.zeros_like(label))
        random.seed(seed)
        img = self.transforms(ori_img)
        if not self.need_name:
            return img, label
        else:
            ori_img = np.asarray(ori_img)
            ori_label = cv2.resize(ori_label, (3072, 2048))
            ori_label = np.expand_dims(ori_label, axis=-1)
            return img, label, name, ori_img, ori_label

class TestSet(torch.utils.data.Dataset):
    def __init__(self, args, main_dir, transform):
        self.args = args
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        if args.len > 0:
            self.total_imgs = sorted(all_imgs)[:args.len]
        else:
            self.total_imgs = sorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, self.total_imgs[idx]

class DRIVEDataset(torch.utils.data.Dataset):
    def __init__(self,args, root, transforms, gt_transforms, step=1):
        super(DRIVEDataset, self).__init__()
        self.root = root
        self.args = args
        self.step = step
        self.transforms = transforms
        self.gt_transforms = gt_transforms
        if args.enhanced:
            self.imgs_path = os.path.join(root, 'enhanced_origin')
        else:
            self.imgs_path =  os.path.join(root, 'origin')
        self.labels_path = os.path.join(root, 'groundtruth')
        self.imgs_list =  sorted(glob.glob(os.path.join(self.imgs_path, '*')))
        self.labels_list =  sorted(glob.glob(os.path.join(self.labels_path, '*')))

    def __len__(self):
        return len(self.imgs_list) * self.step

    def __getitem__(self, index):
        img = Image.open(self.imgs_list[index % len(self.imgs_list)])
        label = Image.open(self.labels_list[index % len(self.imgs_list)])
        # Keep the same transformation
        seed = np.random.randint(999999999)
        random.seed(seed)
        label = self.gt_transforms(label)
        random.seed(seed)
        img = self.transforms(img)
        if self.args.n_colors == 1:
            img = 0.25 * img[0,  ...]  + 0.75 * img[1, ...]
            img = torch.unsqueeze(img, dim=0)
        return img, label

class ScheduledWeightedSampler(Sampler):
    def __init__(self, num_samples, train_targets, initial_weight=BALANCE_WEIGHTS,
                 final_weight=FINAL_WEIGHTS, replacement=True):
        self.num_samples = num_samples
        self.train_targets = train_targets
        self.replacement = replacement

        self.epoch = 0
        self.w0 = initial_weight
        self.wf = final_weight
        self.train_sample_weight = torch.zeros(len(train_targets), dtype=torch.double)

    def step(self):
        self.epoch += 1
        factor = 0.975**(self.epoch - 1)
        self.weights = factor * self.w0 + (1 - factor) * self.wf
        for i, _class in enumerate(self.train_targets):
            self.train_sample_weight[i] = self.weights[_class]

    def __iter__(self):
        return iter(torch.multinomial(self.train_sample_weight, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight, weight_per_class

class PeculiarSampler(Sampler):
    # Only used for Fundus
    def __init__(self, num_samples, train_targets, batch_size, weights, replacement=True):
        self.num_samples = num_samples
        self.train_targets = train_targets
        self.batch_size = batch_size
        self.replacement = replacement
        balance_weight = torch.tensor(weights, dtype=torch.double)
        self.epoch = 0
        self.args = list(range(num_samples))
        self.train_sample_weight = torch.zeros(len(train_targets), dtype=torch.double)
        for i, _class in enumerate(self.train_targets):
            self.train_sample_weight[i] = balance_weight[_class]

        self.epoch_samples = []

    def step(self):
        self.epoch_samples = []

        batch_size = self.batch_size
        batch_num = self.num_samples // self.batch_size
        for i in range(batch_num):
            r = random.random()
            if r < 0.2:
                self.epoch_samples += torch.multinomial(self.train_sample_weight, batch_size, self.replacement).tolist()
            elif r < 0.5:
                self.epoch_samples += random.sample(self.args, batch_size)
            else:
                self.epoch_samples += list(range(i * batch_size, (i + 1) * batch_size))

    def __iter__(self):
        return iter(self.epoch_samples)

    def __len__(self):
        return self.num_samples

class KrizhevskyColorAugmentation(object):
    # This augmentation may boost performance, but its inverse is hard to obtain,
    # We will abandon it
    def __init__(self, sigma=0.5):
        self.sigma = sigma
        self.mean = torch.tensor([0.0])
        self.deviation = torch.tensor([sigma])

    def __call__(self, img, color_vec=None):
        sigma = self.sigma
        if color_vec is None:
            if not sigma > 0.0:
                color_vec = torch.zeros(3, dtype=torch.float32)
            else:
                color_vec = torch.distributions.Normal(self.mean, self.deviation).sample((3,))
            color_vec = color_vec.squeeze()

        alpha = color_vec * EV
        noise = torch.matmul(U, alpha.t())
        noise = noise.view((3, 1, 1))
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={})'.format(self.sigma)


def crop_image(img):
    img = Image.fromarray(np.uint8(img))
    blurred = img.filter(ImageFilter.BLUR)
    ba = np.array(blurred)
    h, w, _ = ba.shape

    if w > 1.2 * h:
        left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)

        foreground = (ba > max_bg + 10).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()
        if bbox != None:
            left, upper, right, lower = bbox
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                bbox = None
    else:\
        bbox = None
    if bbox is None:
        return np.asarray(img)
    return np.asarray(img.crop(bbox))

def crop_and_save(current_file, aimed_list, root_task_path, current_task_path ):
    if current_file not in aimed_list:
        return
    current_file_name = os.path.join(current_task_path, current_file)
    img = imread(current_file_name)
    img = crop_image(img)
    print('saving ', os.path.join(root_task_path, current_file))
    imsave(os.path.join(root_task_path, current_file), img)

'''
========== This is for OD/OC coarse crop ===========
'''
def crop_OD(disc_map, ori_img, ori_gt=None, need_position=False):
    ori_shape = np.shape(ori_img)
    disc_map = disc_map[..., 1]
    disc_map = BW_img(disc_map, 0.5)
    disc_map = disc_map.astype(np.uint8)
    disc_map = cv2.resize(disc_map, (ori_shape[1], ori_shape[0]))

    regions = regionprops(label(disc_map))
    l_h = regions[0].bbox[0]
    h_h = regions[0].bbox[2]
    l_w = regions[0].bbox[1]
    h_w = regions[0].bbox[3]

    h = int(h_h - l_h)
    w = int(h_w - l_w)
    e_w = int((max(h, w) * 2 - w) // 2)
    e_h = int((max(h, w) * 2 - h) // 2)
    n_l_h = l_h - e_h
    n_h_h = h_h + e_h
    n_l_w = l_w - e_w
    n_h_w = h_w + e_w
    if n_l_h < 0:
        n_h_h -= n_l_h
        n_l_h = 0
    if n_h_h > ori_shape[0]:
        n_l_h -= n_h_h - ori_shape[0]
        n_h_h = ori_shape[0]
    if n_l_w < 0:
        n_h_w -= n_l_w
        n_l_w = 0
    if n_h_w > ori_shape[1]:
        n_l_w -= n_h_w - ori_shape[1]
        n_h_w = ori_shape[1]
    mini_img = ori_img[n_l_h: n_h_h, n_l_w: n_h_w, :]
    mini_gt = None
    if ori_gt is not None:
        ori_gt = np.expand_dims(ori_gt, axis=-1)
        mini_gt = ori_gt[n_l_h: n_h_h, n_l_w: n_h_w, :]
    if need_position:
        return mini_img, mini_gt, [n_l_h, n_h_h, n_l_w, n_h_w]
    return mini_img, mini_gt

def BW_img(input, thresholding):
    if input.max() > thresholding:
        binary = input > thresholding
    else:
        binary = input > input.max() / 2.0
    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))