import torch
from torch.utils.data.sampler import Sampler
import random
import os
import glob
from PIL import Image
from scipy.io import loadmat
from torchvision.transforms import ToPILImage
import numpy as np



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


class ORIGIADataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, gt_transforms, stage=0):
        super(ORIGIADataset, self).__init__()
        self.root = root
        self.transforms = transforms
        self.gt_transforms = gt_transforms
        self.imgs_path =  os.path.join(root, 'images_{}'.format(stage))
        self.labels_path = os.path.join(root, 'gts_{}'.format(stage))
        self.imgs_list =  sorted(glob.glob(os.path.join(self.imgs_path, '*.jpg')))
        self.labels_list =  sorted(glob.glob(os.path.join(self.labels_path, '*.mat')))


    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        img = Image.open(self.imgs_list[index])
        label = loadmat(self.labels_list[index])['mask']
        # Keep the same transformation
        seed = np.random.randint(999999999)
        random.seed(seed)

        # TODO: There exists a bug, when you use ToTensor in transform.
        # TODO: The value will be changed in label
        label = np.asarray(self.gt_transforms(label))
        label = torch.from_numpy(label)
        random.seed(seed)
        img = self.transforms(img)

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
