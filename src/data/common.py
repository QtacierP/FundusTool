import torch
from torch.utils.data.sampler import Sampler
import random

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


class ScheduledWeightedSampler(Sampler):
    def __init__(self, args, num_samples, train_targets, replacement=True):

        self.num_samples = num_samples
        self.train_targets = train_targets
        self.replacement = replacement
        self.epoch = 0
        self.decay_rate = args.decay_rate
        self.w0 = torch.as_tensor([1] * args.n_classes, dtype=torch.double)
        self.wf = torch.as_tensor([1] * args.n_classes, dtype=torch.double)
        self.train_sample_weight = torch.zeros(len(train_targets), dtype=torch.double)

    def step(self):
        self.epoch += 1
        factor = self.decay_rate**(self.epoch - 1)
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
        weight_per_class[i] = N/float(count[i])

    return weight_per_class

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
