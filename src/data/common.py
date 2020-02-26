import torch
from torch.utils.data.sampler import Sampler


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