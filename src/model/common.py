import os
import torch
from torch import Tensor
import torch.nn.functional as F
from model.backbone import UNet
from torchvision import datasets, transforms
from PIL.Image import NEAREST


class AbstractModel():
    def __init__(self, args):
        self.args = args
        self._init_model()
        self.callbacks = []
        self._init_callbacks()
        if args.test or args.resume:
            self.load()

    def _init_model(self):
        print("[INFO] Initialize model...")
        pass

    def _init_callbacks(self):
        pass

    def train(self, train_dataloader, val_dataloader):
        pass

    def test(self, test_dataloader):
        pass

    def load(self, force=False):
        if self.args.n_gpus > 1:
            module = self.model.module
        else:
            module = self.model
        if self.args.resume:
            module.load_state_dict(torch.load(os.path.join(self.args.model_path,
                                             '{}_last.pt'.format(self.args.model))))
        elif self.args.test or force:
            print('load model from ', os.path.join(self.args.model_path,
                                             '{}_best.pt'.format(self.args.model)))
            module.load_state_dict(torch.load(os.path.join(self.args.model_path,
                                             '{}_best.pt'.format(self.args.model))))
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class DALoss(torch.nn.Module):
    def __init__(self, model):
        super(DALoss, self).__init__()
        self.model = model

    def forward(self):
        pass

class UncertaintyLoss(torch.nn.Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()


    def forward(self, input, y):
        mean, log_var = input
        input_var = torch.exp(-log_var)
        loss = torch.mean(0.5 * torch.mul((input_var), (mean - y) ** 2) + 0.5 * log_var)
        return loss

