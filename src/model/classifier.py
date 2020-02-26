import torch
import torch.nn as nn
from torch.optim import Adam
from model.common import AbstractModel
from model.backbone import InceptionV3_backbone, resnet101_backbone
from model.callback import MyLogger, WarmupLRScheduler
from torch.utils.data import DataLoader
from utils import accuracy, quadratic_weighted_kappa
import numpy as np


class MyModel(AbstractModel):
    def __init__(self, args):
        super(MyModel, self).__init__(args)

    def _init_model(self):
        super(MyModel, self)._init_model()
        if self.args.model == 'InceptionV3':
            self.model = InceptionV3_backbone(self.args.n_classes)
        elif self.args.model == 'resnet101':
            self.model = resnet101_backbone(self.args.n_classes)
        else:
            raise NotImplementedError('{} not implemented yet'.
                                      format(self.args.model))
        if self.args.n_gpus > 1:
            self.model = nn.DataParallel(self.model)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.loss = nn.CrossEntropyLoss().cuda()


    def train(self, train_dataloader, val_dataloader):
        losses_name = [self.args.loss, 'accuracy']
        step = train_dataloader.__len__()
        warmup_scheduler = None
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR\
            (self.optimizer, T_max=(self.args.epochs -
                                    self.args.warm_epochs) * step)
        logger = MyLogger(self.args, self.args.epochs, self.args.batch_size,
                          losses_name, step=step, model=self.model,
                          metric=self.args.metric, optimizer=self.optimizer, weighted_sampler=train_dataloader.sampler,
                          warmup_scheduler=warmup_scheduler, lr_scheduler=lr_scheduler)
        logger.on_train_begin()
        for epoch in range(self.args.epochs):
            logger.on_epoch_begin()
            for i, batch in enumerate(train_dataloader):
                losses = {}
                logger.on_batch_begin()
                self.optimizer.zero_grad()
                x, y = batch
                x = x.cuda()
                y = y.cuda()
                # Forward
                pred = self.model(x)
                loss = self.loss(pred, y)
                acc, correct = accuracy(pred, y, supervised=True)
                loss.backward()
                self.optimizer.step()
                losses[self.args.loss] = loss.detach().cpu().numpy()
                losses['accuracy'] = [acc, correct]
                logger.on_batch_end(losses)
            val_acc, val_kappa = self.test(val_dataloader)
            metric = {'val_accuracy': val_acc, 'val_kappa': val_kappa}
            logger.on_epoch_end(metric)
        logger.on_train_end()


    def test(self, test_dataloader):
        c_matrix = np.zeros((self.args.n_classes
                             , self.args.n_classes), dtype=int)
        self.model.eval()
        torch.set_grad_enabled(False)
        total = 0
        correct = 0
        for test_data in test_dataloader:
            x, y = test_data
            x, y = x.cuda(), y.long().cuda()
            y_pred = self.model(x)
            total += y.size(0)
            correct += accuracy(y_pred, y, c_matrix) * y.size(0)
        acc = round(correct / total, 4)
        kappa = quadratic_weighted_kappa(c_matrix)
        print(c_matrix)
        self.model.train()
        torch.set_grad_enabled(True)
        return acc, kappa










