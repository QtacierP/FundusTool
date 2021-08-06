import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from model.common import AbstractModel, UncertaintyLoss
from model.backbone import InceptionV3_backbone, \
    resnet101_backbone, simple_classifier, \
    EfficientNet_backbone, resnet50_backbone, UncertaintyDRResNet50
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
            self.model = InceptionV3_backbone(self.args.n_classes, self.args.regression)
        elif self.args.model == 'resnet101':
            self.model = resnet101_backbone(self.args.n_classes, self.args.n_colors, self.args.regression)
        elif self.args.model == 'resnet50':
            self.model = resnet50_backbone(self.args.n_classes, self.args.n_colors, self.args.regression, size=self.args.size)
        elif self.args.model == 'uncertainty_resnet50':
            self.model = UncertaintyDRResNet50(self.args.n_classes, self.args.n_colors, self.args.regression,
                                           size=self.args.size, trainable= not self.args.test).cuda()
        elif self.args.model == 'simple':
            self.model = simple_classifier(self.args.n_classes, self.args.regression)
        elif 'efficient' in self.args.model:
            self.model = EfficientNet_backbone(self.args.n_classes, self.args.regression, self.args.model)
        else:
            raise NotImplementedError('{} not implemented yet'.
                                      format(self.args.model))

        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        #self.optimizer = SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
        if self.args.regression:
            self.loss = nn.MSELoss().cuda()
        else:
            self.loss = nn.CrossEntropyLoss().cuda()
        if 'uncertainty' in self.args.model:
            if self.args.regression:
                self.uncertainty = True
                self.loss = UncertaintyLoss().cuda()
            else:
                self.uncertainty = False
                raise NotImplementedError()
        if self.args.n_gpus > 1:
            self.model = nn.DataParallel(self.model)


    def train(self, train_dataloader, val_dataloader):

        step = train_dataloader.__len__()
        warmup_scheduler = WarmupLRScheduler(self.optimizer, self.args.warm_epochs * step, self.args.lr)
        #warmup_scheduler = Nonewarmup_batch = len(train_loader) * warmup_epoch
        remain_batch = step * (self.args.epochs - self.args.warm_epochs)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=remain_batch)
        logger = MyLogger(self.args, self.args.epochs, self.args.batch_size,
                          step=step, model=self.model,
                          metric=self.args.metric, optimizer=self.optimizer,
                          warmup_scheduler=warmup_scheduler, lr_scheduler=lr_scheduler, weighted_sampler=train_dataloader.sampler)
        logger.on_train_begin()
        for epoch in range(self.args.epochs):
            over_all = 0
            logger.on_epoch_begin()
            for i, batch in enumerate(train_dataloader):
                losses = {}
                logger.on_batch_begin()
                self.optimizer.zero_grad()
                x, y = batch
                x = x.cuda()
                y = y.long().cuda()
                # Forward
                if self.uncertainty:
                    pred, logger_var = self.model([x, True])
                    loss = self.loss([pred, logger_var], y)
                else:
                    pred = self.model(x)
                    loss = self.loss(pred, y)
                acc, correct = accuracy(pred, y, supervised=True, regression=self.args.regression)
                loss.backward()
                self.optimizer.step()
                losses['loss'] = loss.detach().cpu().numpy()
                losses['accuracy'] = acc
                over_all += correct
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
        logits = []
        labels = []
        for test_data in test_dataloader:
            x, y = test_data
            x, y = x.cuda(), y.long().cuda()
            y_pred = self.model(x)
            total += y.size(0)
            logits += y_pred.cpu().tolist()
            labels  += y.cpu().tolist()
            correct += accuracy(y_pred, y, c_matrix, regression=self.args.regression) * y.size(0)
        acc = round(correct / total, 4)
        kappa = quadratic_weighted_kappa(c_matrix)
        print('')
        print(c_matrix)
        self.model.train()
        torch.set_grad_enabled(True)
        return acc, kappa










