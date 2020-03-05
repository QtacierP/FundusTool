from model.common import AbstractModel
from model.backbone import UNet
from torch.optim import Adam, SGD, lr_scheduler
import torch.nn as nn
from model.callback import MyLogger
import torch
from utils import accuracy, seg_accuracy, dice, convert_to_one_hot
import numpy as np
import os
from imageio import imsave
from tqdm import tqdm
from utils import UnNormalize
import cv2


class MyModel(AbstractModel):
    def __init__(self, args):
        super(MyModel, self).__init__(args)

    def _init_model(self):
        if self.args.model == 'unet':
            self.model = UNet(num_classes=self.args.n_classes,
                              regression=self.args.regression).cuda()

        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        # self.optimizer = SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
        if self.args.regression:
            self.loss = nn.L1Loss().cuda()
        else:
            if self.args.n_classes == 2:
                self.loss = nn.BCELoss().cuda()
            else:
                self.loss = nn.CrossEntropyLoss().cuda()
        if self.args.n_gpus > 1:
            self.model = nn.DataParallel(self.model)



    def train(self, train_dataloader, val_dataloader):
        losses_name = ['loss', 'disc_dice',
                       'cup_dice', 'accuracy']
        step = train_dataloader.__len__()
        # warmup_scheduler = WarmupLRScheduler(self.optimizer, self.args.epochs * step, self.args.lr)
        warmup_scheduler = None
        milestones = [60, 80, 90]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        logger = MyLogger(self.args, self.args.epochs, self.args.batch_size,
                          losses_name, step=step, model=self.model,
                          metric=self.args.metric, optimizer=self.optimizer,
                          warmup_scheduler=warmup_scheduler, lr_scheduler=lr_scheduler)
        self.logger = logger
        logger.on_train_begin()

        for epoch in range(self.args.epochs):
            logger.on_epoch_begin()
            for i, batch in enumerate(train_dataloader):
                losses = {}
                logger.on_batch_begin()
                self.optimizer.zero_grad()
                x, y = batch
                x = x.cuda()
                y = y.long().cuda()
                # Forward
                pred = self.model(x)
                loss = self.loss(pred, y.float())
                loss.backward()
                acc, correct = seg_accuracy(pred, y, supervised=True,
                                            regression=self.args.regression)
                overall_dice = dice(y_true=y, y_pred=pred, target=0)
                cup_dice = dice(y_true=y, y_pred=pred, target=2)
                self.optimizer.step()
                losses['loss'] = loss.detach().cpu().numpy()
                losses['accuracy'] = acc
                losses['disc_dice'] = overall_dice.detach().cpu().numpy()
                losses['cup_dice'] = cup_dice.detach().cpu().numpy()
                logger.on_batch_end(losses)

            vis = True

            val_acc, val_disc_dice, val_cup_dice,  = self.test(val_dataloader, val=True, vis=vis)
            metric = {'val_accuracy': val_acc, 'val_disc_dice': val_disc_dice,
                      'val_cup_dice': val_cup_dice}
            logger.on_epoch_end(metric)
        logger.on_train_end()

    def test(self, test_dataloader, val=False, vis=True):
        self.model.eval()
        torch.set_grad_enabled(False)
        total = 0
        correct = 0
        tp = [0, 0]
        tn = [0, 0]
        fp = [0, 0]
        fn = [0, 0]
        images = []
        preds = []
        gts = []
        for test_data in tqdm(test_dataloader):
            x, y = test_data
            x, y = x.cuda(), y.long().cuda()
            if vis:
                un = UnNormalize(self.args.mean,
                                     self.args.std)
                images += [un(x[i, ...]).permute(1, 2, 0).cpu().numpy()
                           for i in range(y.size(0))]
                gts += [y.cpu().numpy()[i, ...]
                           for i in range(y.size(0))]
            y_pred = self.model(x)

            if vis:
                if self.args.n_classes >= 2: # Regularize to [0, 1]
                    softmax_pred = nn.Softmax(dim=1)(y_pred)
                else:
                    softmax_pred = y_pred
                preds += [softmax_pred.detach()[i, ...].permute(1, 2, 0).cpu().numpy()
                           for i in range(y.size(0))]
            total += y.size(0)
            correct += seg_accuracy(y_pred, y,regression=self.args.regression) * y.size(0)
            for i in range(2):
                tp_, tn_, fp_, fn_, _ = dice(y, y_pred, supervised=True, target=i)
                tp[i] += tp_
                tn[i] += tn_
                fp[i] += fp_
                fn[i] += fn_
        epsilon = 1e-7
        dice_list = []
        for i in range(2):
            precision = tp[i] / (tp[i] + fp[i] + epsilon)
            recall = tp[i] / (tp[i] + fn[i] + epsilon)
            dice_list.append(2 * (precision * recall) /
                        (precision + recall + epsilon))

        acc = round(correct / total, 4)
        print('')
        self.model.train()
        torch.set_grad_enabled(True)
        if vis:
            self._vis(images, gts=gts, preds=preds, val=val)
        return acc , dice_list[0].detach().cpu().numpy(), dice_list[1].detach().cpu().numpy()

    def _vis(self, images, gts, preds, out_dir=None, val=False):
        # TODO: Only works for OD & OC segmenting
        if out_dir is None:
            if val:
                out_dir = os.path.join(self.args.model_path, 'val_samples')
            else:
                out_dir = os.path.join(self.args.model_path, 'test_samples')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # Convert gts
        print('Saving samples ...')
        for i in tqdm(range(len(images))):
            image = (images[i] * 255).astype(np.uint8)
            from sklearn.metrics import accuracy_score
            gt = (gts[i] * 255).astype(np.uint8)
            pred = (preds[i] * 255).\
                astype(np.uint8)

            if gt.shape[-1] == 1 and len(gt.shape) == 3:
                gt = np.squeeze(gt, axis=-1)
            if pred.shape[-1] == 1 and len(pred.shape) == 3:
                pred = np.squeeze(pred, axis=-1)
            if len(gt.shape) == 2:
                gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
            if len(pred.shape) == 2:
                pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)

            result = np.hstack((image, gt, pred))
            imsave(os.path.join(out_dir, '{}.jpg'.format(i)), result)
