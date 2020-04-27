import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
import os
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy
from tqdm import tqdm
import heapq

def normalize(imgs):
    return imgs / 127.5 - 1

def classify(predict, thresholds=[0, 0.5, 1.5, 2.5, 3.5]):
    predict = max(predict, thresholds[0])
    for i in reversed(range(len(thresholds))):
        if predict >= thresholds[i]:
            return i

def accuracy(predictions, targets, c_matrix=None, supervised=False, regression=False):
    predictions = predictions.data
    targets = targets.data
    if not regression:
        # avoid modifying origin predictions

        predicted = torch.tensor(
            [torch.argmax(p) for p in predictions]
        ).cuda().long()
    else:

        predicted =  torch.tensor(
        [classify(p.item()) for p in predictions]
    ).cuda().float()

    # update confusion matrix
    if c_matrix is not None:
        for i, p in enumerate(predicted):
            c_matrix[int(targets[i])][int(p.item())] += 1

    correct = (predicted == targets).sum().item()
    if supervised:
        return correct / len(predicted), correct
    else:
        return correct / len(predicted)

def seg_accuracy(predictions, targets, supervised=False, regression=False):
    if len(predictions.shape) == 4 and predictions.shape[1] > 1:
        predicted = torch.argmax(predictions, dim=1).cuda().long().flatten()
    else:
        predicted = predictions.cuda().long().flatten()
        predicted = torch.where(predicted >= 0.5, torch.ones_like(predicted),
                                torch.zeros_like(predicted))
    targets = targets.data.flatten()
    correct = (predicted == targets).sum().item()
    if supervised:
        return correct / len(predicted), correct
    else:
        return correct / len(predicted)

class DICELoss(nn.Module):
    def __init__(self):
        super(DICELoss, self).__init__()
    def forward(self, output, mask):
        probs = torch.squeeze(output, 1)
        mask = torch.squeeze(mask, 1)

        intersection = probs * mask
        intersection = torch.sum(intersection, 2)
        intersection = torch.sum(intersection, 1)

        den1 = probs * probs
        den1 = torch.sum(den1, 2)
        den1 = torch.sum(den1, 1)

        den2 = mask * mask
        den2 = torch.sum(den2, 2)
        den2 = torch.sum(den2, 1)

        eps = 1e-8
        dice = 2 * ((intersection + eps) / (den1 + den2 + eps))
        dice_eso = dice

        loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)
        return loss

def dice(y_true: torch.Tensor, y_pred: torch.Tensor, target=0, supervised=False) -> torch.Tensor:
    # Concert to flatten tensor
    if len(y_pred.shape) == 4 and y_pred.shape[1] > 1:
        y_pred = torch.argmax(y_pred, dim=1) # Get label map
    else:
        y_pred = torch.where(y_pred >= 0.5, torch.ones_like(y_pred),
                             torch.zeros_like(y_pred))
    target_pred = torch.where(y_pred == target,
                              torch.ones_like(y_pred),
                              torch.zeros_like(y_pred)).flatten()
    target_true = torch.where(y_true == target,
                         torch.ones_like(y_true),
                         torch.zeros_like(y_true)).flatten()
    
    tp = (target_true * target_pred).sum().to(torch.float32)
    tn = ((1 - target_true) * (1 - target_pred)).sum().to(torch.float32)
    fp = ((1 - target_true) * target_pred).sum().to(torch.float32)
    fn = (target_true * (1 - target_pred)).sum().to(torch.float32)

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    if supervised:
        return tp, \
               tn, \
               fp, \
               fn, f1
    else:
        return f1

def quadratic_weighted_kappa(conf_mat):
    assert conf_mat.shape[0] == conf_mat.shape[1]
    cate_num = conf_mat.shape[0]

    # Quadratic weighted matrix
    weighted_matrix = np.zeros((cate_num, cate_num))
    for i in range(cate_num):
        for j in range(cate_num):
            weighted_matrix[i][j] = 1 - float(((i - j)**2) / ((cate_num - 1)**2))

    # Expected matrix
    ground_truth_count = np.sum(conf_mat, axis=1)
    pred_count = np.sum(conf_mat, axis=0)
    expected_matrix = np.outer(ground_truth_count, pred_count)

    # Normalization
    conf_mat = conf_mat / conf_mat.sum()
    expected_matrix = expected_matrix / expected_matrix.sum()

    observed = (conf_mat * weighted_matrix).sum()
    expected = (expected_matrix * weighted_matrix).sum()
    return (observed - expected) / (1 - expected)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def convert_to_one_hot(gt, c):
    h, w = np.shape(gt)[: 2]
    one_hot = np.zeros((h, w, c))
    for i in range(h):
        for j in range(w):
            one_hot[i, j, gt[i, j]] = 1
    return one_hot

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]



def get_statics(args, preds, gts, imgs, samples=5):
    #  Only for optic segmentation
    assert len(preds.shape) == 4
    assert preds.shape[-1] == 3
    ori_preds = np.copy(preds)
    preds = np.argmax(preds, axis=-1)
    if len(gts.shape) == 4:
        if gts.shape[-1] != 1:
            gts = np.argmax(gts, axis=-1)
    assert len(preds.shape) == 3
    if len(gts.shape) == 4:
        assert gts.shape[-1] == 1
        gts = np.squeeze(gts, axis=-1)
    dices = []
    for i in range(2):
        dices.append([])
    target_list = [0 ,2]
    N = preds.shape[0]
    for i in tqdm(range(N)):
        pred = preds[i, ...]
        gt = gts[i, ...]
        for target in range(2):
            target_pred = np.where(pred == target_list[target], np.ones_like(pred),
                                   np.zeros_like(pred))
            target_gt = np.where(gt == target_list[target], np.ones_like(gt),
                                 np.zeros_like(gt))
            dice_ = f1_score(target_gt.flatten(), target_pred.flatten())
            dices[target].append(dice_)
    dice_names = ['disc_dice', 'cup_dice']
    max_dices = [{}, {}]
    min_dices = [{}, {}]
    max_dice = []
    min_dice = []
    mean_dices = []
    ori_dices = deepcopy(dices)
    for i in range(2):
        dice_list = np.asarray(dices[i])
        mean_dices.append(np.mean(dice_list))
        max_temp  = heapq.nlargest(samples, range(len(dice_list)), dice_list.take)
        for id in max_temp:
            max_dices[i][id] = dice_list[id]
        max_dice.append(dice_list[max_temp[0]])
        min_temp = heapq.nsmallest(samples, range(len(dice_list)), dice_list.take)
        for id in min_temp:
            min_dices[i][id] = dice_list[id]
        min_dice.append(dice_list[min_temp[0]])

    out_dir = '../data/experiment/optic_statics/{}'.format(args.dataset)
    # Save part
    for i in range(2):
        for id in max_dices[i].keys():
            task_dir = os.path.join(out_dir,  dice_names[i], 'max')
            if not os.path.exists(task_dir):
                os.makedirs(task_dir)
            max_img = (imgs[id] * 255).astype(np.uint8)
            max_gt = (gts[id] / (args.n_classes - 1)* 255).astype(np.uint8)
            max_pred = (ori_preds[id] / (args.n_classes - 1) * 255).astype(np.uint8)
            max_dice_ = max_dices[i][id]
            file_name = os.path.join(task_dir, '{}.jpg'.format(max_dice_))
            max_gt = cv2.cvtColor(max_gt, cv2.COLOR_GRAY2RGB)
            result = np.hstack((max_img, max_gt, max_pred))
            plt.imsave(file_name, result)
        for id in min_dices[i].keys():
            task_dir = os.path.join(out_dir, dice_names[i], 'min')
            if not os.path.exists(task_dir):
                os.makedirs(task_dir)
            min_img = (imgs[id] * 255).astype(np.uint8)
            min_gt = (gts[id]  / (args.n_classes - 1) * 255).astype(np.uint8)
            min_pred = (ori_preds[id] / (args.n_classes - 1) * 255).astype(np.uint8)
            min_dice_ = min_dices[i][id]
            file_name = os.path.join(task_dir, '{}.jpg'.format(min_dice_))
            min_gt = cv2.cvtColor(min_gt, cv2.COLOR_GRAY2RGB)
            result = np.hstack((min_img, min_gt, min_pred))
            plt.imsave(file_name, result)
    print(mean_dices)
    print(min_dice)
    f = open(os.path.join(out_dir, 'result.txt'), 'w')
    for i in range(2):
        sigma = min(mean_dices[i] - min_dice[i],
                    max_dice[i] - mean_dices[i])
        f.write('{}: {} +- {} \n'.format(dice_names[i], mean_dices[i], sigma))
    f.close()
    return ori_dices





