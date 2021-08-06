import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import os
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy
from tqdm import tqdm
import heapq
from  scipy.stats import ttest_rel
from torchvision import datasets, transforms
import csv
import imageio
import multiprocessing
EPS = 1e-7


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
        sigma = np.std(dices[i])
        f.write('{}: {}  +- {} \n'.format(dice_names[i], mean_dices[i], sigma))
    f.close()
    return ori_dices

def get_compare(args, imgs, e_imgs, preds, e_preds,
                gts, dices, e_dices, samples=5, grade=''):
    diffs = []
    for i in range(2):
        diffs.append(np.asarray(e_dices[i]) - np.asarray(dices[i]))
    dice_names = ['disc_dice', 'cup_dice']
    max_diffs = [{}, {}]
    min_diffs = [{}, {}]

    for i in range(2):
        diff = diffs[i]
        max_samples = heapq.nlargest(samples, range(len(diff)), diff.take)
        for max_id in max_samples:
            max_diffs[i][max_id] = diff[max_id]
        min_samples = heapq.nsmallest(samples, range(len(diff)), diff.take)
        for min_id in min_samples:
            min_diffs[i][min_id] = diff[min_id]

    # Save Part
    min_max_name = ['max', 'min']
    min_max_list = [max_diffs, min_diffs]
    for sample in range(samples):
        for i in range(2):
            for m in range(2):
                if grade == '':
                    task_dir = os.path.join('../data/experiment/diff/{}/{}'.format(dice_names[i], min_max_name[m]))
                else:
                    task_dir = os.path.join('../data/experiment/diff_{}/{}/{}'.
                    format(grade, dice_names[i], min_max_name[m]))
                if not os.path.exists(task_dir):
                    os.makedirs(task_dir)
                for index in min_max_list[m][i].keys():
                    diff_ = min_max_list[m][i][index]
                    img = (imgs[index] * 255).astype(np.uint8)
                    e_img = (e_imgs[index]* 255).astype(np.uint8)
                    pred = (preds[index] / (args.n_classes - 1) * 255).astype(np.uint8)
                    e_pred = (e_preds[index] / (args.n_classes - 1) * 255).astype(np.uint8)
                    gt = (gts[index]/ (args.n_classes - 1) * 255).astype(np.uint8)
                    gt = cv2.cvtColor(gt , cv2.COLOR_GRAY2RGB)
                    result = np.hstack((img, e_img, gt, pred, e_pred))
                    file_name = os.path.join(task_dir, '{}.jpg'.format(diff_))
                    plt.imsave(file_name, result)


    for i in range(2):
        if grade == '':
            out_dir = os.path.join('../data/experiment/diff/{}/'.format(dice_names[i]))
        else:
            out_dir = os.path.join('../data/experiment/diff_{}/{}/'
                .format(grade,dice_names[i]))
        f = open(os.path.join(out_dir, 'result.txt'), 'w')
        sigma = np.std(diffs[i])
        mean  = np.mean(diffs[i])
        f.write('{}: {}  +- {} \n'.format(dice_names[i], mean, sigma))
        f.close()
    return diffs

def get_p_value(dices, e_dices):
    z_list = []
    for i in range(2):
        dice_list = dices[i]
        e_dice_list = e_dices[i]
        t_val, p = ttest_rel(dice_list, e_dice_list)
        z_list.append(p)
    return z_list

def divide_score(args, imgs, e_imgs, dices, e_dices, preds, e_preds, gts):
    from model.backbone import resnet50_backbone
    model = resnet50_backbone(num_classes=3).cuda()
    model.load_state_dict(torch.load('../model/iqa/EyeQ_512_512/resnet50/resnet50_best.pt'))
    torch.set_grad_enabled(False)
    model.eval()
    N = len(imgs)
    good_imgs = []
    good_dices = [[], []]
    good_preds = []
    good_gts = []
    good_e_dices = [[], []]
    good_e_preds = []
    good_e_imgs = []
    bad_imgs = []
    bad_dices = [[], []]
    bad_preds = []
    bad_gts = []
    bad_e_dices = [[], []]
    bad_e_preds = []
    bad_e_imgs = []
    mid_imgs = []
    mid_dices = [[], []]
    mid_preds = []
    mid_gts = []
    mid_e_dices = [[], []]
    mid_e_preds = []
    mid_e_imgs = []

    for i in tqdm(range(N)):
        ori_img = imgs[i]
        img = cv2.resize(ori_img, (args.size, args.size))
        img = torch.from_numpy(img).cuda().permute(2, 0, 1)
        img = transforms.Normalize(args.mean,
                                 args.std)(img)
        img = torch.unsqueeze(img, dim=0)
        score = model(img)
        grade = torch.argmax(score[0], dim=0).detach().cpu().numpy()
        print(grade)
        if grade == 0:
            good_dices[0].append(dices[0][i])
            good_dices[1].append(dices[1][i])
            good_imgs.append(imgs[i])
            good_preds.append(preds[i])
            good_e_dices[0].append(e_dices[0][i])
            good_e_dices[1].append(e_dices[1][i])
            good_e_preds.append(e_preds[i])
            good_e_imgs.append(e_imgs[i])
            good_gts.append(gts[i])
        elif grade == 1:
            mid_dices[0].append(dices[0][i])
            mid_dices[1].append(dices[1][i])
            mid_imgs.append(imgs[i])
            mid_preds.append(preds[i])
            mid_e_dices[0].append(e_dices[0][i])
            mid_e_dices[1].append(e_dices[1][i])
            mid_e_preds.append(e_preds[i])
            mid_e_imgs.append(e_imgs[i])
            mid_gts.append(gts[i])
        else:
            bad_dices[0].append(dices[0][i])
            bad_dices[1].append(dices[1][i])
            bad_imgs.append(imgs[i])
            bad_preds.append(preds[i])
            bad_e_dices[0].append(e_dices[0][i])
            bad_e_dices[1].append(e_dices[1][i])
            bad_e_preds.append(e_preds[i])
            bad_e_imgs.append(e_imgs[i])
            bad_gts.append(gts[i])
    torch.set_grad_enabled(True)
    return [good_imgs, mid_imgs, bad_imgs], \
           [good_e_imgs, mid_e_imgs, bad_e_imgs], \
           [good_dices, mid_dices, bad_dices], \
           [good_e_dices, mid_e_dices, bad_e_dices], \
           [good_preds, mid_preds, bad_preds],\
            [good_e_preds, mid_e_preds, bad_e_preds], \
            [good_gts, mid_gts, bad_gts]


def dice_coefficient(binary_segmentation, binary_gt_label):
    '''
    Compute the Dice coefficient between two binary segmentation.
    Dice coefficient is defined as here: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Input:
        binary_segmentation: binary 2D numpy array representing the region of interest as segmented by the algorithm
        binary_gt_label: binary 2D numpy array representing the region of interest as provided in the database
    Output:
        dice_value: Dice coefficient between the segmentation and the ground truth
    '''

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=np.bool)

    # compute the intersection
    intersection = np.logical_and(binary_segmentation, binary_gt_label)

    # count the number of True pixels in the binary segmentation
    segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
    # same for the ground truth
    gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
    # same for the intersection
    intersection = float(np.sum(intersection.flatten()))

    # compute the Dice coefficient
    dice_value = 2 * intersection / (segmentation_pixels + gt_label_pixels)

    # return it
    return dice_value


def vertical_diameter(binary_segmentation):
    '''
    Get the vertical diameter from a binary segmentation.
    The vertical diameter is defined as the "fattest" area of the binary_segmentation parameter.
    Input:
        binary_segmentation: a boolean 2D numpy array representing a region of interest.
    Output:
        diameter: the vertical diameter of the structure, defined as the largest diameter between the upper and the lower interfaces
    '''

    # turn the variable to boolean, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool)

    # get the sum of the pixels in the vertical axis
    vertical_axis_diameter = np.sum(binary_segmentation, axis=0)

    # pick the maximum value
    diameter = np.max(vertical_axis_diameter)

    # return it
    return float(diameter)


def vertical_cup_to_disc_ratio(segmentation):
    '''
    Compute the vertical cup-to-disc ratio from a given labelling map.
    The vertical cup to disc ratio is defined as here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1722393/pdf/v082p01118.pdf
    Input:
        segmentation: binary 2D numpy array representing a segmentation, with 0: optic cup, 128: optic disc, 255: elsewhere.
    Output:
        cdr: vertical cup to disc ratio
    '''

    # compute the cup diameter
    cup_diameter = vertical_diameter(segmentation == 0)
    # compute the disc diameter
    disc_diameter = vertical_diameter(segmentation < 255)

    return cup_diameter / (disc_diameter + EPS)


def absolute_error(predicted, reference):
    '''
    Compute the absolute error between a predicted and a reference outcomes.
    Input:
        predicted: a float value representing a predicted outcome
        reference: a float value representing the reference outcome
    Output:
        abs_err: the absolute difference between predicted and reference
    '''

    return abs(predicted - reference)

def evaluate_binary_segmentation(segmentation, gt_label):
    '''
    Compute the evaluation metrics of the REFUGE challenge by comparing the segmentation with the ground truth
    Input:
        segmentation: binary 2D numpy array representing the segmentation, with 0: optic cup, 128: optic disc, 255: elsewhere.
        gt_label: binary 2D numpy array representing the ground truth annotation, with the same format
    Output:
        cup_dice: Dice coefficient for the optic cup
        disc_dice: Dice coefficient for the optic disc
        cdr: absolute error between the vertical cup to disc ratio as estimated from the segmentation vs. the gt_label, in pixels
    '''

    # compute the Dice coefficient for the optic cup
    cup_dice = dice_coefficient(segmentation==0, gt_label==0)
    cup_conf = confusion_matrix(y_pred= (gt_label==0).flatten(), y_true=(segmentation==0).flatten())
    if float(cup_conf[0, 0] + cup_conf[0, 1]) == 0:
        cup_spe = 0
    else:
        cup_spe = float(cup_conf[0, 0]) / float(cup_conf[0, 0] + cup_conf[0, 1])
    if float(cup_conf[1, 1] +cup_conf [1, 0]) == 0:
        cup_sen = 0
    else:
        cup_sen = float(cup_conf[1, 1]) / float(cup_conf[1, 1] +cup_conf [1, 0])
    cup_acc = (cup_sen + cup_spe) / 2
    # compute the Dice coefficient for the optic disc
    disc_dice = dice_coefficient(segmentation<255, gt_label<255)
    disc_conf = confusion_matrix(y_true=(gt_label<255).flatten(), y_pred=(segmentation<255).flatten())
    if float(disc_conf[0, 0] + disc_conf[0, 1]) == 0:
        disc_spe = 0
    else:
        disc_spe = float(disc_conf[0, 0]) / float(disc_conf[0, 0] + disc_conf[0, 1])
    if float(disc_conf[1, 1] + disc_conf[1, 0]) == 0:
        disc_sen = 0
    else:
        disc_sen = float(disc_conf[1, 1]) / float(disc_conf[1, 1] + disc_conf[1, 0])
    disc_acc = (disc_sen + disc_spe) / 2
    # compute the absolute error between the cup to disc ratio estimated from the segmentation vs. the gt label
    cdr = absolute_error(vertical_cup_to_disc_ratio(segmentation), vertical_cup_to_disc_ratio(gt_label))

    return cup_dice, disc_dice, cdr, cup_acc, disc_acc

def generate_table_of_results(image_filenames, segmentation_folder, gt_folder, is_training=False):
    '''
    Generates a table with image_filename, cup_dice, disc_dice and cdr values
    Input:
        image_filenames: a list of strings with the names of the images.
        segmentation_folder: a string representing the full path to the folder where the segmentation files are
        gt_folder: a string representing the full path to the folder where the ground truth annotation files are
        is_training: a boolean value indicating if the evaluation is performed on training data or not
    Output:
        image_filenames: same as the input parameter
        cup_dices: a numpy array with the same length than the image_filenames list, with the Dice coefficient for each optic cup
        disc_dices: a numpy array with the same length than the image_filenames list, with the Dice coefficient for each optic disc
        ae_cdrs: a numpy array with the same length than the image_filenames list, with the absolute error of the vertical cup to disc ratio
    '''

    # initialize an array for the Dice coefficients of the optic cups
    cup_dices = np.zeros(len(image_filenames), dtype=np.float)
    # initialize an array for the Dice coefficients of the optic discs
    disc_dices = np.zeros(len(image_filenames), dtype=np.float)
    # initialize an array for the absolute errors of the vertical cup to disc ratios
    ae_cdrs = np.zeros(len(image_filenames), dtype=np.float)
    cup_acc = np.zeros(len(image_filenames), dtype=np.float)
    disc_acc = np.zeros(len(image_filenames), dtype=np.float)

    p = multiprocessing.Pool(256)

    # iterate for each image filename

    # Improved with multi-processes
    for i in range(len(image_filenames)):
        result = p.apply_async(evaluate_one,  args=(segmentation_folder, is_training, image_filenames, gt_folder, i))
        cup_dices[i], disc_dices[i], ae_cdrs[i], cup_acc[i], disc_acc[i] = result.get()
    p.close()
    p.join()
     # read the segmentation
    # return the colums of the table
    return image_filenames, cup_dices, disc_dices, ae_cdrs, cup_acc, disc_acc

def evaluate_one(segmentation_folder, is_training, image_filenames, gt_folder, i):
    segmentation = imageio.imread(os.path.join(segmentation_folder, image_filenames[i]))
    if len(segmentation.shape) > 2:
        segmentation = segmentation[:, :, 0]
    # read the gt
    if is_training:
        gt_filename = os.path.join(gt_folder, 'Glaucoma', image_filenames[i])
        if os.path.exists(gt_filename):
            gt_label = imageio.imread(gt_filename)
        else:
            gt_filename = os.path.join(gt_folder, 'Non-Glaucoma', image_filenames[i])
            if os.path.exists(gt_filename):
                gt_label = imageio.imread(gt_filename)
            else:
                raise ValueError(
                    'Unable to find {} in your training folder. Make sure that you have the folder organized as provided in our website.'.format(
                        image_filenames[i]))
    else:
        gt_filename = os.path.join(gt_folder, image_filenames[i])
        if os.path.exists(gt_filename):
            gt_label = imageio.imread(gt_filename)
        else:
            raise ValueError(
                'Unable to find {} in your ground truth folder. If you are using training data, make sure to use the parameter is_training in True.'.format(
                    image_filenames[i]))

    # evaluate the results and assign to the corresponding row in the table
    return evaluate_binary_segmentation(segmentation,
                                                                                                    gt_label)

def get_mean_values_from_table(cup_dices, disc_dices, ae_cdrs, cup_accs, disc_accs):
    '''
    Compute the mean evaluation metrics for the segmentation task.
    Input:
        cup_dices: a numpy array with the same length than the image_filenames list, with the Dice coefficient for each optic cup
        disc_dices: a numpy array with the same length than the image_filenames list, with the Dice coefficient for each optic disc
        ae_cdrs: a numpy array with the same length than the image_filenames list, with the absolute error of the vertical cup to disc ratio
    Output:
        mean_cup_dice: the mean Dice coefficient for the optic cups
        mean_disc_dice: the mean Dice coefficient for the optic disc
        mae_cdr: the mean absolute error for the vertical cup to disc ratio
    '''

    # compute the mean values of each column
    mean_cup_dice = np.mean(cup_dices)
    mean_disc_dice = np.mean(disc_dices)
    mae_cdr = np.mean(ae_cdrs)
    cup_acc = np.mean(cup_accs)
    disc_acc = np.mean(disc_accs)
    return mean_cup_dice, mean_disc_dice, mae_cdr, cup_acc, disc_acc

def get_filenames(path_to_files, extension):
    '''
    Get all the files on a given folder with the given extension
    Input:
        path_to_files: string to a path where the files are
        [extension]: string representing the extension of the files
    Output:
        image_filenames: a list of strings with the filenames in the folder
    '''

    # initialize a list of image filenames
    image_filenames = []
    # add to this list only those filenames with the corresponding extension
    for file in os.listdir(path_to_files):
        if file.endswith('.' + extension):
            image_filenames = image_filenames + [ file ]

    return image_filenames

def evaluate_segmentation_results(segmentation_folder, gt_folder, output_path=None, export_table=False, is_training=False):
    '''
    Evaluate the segmentation results of a single submission
    Input:
        segmentation_folder: full path to the segmentation files
        gt_folder: full path to the ground truth files
        [output_path]: a folder where the results will be saved. If not provided, the results are not saved
        [export_table]: a boolean value indicating if the table will be exported or not
        [is_training]: a boolean value indicating if the evaluation is performed on training data or not
    Output:
        mean_cup_dice: the mean Dice coefficient for the optic cups
        mean_disc_dice: the mean Dice coefficient for the optic disc
        mae_cdr: the mean absolute error for the vertical cup to disc ratio
    '''

    # get all the image filenames
    image_filenames = get_filenames(segmentation_folder, 'bmp')
    if len(image_filenames)==0:
        print('** The segmentation folder does not include any bmp file. Check the files extension and resubmit your results.')
        raise ValueError()
    # create output path if it does not exist
    if not (output_path is None) and not (os.path.exists(output_path)):
        os.makedirs(output_path)

    # generate a table of results
    _, cup_dices, disc_dices, ae_cdrs, cup_accs, disc_accs = generate_table_of_results(image_filenames, segmentation_folder, gt_folder, is_training)
    # if we need to save the table
    if not(output_path is None) and (export_table):
        # initialize the table filename
        table_filename = os.path.join(output_path, 'evaluation_table_segmentation.csv')
        # save the table
        save_csv_segmentation_table(table_filename, image_filenames, cup_dices, disc_dices, ae_cdrs)

    # compute the mean values
    mean_cup_dice, mean_disc_dice, mae_cdr, mean_cup_acc, mean_disc_acc = get_mean_values_from_table(cup_dices, disc_dices, ae_cdrs, cup_accs, disc_accs)
    # print the results on screen
    print('Dice Optic Cup = {}\nDice Optic Disc = {}\nMAE CDR = {}\nCUP_ACC={}\nDISC_ACC={}'.format(str(mean_cup_dice), str(mean_disc_dice), str(mae_cdr), str(mean_cup_acc), str(mean_disc_acc)))
    # save the mean values in the output path
    if not(output_path is None):
        # initialize the output filename
        output_filename = os.path.join(output_path, 'evaluation_segmentation.csv')
        # save the results
        save_csv_mean_segmentation_performance(output_filename, mean_cup_dice, mean_disc_dice, mae_cdr, mean_cup_acc, mean_disc_acc)

    # return the average performance
    return mean_cup_dice, mean_disc_dice, mae_cdr, mean_cup_acc, mean_disc_acc

def save_csv_segmentation_table(table_filename, image_filenames, cup_dices, disc_dices, ae_cdrs):
    '''
    Save the table of segmentation results as a CSV file.
    Input:
        table_filename: a string with the full path and the table filename (with .csv extension)
        image_filenames: a list of strings with the names of the images
        cup_dices: a numpy array with the same length than the image_filenames list, with the Dice coefficient for each optic cup
        disc_dices: a numpy array with the same length than the image_filenames list, with the Dice coefficient for each optic disc
        ae_cdrs: a numpy array with the same length than the image_filenames list, with the absolute error of the vertical cup to disc ratio
    '''

    # write the data
    with open(table_filename, 'w') as csv_file:
        # initialize the writer
        table_writer = csv.writer(csv_file)
        # write the column names
        table_writer.writerow(['Filename', 'Cup-Dice', 'Disc-Dice', 'AE-CDR'])
        # write each row
        for i in range(len(image_filenames)):
            table_writer.writerow( [image_filenames[i], str(cup_dices[i]), str(disc_dices[i]), str(ae_cdrs[i])] )

def save_csv_mean_segmentation_performance(output_filename, mean_cup_dice, mean_disc_dice, mae_cdrs, mean_cup_acc, mean_disc_acc):
    '''
    Save a CSV file with the mean performance
    Input:
        output_filename: a string with the full path and the table filename (with .csv extension)
        mean_cup_dice: average Dice coefficient for the optic cups
        mean_disc_dice: average Dice coefficient for the optic discs
        mae_cdrs: mean absolute error of the vertical cup to disc ratios
    '''

    # write the data
    with open(output_filename, 'w') as csv_file:
        # initialize the writer
        table_writer = csv.writer(csv_file)
        # write the column names
        table_writer.writerow(['Cup-Dice', 'Disc-Dice', 'AE-CDR', 'Cup-Acc', 'Disc-Acc', ])
        # write each row
        table_writer.writerow( [ str(mean_cup_dice), str(mean_disc_dice), str(mae_cdrs),
                                 str(mean_cup_acc), str(mean_disc_acc)] )