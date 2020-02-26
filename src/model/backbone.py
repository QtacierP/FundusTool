import torch
import torch.nn as nn
from torchvision.models import inception_v3, resnet101, Inception3

def InceptionV3_backbone(num_classes):
    base_model = Inception3(num_classes=num_classes, aux_logits=False)
    num_ftrs = base_model.fc.in_features
    base_model.fc = nn.Linear(num_ftrs, num_classes)

    return base_model.cuda()

def resnet101_backbone(num_classes):
    base_model = resnet101()
    num_ftrs = base_model.fc.in_features
    base_model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return base_model.cuda()

