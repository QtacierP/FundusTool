import torch
import torch.nn as nn
from torchvision.models import inception_v3, resnet101, Inception3
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

def InceptionV3_backbone(num_classes, regression=False):
    base_model = Inception3(num_classes=num_classes, aux_logits=False)
    num_ftrs = base_model.fc.in_features
    if regression:
        base_model.fc = nn.Linear(num_ftrs, 1)
    else:
        base_model.fc = nn.Linear(num_ftrs, num_classes)
    return base_model.cuda()

def resnet101_backbone(num_classes, regression=False):
    base_model = resnet101(pretrained=True)
    num_ftrs = base_model.fc.in_features
    if regression:
        base_model.fc =  nn.Linear(num_ftrs, 1)
    else:
        base_model.fc = nn.Linear(num_ftrs, num_classes)

    return base_model.cuda()

class ResBlock(nn.Module):
    def __init__(self, in_features):
        super(ResBlock, self).__init__()

        conv_block = [  nn.Conv2d(in_features, in_features, 3, stride=1, padding=1),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_features, in_features, 3, stride=1, padding=1),
                        nn.InstanceNorm2d(in_features) ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResClassifier(nn.Module):
    def __init__(self, n_colors, n_classes, n_layers=4):
        super(ResClassifier, self).__init__()
        filters = 32
        in_features = n_colors
        model = []
        model += [nn.Conv2d(in_features, filters, kernel_size=3, stride=2, padding=1),
                      nn.BatchNorm2d(filters),
                      nn.ReLU(inplace=True)]

        for i in range(n_layers):
            in_features = filters
            model += [ResBlock(in_features=in_features)]
            filters *= 2
            model += [nn.Conv2d(in_features, filters, kernel_size=3, stride=2, padding=1),
                      nn.BatchNorm2d(filters),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        model += [nn.AdaptiveAvgPool2d((1, 1)),
                  nn.Flatten(),
                  nn.Linear(filters, n_classes)] # No sigmoid
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Classifier(nn.Module):
    def __init__(self, n_colors, n_classes, n_layers=4):
        super(Classifier, self).__init__()
        filters = 32
        in_features = n_colors
        model = []
        model += [nn.Conv2d(in_features, filters, kernel_size=3, stride=2, padding=1),
                      nn.BatchNorm2d(filters),
                      nn.ReLU(inplace=True)]

        for i in range(n_layers):
            in_features = filters
            filters *= 2
            model += [nn.Conv2d(in_features, filters, kernel_size=3, stride=2, padding=1),
                      nn.BatchNorm2d(filters),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        model += [nn.AdaptiveAvgPool2d((1, 1)),
                  nn.Flatten(),
                  nn.Linear(filters, n_classes)] # No sigmoid
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def simple_classifier(num_classes, regression=False):
    if regression:
        return ResClassifier(3, 1, n_layers=6).cuda()
    else:
        return ResClassifier(3, num_classes, n_layers=6).cuda()

def EfficientNet_backbone(num_classes, regression=False, name='efficientnet-b1'):
    if regression:
        num_classes = 1
    model = EfficientNet.from_pretrained(model_name=name, num_classes=num_classes)
    return model.cuda()

class UNet(nn.Module):
    # Simple UNet
    def __init__(self, num_classes, regression=False, num_downs=4, norm_layer=nn.BatchNorm2d):
        super(UNet, self).__init__()
        if regression:
            num_classes = 1
        use_bias = norm_layer == nn.InstanceNorm2d
        self.num_downs = num_downs
        filters = 3
        downs = []
        out_filters = 64
        for down in range(num_downs):
            downs.append([
                         nn.Conv2d(filters, out_filters, kernel_size=3, bias=use_bias, padding=1),
                         norm_layer(out_filters),
                         nn.ReLU(True),
                         nn.Conv2d(out_filters, out_filters, kernel_size=3, bias=use_bias, padding=1),
                         norm_layer(out_filters),
                         nn.ReLU(True),
                         nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1),
                         norm_layer(out_filters),
                         nn.ReLU(inplace=True)])
            filters = out_filters
            out_filters *= 2

        mid = []
        for _ in range(9):
            mid += [ResBlock(filters)]

        ups = []
        out_filters = filters // 2
        for up in range(num_downs):
            ups.append([nn.Conv2d(filters * 2, out_filters, kernel_size=3, bias=use_bias, padding=1),
                        norm_layer(out_filters),
                        nn.ReLU(True),
                        nn.Conv2d(out_filters, out_filters, kernel_size=3, bias=use_bias, padding=1),
                        norm_layer(out_filters),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(out_filters, out_filters,
                                          kernel_size=4, stride=2, padding=1),
                        norm_layer(out_filters),
                        nn.ReLU(inplace=True)])
            filters = out_filters
            out_filters //= 2
        if num_classes == 1:
            tail = [nn.Conv2d(filters, num_classes, kernel_size=3, padding=1),
                    nn.Sigmoid()]
        else:
            tail = [nn.Conv2d(filters, num_classes, kernel_size=3, padding=1)]

        self.downs = nn.ModuleList([nn.Sequential(*downsample) for downsample in downs])
        self.ups = nn.ModuleList([nn.Sequential(*upsample) for upsample in ups])
        self.tail = nn.Sequential(*tail)
        self.mid = nn.Sequential(*mid)

    def forward(self, x):
        downs = []
        for i in range(self.num_downs):
            x = self.downs[i](x)
            downs.append(x)
        x = self.mid(x)
        for i in range(self.num_downs):
            x = torch.cat((x, downs[- i - 1]), dim=1)
            x = self.ups[i](x)
        x = self.tail(x)
        return x
