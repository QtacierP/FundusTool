import torch
import torch.nn as nn
from torchvision.models import inception_v3, resnet101, Inception3, resnet50
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

def resnet101_backbone(num_classes, n_colors=3, regression=False):
    base_model = resnet101(pretrained=True)
    base_model.conv1 = nn.Conv2d(n_colors, 64, kernel_size=7, stride=2, padding=3,bias=False)
    num_ftrs = base_model.fc.in_features
    if regression:
        base_model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, 1))
    else:
        base_model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, num_classes))

    return base_model.cuda()

def resnet50_backbone(num_classes, n_colors=3, regression=False):
    base_model = resnet50(pretrained=True)
    base_model.conv1 = nn.Conv2d(n_colors, 64, kernel_size=7, stride=2, padding=3,bias=False)
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
    def __init__(self, n_colors, num_classes, regression=False, num_downs=5, norm_layer=nn.BatchNorm2d):
        super(UNet, self).__init__()
        if regression:
            num_classes = 1
        use_bias = norm_layer == nn.InstanceNorm2d
        self.num_downs = num_downs
        filters = n_colors
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
            tail = [nn.Conv2d(filters, 1, kernel_size=3, padding=1),
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

class UNet_V2(nn.Module):
    def __init__(self, n_colors, num_classes, regression=False, bilinear=False):
        super(UNet_V2, self).__init__()
        self.n_channels = n_colors
        self.n_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_colors, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)