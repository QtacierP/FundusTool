from keras.layers import *
from keras.models import Model
import tensorflow as tf
from model.common import MyConv2d, add_new_last_layer
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import keras.backend as K
from keras.applications.inception_resnet_v2 import InceptionResNetV2

def UNetBN_backbone(inputs, num_class, name=''):
    # 288 x 288 x 32
    conv1 = MyConv2d(inputs, kernel_size=3, filters=32, name=name+'conv1_1')
    conv1 = MyConv2d(conv1, kernel_size=3, filters=32, name=name+'conv1_2')
    pool1 = MyConv2d(conv1, filters=32, strides=2, kernel_size=4,
                      activation='LeakyReLU', name=name+'pool1')

    # 144 x 144 x 64
    conv2 = MyConv2d(pool1, kernel_size=3, filters=64, name=name+'conv2_1')
    conv2 = MyConv2d(conv2, kernel_size=3, filters=64, name=name+'conv2_2')
    pool2 = MyConv2d(conv2, filters=64, strides=2, kernel_size=4,
                      activation='LeakyReLU', name=name+'pool2')

    # 72 x 72 x 128
    conv3 = MyConv2d(pool2, kernel_size=3, filters=128, name=name+'conv3_1')
    conv3 = MyConv2d(conv3, kernel_size=3, filters=128, name=name+'conv3_2')
    pool3 = MyConv2d(conv3, filters=128, strides=2, kernel_size=4,
                      activation='LeakyReLU', name=name+'pool3')

    # 36 x 36 x 256
    conv4 = MyConv2d(pool3, kernel_size=3, filters=256, name=name+'conv4_1')
    conv4 = MyConv2d(conv4, kernel_size=3, filters=256, name=name+'conv4_2')
    pool4 = MyConv2d(conv4, filters=256, strides=2, kernel_size=4,
                      activation='LeakyReLU', name=name+'pool4')

    # 36 x 36 x 512
    conv5 = MyConv2d(pool4, kernel_size=3, filters=512, name=name+'conv5_1')
    conv5 = MyConv2d(conv5, kernel_size=3, filters=512, name=name+'conv5_2')

    # Up-sampling: 72 x 72 x 256
    up1 = Conv2DTranspose(256, (3, 3), strides=2, activation='relu', padding='same', name=name+'up1')(conv5)
    up1 = concatenate([conv4, up1], name=name+'up1_c')
    conv6 = MyConv2d(up1, kernel_size=3, filters=256, name=name+'conv6_1')
    conv6 = MyConv2d(conv6, kernel_size=3, filters=256, name=name+'conv6_2')

    # Up-sampling: 144 x 144 x 128
    up2 = Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same', name=name+'up2')(conv6)
    up2 = concatenate([conv3, up2], name=name+'up2_c')
    conv7 = MyConv2d(up2, kernel_size=3, filters=128, name=name+'conv7_1')
    conv7 = MyConv2d(conv7, kernel_size=3, filters=128, name=name+'conv7_2')

    # Up-sampling: 288 x 288 x 64
    up3 = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same', name=name+'up3')(conv7)
    up3 = concatenate([conv2, up3], name=name+'up3_c')
    conv8 = MyConv2d(up3, kernel_size=3, filters=64, name=name+'conv8_1')
    conv8 = MyConv2d(conv8, kernel_size=3, filters=64, name=name+'conv8_2')

    # Up-sampling: 576 x 576 x 32
    up4 = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same', name=name+'up4')(conv8)
    up4 = concatenate([conv1, up4], name=name+'up4_c')
    conv9 = MyConv2d(up4, kernel_size=3, filters=32, name=name+'conv9_1')
    conv9 = MyConv2d(conv9, kernel_size=3, filters=32, name=name+'conv9_2')
    conv9 = Conv2D(num_class, (1, 1), padding='same', name=name+'conv9_3')(conv9)
    outputs = Activation('tanh', name=name + 'output')(conv9) # Regression problem
    model = Model(input=inputs, output=outputs, name=name)
    return model, outputs


def InceptionV3_backbone(inputs, num_class, name=''):

    base_model = InceptionV3(weights=None,
                             include_top=False,
                             input_shape=K.int_shape(inputs)[1: ])
    return add_new_last_layer(base_model, nb_classes=num_class, name=name)

def InceptionResNetV2_backbone(inputs, num_class, name=''):

    base_model = InceptionResNetV2(weights=None,
                             include_top=False,
                             input_shape=K.int_shape(inputs)[1:])
    return add_new_last_layer(base_model, nb_classes=num_class, name=name)

def Classifier(inputs, num_class, name=''):

    x = inputs
    filters = 32
    global_features = []
    for i in range(4):
        x = MyConv2d(x, filters=filters, kernel_size=3, padding='same',
                    name=name+'_conv_{}_1'.format(i))
        x = MyConv2d(x, filters=filters, kernel_size=3, padding='same',
                    name=name+'_conv_{}_2'.format(i))
        global_features.append(GlobalAveragePooling2D(name=name+'_gb_{}'.format(i))(x))
        x = MyConv2d(x, filters=filters, strides=2, kernel_size=4,
                         activation='LeakyReLU', name=name + '_pool_{}'.format(i))
        filters *= 2
    x = Concatenate(name=name+'_global_features')(global_features)
    outputs = Dense(num_class, activation='sigmoid', name=name+'_output')(x)
    model = Model(input=inputs, output=outputs, name=name)
    return model
