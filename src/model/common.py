from keras.layers import *
from keras.models import Model
from sklearn.metrics import *
from keras.callbacks import Callback
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import model_from_json
from keras.utils import multi_gpu_model
import os
import tensorflow as tf
import keras.backend as K
from keras_contrib.layers.normalization.instancenormalization \
    import InstanceNormalization

def MyConv2d(x,
              filters,
              kernel_size,
              strides=1,
              dilated_rate=(1, 1),
              padding='same',
              activation='relu',
              use_bias=False,
              normalization='bn',
              name=None):
    x = Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      name=name, dilation_rate=dilated_rate)(x)
    if not use_bias:
        if normalization == 'bn':
            bn_axis = 3
            bn_name = None if name is None else name + '_bn'
            x = BatchNormalization(axis=bn_axis,
                                          scale=False,
                                          name=bn_name)(x)
        else:
            x = InstanceNormalization(name=name+'_in')(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        if activation == 'LeakyReLU':
            x = LeakyReLU(alpha=0.2)(x)
        else:
            x = Activation(activation, name=ac_name)(x)
    return x

class AbstractModel():
    def __init__(self, args):
        self.args = args
        self._init_model()
        self.callbacks = []
        self._init_callbacks()

    def _init_model(self):
        pass

    def _init_callbacks(self):
        pass

    def load(self):
        self.single_model = model_from_json(
            open(os.path.join(self.args.model_path, '{}_architecture.json'.format(self.args.model))).read())
        self.single_model.load_weights(os.path.join(self.args.model_path, '{}_best_weights.h5'.format(self.args.model)), by_name=True)
        if self.args.n_gpus > 1:
            self.model = multi_gpu_model(self.single_model, gpus=self.args.n_gpus)
        else:
            self.model = self.single_model
        print('[Load Model]')

    def save(self):
        if self.single_model is None:
            raise Exception("[Exception] You have to build the model first.")
        print("[INFO] Saving model...")
        for i in range(4):
            json_string = self.single_model.to_json()
            open(os.path.join(self.args.model_path, '{}_architecture.json'.format(self.args.model)), 'w').write(json_string)
        print("[INFO] Model saved")

    def train(self, train_data, val_data):
        pass

    def test(self, test_data):
        pass

def add_new_last_layer(base_model, nb_classes, fc_size=1024, name=''):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(fc_size, activation='relu')(x) #new FC layer, random init
    if nb_classes > 2:
        predictions = Dense(nb_classes, activation='softmax')(x)
    else:
        predictions = Dense(1, activation='sigmoid')(x)  # new sigmoid layer
    model = Model(input=base_model.input, output=predictions, name=name)
    return model


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)

class MyCheckPoint(ModelCheckpoint):
    def __init__(self, model,  filepath, opt = 'f1', monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False, train_data=None,  train_label=None,
                 mode='auto', period=1, multil_gpus=False, val_data=None, val_label=None):

        self.file_path = filepath
        self.mutil_gpus = multil_gpus
        self.single_model = model
        self.opt = opt

        super(MyCheckPoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                      mode, period)
        self.validation_data = []
        self.validation_data.append(val_data)
        self.validation_data.append(val_label)

        self.train_data = []
        self.train_data.append(train_data)
        self.train_data.append(train_label)


    def set_model(self, model):
        if self.mutil_gpus:
            self.model = self.single_model
        else:
            self.model = model

    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.best_one = 0
        self.val_recalls = []
        self.val_precisions = []
        self.val_aucs = []
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs=None):
        val_score = self.model.predict(self.validation_data[0])
        val_predict = (val_score >= 0.5).astype(np.int)
        val_targ = self.validation_data[1]

        _val_hamming = hamming_loss(self.validation_data[1], val_targ)

        val_score = val_score.flatten()
        val_predict = val_predict.flatten()
        val_targ = val_targ.flatten()

        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_auc = roc_auc_score(val_targ, val_score)
        _val_kappa = cohen_kappa_score(val_targ, val_predict)
        _val_confusion = confusion_matrix(val_targ, val_predict)


        train_score = self.model.predict(self.train_data[0])
        train_predict = (train_score >= 0.5).astype(np.int)
        train_targ = self.train_data[1]
        _train_hamming = hamming_loss(self.train_data[1], train_targ)

        train_score = train_score.flatten()
        train_predict = train_predict.flatten()
        train_targ = train_targ.flatten()


        _train_f1 = f1_score(train_targ, train_predict, average='micro')
        _train_recall = recall_score(train_targ, train_predict)
        _train_precision = precision_score(train_targ, train_predict)
        _train_auc = roc_auc_score(train_targ, train_score)
        _train_kappa = cohen_kappa_score(train_targ, train_predict)
        _train_confusion = confusion_matrix(train_targ, train_predict)
        print('Train Confusion',_train_confusion)
        print('Val Confusion', _val_confusion)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_aucs.append(_val_auc)
        self.val_kappas.append(_val_kappa)
        if self.opt == 'f1':
            one = _val_f1
        elif self.opt == 'auc':
            one = _val_auc
        else:
            one = _val_kappa
        print('[Train]: F1 Score : {}, Precision : {}, Recall : {},  AUC : {}, Kappa : {}, hamming loss {}'.
              format(_train_f1, _train_precision, _train_recall, _train_auc, _train_kappa, _train_hamming))
        print('[Val]: F1 Score : {}, Precision : {}, Recall : {},  AUC : {}, Kappa : {}, hamming loss {}'.
              format(_val_f1, _val_precision, _val_recall, _val_auc, _val_kappa, _val_hamming))
        if one > self.best_one:
            self.model.save_weights(self.file_path, overwrite=True)
            print('val_{} improved from {} to {}'.format(self.opt, self.best_one,one))
            self.best_one = one
        else:
            print("val {}: {}, but did not improve from the best {} {}".
                  format(self.opt, one, self.opt, self.best_one))
        return

def get_check_point(filepath, model, metrics='loss',
                    verbose=1,
                    mode='auto',
                    moniter='val_loss',
                    save_best_only=True, multi_gpus=False, val_data=None, val_label=None, train_data=None, train_label=None):

    if multi_gpus:
        return ParallelModelCheckpoint(
            filepath=filepath,
            model=model,
            verbose=verbose,
            monitor=moniter,
            mode=mode,
            save_best_only=save_best_only
        )
    else:
        return ModelCheckpoint(
            filepath=filepath,
            verbose=verbose,
            monitor=moniter,
            mode=mode,
            save_best_only=save_best_only
        )


def _cohen_kappa(y_true, y_pred, num_classes, weights=None, metrics_collections=None, updates_collections=None, name=None):
   kappa, update_op = tf.contrib.metrics.cohen_kappa(y_true, y_pred, num_classes, weights, metrics_collections, updates_collections, name)
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([update_op]):
      kappa = tf.identity(kappa)
   return kappa

def cohen_kappa(num_classes, weights=None, metrics_collections=None, updates_collections=None, name=None):
   def cohen_kappa_func(y_true, y_pred):
        y_true = K.argmax(y_true, axis=-1)
        y_pred= K.argmax(y_pred, axis=-1)
        return _cohen_kappa(y_true, y_pred, num_classes, weights, metrics_collections, updates_collections, name)
   return cohen_kappa_func