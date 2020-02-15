from model.common import AbstractModel
from keras.layers import Input
from model.common import add_new_last_layer, get_check_point, TensorBoard
from keras.utils import multi_gpu_model
from keras_preprocessing.image import ImageDataGenerator
from utils import normalize
import os
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from model.common import cohen_kappa
from model.backbone import InceptionResNetV2_backbone, InceptionV3_backbone




class MyModel(AbstractModel):
    def __init__(self, args):
        super(MyModel, self).__init__(args)

    def _init_model(self):
        inputs = Input((self.args.size, self.args.size, self.args.n_colors))
        if self.args.model == 'InceptionV3':
            self.single_model = InceptionV3_backbone(inputs, num_class=self.args.n_classes)
        elif self.args.model == 'InceptionResNetV2':
            self.single_model = InceptionResNetV2_backbone(inputs, num_class=self.args.n_classes)
        else:
            raise NotImplementedError('{} model hasn\'nt been implemented'
                                      .format(self.args.model))
        if self.args.n_gpus > 1:
            self.model = multi_gpu_model(self.single_model,
                                         gpus=self.args.n_gpus)
        else:
            self.model = self.single_model
        self.model.compile(optimizer=Adam(lr=self.args.lr),
                           loss=categorical_crossentropy,
                           metrics=[categorical_accuracy, cohen_kappa(num_classes=self.args.n_classes)])

    def _init_callbacks(self):
        self.callbacks.append(
            get_check_point(
                filepath=os.path.join(self.args.model_path,
                                      '{}_best_weights.h5'.format(self.args.model)),
                metrics=self.args.metric,
                verbose=1,
                model=self.single_model,
                moniter=self.args.metric,
                multi_gpus=self.args.n_gpus,
                mode='auto',
                save_best_only=True,
            )
        )
        self.callbacks.append(TensorBoard(
                log_dir=self.args.checkpoint,
                write_images=True,
                write_graph=True,
            ))

    def train(self, train_data, val_data):
        train_gen = ImageDataGenerator(preprocessing_function=normalize, vertical_flip=True,
                            horizontal_flip=True)
        val_gen = ImageDataGenerator(preprocessing_function=normalize)
        if isinstance(train_data, str): # Load data from dir
            train_load_data_func = train_gen.flow_from_directory\
                (train_data, target_size=(self.args.size, self.args.size),
                 batch_size=self.args.batch_size)
            val_load_data_func = val_gen.flow_from_directory\
                (val_data, target_size=(self.args.size, self.args.size),
                 batch_size=self.args.batch_size)
        else:
            train_load_data_func = train_gen.flow\
                (x=train_data[0], y=train_data[1],
                 batch_size=self.args.batch_size)
            val_load_data_func = val_gen.flow\
                (x=val_data[0], y=val_data[1],
                 batch_size=self.args.batch_size)

        self.model.fit_generator(
            generator=train_load_data_func,
            epochs=self.args.epochs,
            verbose=1,
            validation_data=val_load_data_func,
            callbacks=self.callbacks
        )
        self.single_model.save_weights(os.path.join(self.args.model_path, '{}_last_weights.h5'.format(self.args.model)))

    def test(self, data_dir):
        gen = ImageDataGenerator(preprocessing_function=normalize)
        results = self.model.evaluate_generator(generator=gen.flow_from_directory(data_dir, target_size=(self.args.size, self.args.size)))
        print(results)