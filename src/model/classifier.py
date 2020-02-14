from model.common import AbstractModel
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from model.common import add_new_last_layer, get_check_point, TensorBoard
from keras.utils import multi_gpu_model
from keras_preprocessing.image import ImageDataGenerator
from utils import normalize
import os
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from model.common import cohen_kappa_loss




class MyModel(AbstractModel):
    def __init__(self, args):
        super(MyModel, self).__init__(args)

    def _init_model(self):
        if self.args.model == 'InceptionV3':
            base_model = InceptionV3(weights=None,
                        include_top=False,
                        input_shape=(self.args.size, self.args.size, self.args.n_colors))
            self.single_model = add_new_last_layer(base_model, nb_classes=self.args.n_classes)
            if self.args.n_gpus > 1:
                self.model = multi_gpu_model(self.single_model,
                                             gpus=self.args.n_gpus)
            else:
                self.model = self.single_model
            self.model.compile(optimizer=Adam(lr=self.args.lr),
                               loss=categorical_crossentropy,
                               metrics=[categorical_accuracy, cohen_kappa_loss(num_classes=self.args.n_classes)])

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

    def train(self, data_dir):
        train_gen = ImageDataGenerator(preprocessing_function=normalize, vertical_flip=True,
                            horizontal_flip=True)
        val_gen = ImageDataGenerator(preprocessing_function=normalize)
        self.model.fit_generator(
            generator=train_gen.flow_from_directory(os.path.join(data_dir, 'train'), target_size=(self.args.size, self.args.size)),
            epochs=self.args.epochs,
            verbose=1,
            validation_data=val_gen.flow_from_directory(os.path.join(data_dir, 'val'), target_size=(self.args.size, self.args.size)),
            callbacks=self.callbacks
        )
        self.single_model.save_weights(os.path.join(self.args.model_path, '{}_last_weights.h5'.format(self.args.model)))

    def test(self, data_dir):
        gen = ImageDataGenerator(preprocessing_function=normalize)
        results = self.model.evaluate_generator(generator=gen.flow_from_directory(data_dir, target_size=(self.args.size, self.args.size)))
        print(results)