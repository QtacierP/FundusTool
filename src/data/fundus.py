from data.common import AbstractDataLoader
from keras_preprocessing.image import ImageDataGenerator
import os

class MyDataLoader(AbstractDataLoader):
    def __init__(self, args):
        super(MyDataLoader, self).__init__(args)

    def load(self):
        # For fundus, we only extract the directory
        train_path = os.path.join(self.args.data_dir, self.args.task, 'train')
        val_path = os.path.join(self.args.data_dir, self.args.task, 'val')
        test_path = os.path.join(self.args.data_dir, self.args.task, 'test')
        return train_path, val_path, test_path



