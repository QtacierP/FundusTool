from data import AbstractDataLoader
from keras_preprocessing.image import ImageDataGenerator

class MyDataLoader(AbstractDataLoader):
    def __init__(self, args):
        super(MyDataLoader, self).__init__(args)



