
import torch

class AbstractModel():
    def __init__(self, args):
        self.args = args
        self._init_model()
        self.callbacks = []
        self._init_callbacks()

    def _init_model(self):
        print("[INFO] Initialize model...")
        pass

    def _init_callbacks(self):
        pass

    def train(self, train_dataloader, val_dataloader):
        pass

    def test(self, test_dataloader):
        pass

