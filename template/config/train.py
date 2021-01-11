"""doc
# Train Config

This is the main configuration file used for training the approach.
"""
import os
from deeptech.core import Config, cli
from deeptech.model.module_from_json import Module
from deeptech.training.trainers import SupervisedTrainer
from deeptech.training.optimizers import smart_optimizer
from torch.optim import SGD

from ..data.dataset import FashionMNISTDataset
from ..training.loss import SparseCrossEntropyLossFromLogits


class FashionMNISTConfig(Config):
    def __init__(self, training_name, data_path, training_results_path):
        super().__init__(training_name, data_path, training_results_path)
        # Config of the data
        self.data_dataset = FashionMNISTDataset

        # Config of the model
        model_json = os.path.join(os.path.dirname(__file__), "..", "model", "mnist_model.json")
        self.model_model = lambda: Module.create_from_file(model_json, "MNISTModel", num_classes=10, logits=True)

        # Config for training
        self.training_loss = SparseCrossEntropyLossFromLogits
        self.training_optimizer = smart_optimizer(SGD)
        self.training_trainer = SupervisedTrainer
        self.training_epochs = 10
        self.training_batch_size = 32


# Run with parameters parsed from commandline.
# python -m deeptech.examples.mnist_simple --mode=train --input=Datasets --output=Results
if __name__ == "__main__":
    cli.run(FashionMNISTConfig)
