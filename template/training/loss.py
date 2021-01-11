"""doc
# Main Loss

This file contains the main loss with which to train the network.

As a simple example here the implementation from deeptech.training.losses.classification is copied.
"""
from torch.nn import Module
from torch import Tensor
from torch.nn import CrossEntropyLoss


class SparseCrossEntropyLossFromLogits(Module):
    def __init__(self, model=None, reduction: str = "mean"):
        """
        Compute a sparse cross entropy.
        
        This means that the preds are logits and the targets are not one hot encoded.
        
        :param reduction: Specifies the reduction to apply to the output: `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Default: 'mean'.
        """
        super().__init__()
        self.loss_fun = CrossEntropyLoss(reduction=reduction)
        
    def forward(self, y_pred, y_true):
        """
        Compute the sparse cross entropy assuming y_pred to be logits.
        
        :param y_pred: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        :param y_true: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        """
        if not isinstance(y_true, Tensor):
            y_true = y_true.class_id
        if not isinstance(y_pred, Tensor):
            y_pred = y_pred.class_id
        y_true = y_true.long()
        if len(y_true.shape) == len(y_pred.shape) and y_true.shape[1] == 1:
            y_true = y_true[:, 0]
        return self.loss_fun(y_pred, y_true)
