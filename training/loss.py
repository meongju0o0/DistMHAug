import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

class HLoss(Loss):
    """
    Loss for Regularizing
    Penalizes unconfident predictions and sharpens predictions
    """

    def __init__(self):
        super(HLoss, self).__init__()

    @staticmethod
    def forward(pred, full=False):
        """
        Parameters
        ----------
        pred : torch.Tensor
            Predicted labels.
        full : bool
            Decide whether to calculate sum of the loss or mean of the loss

        Returns
        -------
        Calculated Loss
        """
        num_data = pred.shape[0]
        b = F.softmax(pred, dim=1) * F.log_softmax(pred, dim=1)
        if full:
            return -1.0 * b.sum(1)
        b = -1.0 * b.sum()
        b = b / num_data
        return b


class XeLoss(Loss):
    """
    Loss using Kullback-Leibler Divergence
    Encourages the consistency of predictions on two consecutive augmented samples G^(t), G^(t+1)
    """
    def __init__(self):
        super(XeLoss, self).__init__()

    @staticmethod
    def forward(pred, label):
        """
        Parameters
        ----------
        pred : torch.Tensor
            Predicted labels.
        label : torch.Tensor
            Ground-truth labels.

        Returns
        -------
        Calculated Loss
        """
        num_data = pred.shape[0]
        b = F.softmax(label, dim=1) * F.log_softmax(pred, dim=1) - F.softmax(label, dim=1) * F.log_softmax(label, dim=1)
        b = -1.0 * b.sum()
        b = b / num_data
        return b


class Jensen_Shannon(Loss):
    """
    Symmetrical KL Divergence
    """
    def __init__(self):
        super(Jensen_Shannon, self).__init__()

    @staticmethod
    def forward(pred, label):
        """
        Parameters
        ----------
        pred : torch.Tensor
            Predicted labels.
        label : torch.Tensor
            Ground-truth labels.

        Returns
        -------
        Calculated Loss
        """
        num_data = pred.shape[0]
        b = F.softmax(label, dim=1) * F.log_softmax(pred, dim=1) - F.softmax(label, dim=1) * F.log_softmax(label, dim=1)
        b += F.softmax(pred, dim=1) * F.log_softmax(label, dim=1) - F.softmax(pred, dim=1) * F.log_softmax(pred, dim=1)
        b = -0.5 * b.sum()
        b = b / num_data
        return b
