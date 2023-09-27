import torch.nn as nn
import torch.nn.functional as F


class HLoss(nn.Module):
    """
    Loss for Regularizing
    Penalizes unconfident predictions and sharpens predictions
    """

    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, pred, full=False):
        """
        Parameters
        ----------
        pred : DistTensor
            Predicted Value
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


class XeLoss(nn.Module):
    """
    Loss using Kullback-Leibler Divergence
    Encourages the consistency of predictions on two consecutive augmented samples G^(t), G^(t+1)
    """
    def __init__(self):
        super(XeLoss, self).__init__()

    def forward(self, pred, label):
        """
        Parameters
        ----------
        pred : DistTensor
            Predicted Value
        label : DistTensor
            Label

        Returns
        -------
        Calculated Loss
        """
        num_data = pred.shape[0]
        b = F.softmax(label, dim=1) * F.log_softmax(pred, dim=1) - F.softmax(label, dim=1) * F.log_softmax(label, dim=1)
        b = -1.0 * b.sum()
        b = b / num_data
        return b


class Jensen_Shannon(nn.Module):
    """
    Symmetrical KL Divergence
    """
    def __init__(self):
        super(Jensen_Shannon, self).__init__()

    def forward(self, pred, label):
        """
        Parameters
        ----------
        pred : DistTensor
            Predicted Value
        label : DistTensor
            Label

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
