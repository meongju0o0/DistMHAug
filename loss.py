import torch.nn as nn
import torch.nn.functional as F


class HLoss(nn.Module):
    """
    Loss for Regularizing
    Penalizes unconfident predictions and sharpens predictions

    Parameters
    ----------
    x : DistTensor
        Predicted Value

    Returns
    -------
    Loss
    """

    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x, full=False):
        num_data = x.shape[0]
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        if full:
            return -1.0 * b.sum(1)
        b = -1.0 * b.sum()
        b = b / num_data
        return b


class XeLoss(nn.Module):
    """
    Loss using Kullback-Leibler Divergence
    Encourages the consistency of predictions on two consecutive augmented samples G^(t), G^(t+1)

    Parameters
    ----------
    y : DistTensor
        Label
    x : DistTensor
        Predicted Value

    Returns
    -------
    Loss
    """
    def __init__(self):
        super(XeLoss, self).__init__()

    def forward(self, y, x):
        num_data = x.shape[0]
        b = F.softmax(y, dim=1) * F.log_softmax(x, dim=1) - F.softmax(y, dim=1) * F.log_softmax(y, dim=1)
        b = -1.0 * b.sum()
        b = b / num_data
        return b


class Jensen_Shannon(nn.Module):
    """
    Symmetrical KL Divergence

    Parameters
    ----------
    y : DistTensor
        Label
    x : DistTensor
        Predicted Value

    Returns
    -------
    Loss
    """
    def __init__(self):
        super(Jensen_Shannon, self).__init__()

    def forward(self, y, x):
        num_data = x.shape[0]
        b = F.softmax(y, dim=1) * F.log_softmax(x, dim=1) - F.softmax(y, dim=1) * F.log_softmax(y, dim=1)
        b += F.softmax(x, dim=1) * F.log_softmax(y, dim=1) - F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -0.5 * b.sum()
        b = b / num_data
        return b
