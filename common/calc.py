import torch as th


def log_normal(a, b, sigma):
    return -1 * th.pow(a - b, 2) / (2 * th.pow(sigma, 2))


def one_hot_encode(labels, num_classes):
    """
    Convert labels to one-hot encoded format.

    Parameters:
    - labels (torch.Tensor): 1D tensor of class indices.
    - num_classes (int): Total number of classes.

    Returns:
    - torch.Tensor: One-hot encoded labels.
    """
    one_hot = th.zeros(labels.size(0), num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot
