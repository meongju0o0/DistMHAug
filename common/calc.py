import torch as th


def log_normal(a, b, sigma):
    return -1 * th.pow(a - b, 2) / (2 * th.pow(sigma, 2))
