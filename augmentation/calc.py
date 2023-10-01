import torch as th
import dgl
from scipy.stats import truncnorm


def log_normal(a, b, sigma):
    return -1 * th.pow(a - b, 2) / (2 * th.pow(sigma, 2))

def our_truncnorm(a, b, mu, sigma, x=None, mode='pdf'):
    a, b = (a - mu) / sigma, (b - mu) / sigma
    if mode=='pdf':
        return truncnorm.pdf(x, a, b, loc = mu, scale = sigma)
    elif mode=='rvs':
        return truncnorm.rvs(a, b, loc = mu, scale = sigma)
