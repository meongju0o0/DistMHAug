import copy

import torch as th
from scipy.stats import truncnorm
import dgl


def log_normal(a, b, sigma):
    return -1 * th.pow(a - b, 2) / (2 * th.pow(sigma, 2))


def our_truncnorm(a, b, mu, sigma, x=None, mode='pdf'):
    a, b = (a - mu) / sigma, (b - mu) / sigma
    if mode=='pdf':
        return truncnorm.pdf(x, a, b, loc = mu, scale = sigma)
    elif mode=='rvs':
        return truncnorm.rvs(a, b, loc = mu, scale = sigma)


def aggregate(g, agg_model):
    g = dgl.add_self_loop(g)
    s_vec = agg_model(g)
    return s_vec


def augment(org_g, delta_G_e, delta_G_v):
    aug_g = copy.deepcopy(org_g)

    drop_edge = dgl.transforms.DropEdge(delta_G_e)
    drop_node = dgl.transforms.FeatMask(delta_G_v, node_feat_names=["features"])

    aug_g = drop_edge(aug_g)
    aug_g = drop_node(aug_g)

    return aug_g
