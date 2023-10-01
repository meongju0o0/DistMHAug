import dgl
import torch as th
from scipy.stats import truncnorm

from calc import log_normal
from drop import MHEdgeDropping, MHNodeDropping


def aggregate(g, agg_model):
    g = dgl.add_self_loop(g)
    s_vec = agg_model(g)
    return s_vec

def augment(org_g, delta_g_e_aug, delta_g_v_aug):
    aug_g = MHEdgeDropping(org_g, delta_g_e_aug)
    aug_g = MHNodeDropping(aug_g, delta_g_v_aug)
    return aug_g

def mh_algorithm(args, org_g, aug_g):
    delta_g_e = 1 - aug_g.num_edges() / org_g.num_edges()
    delta_g_e_aug = truncnorm.rvs(0, 1, loc=delta_g_e, sigma=args.sigma_delta_e)

    delta_g_v = 1 - aug_g.num_nodes() / org_g.num_nodes()
    delta_g_v_aug = truncnorm.rvs(0, 1, loc=delta_g_v, sigma=args.sigma_delta_v)

    aug_g_2 = augment(org_g, delta_g_e_aug, delta_g_v_aug)
