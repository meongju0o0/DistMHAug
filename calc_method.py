import torch as th
from scipy.stats import truncnorm
import torch_geometric.utils as u

def our_truncnorm(a, b, mu, sigma, x=None, mode='pdf'):
    a, b = (a - mu) / sigma, (b - mu) / sigma
    if mode=='pdf':
        return truncnorm.pdf(x, a, b, loc = mu, scale = sigma)
    elif mode=='rvs':
        return truncnorm.rvs(a, b, loc = mu, scale = sigma)


def aggregate(features, edge_index, agg_model, num_hop):
    n = features.shape[0]
    edge_index_w_sl = u.add_self_loops(edge_index, num_nodes=n)[0]
    s_vec = agg_model(features, edge_index_w_sl)
    return s_vec


def log_normal(a, b, sigma):
    return -1 * th.pow(a - b, 2) / (2 * th.pow(sigma, 2))  # /root2pi / sigma


def augment(args, org_edge_index, org_feature, delta_G_e, delta_G_v):
    m = org_edge_index.shape[1]
    num_edge_drop = int(m * delta_G_e)
    #######  flip_edge (A=1)  #######
    idx = th.randperm(m, device='cuda')[:m - num_edge_drop]
    aug_edge_index = org_edge_index[:, idx]
    #################################

    n = org_feature.shape[0]
    num_node_drop = int(n * delta_G_v)

    aug_feature = org_feature.clone()
    node_list = th.ones(n, 1, device='cuda')
    ##########  flip_feat  ##########
    idx = th.randperm(n, device='cuda')[:num_node_drop]
    aug_feature[idx] = 0
    node_list[idx] = 0

    if num_node_drop:
        aug_feature *= n / (n - num_node_drop)
    #################################
    return aug_edge_index, aug_feature, node_list