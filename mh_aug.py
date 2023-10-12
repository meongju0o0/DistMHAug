import random

import dgl
import torch as th
import numpy as np
from scipy.stats import truncnorm
from scipy.special import betaln

from augmentation.drop import MHEdgeDropping, MHNodeDropping
from training.loss import HLoss
from training.model import SimpleAGG
from common.calc import log_normal


def aggregate(g, agg_model):
    g = dgl.add_self_loop(g)
    s_vec = agg_model(g)
    return s_vec


def drop_node_edge(org_g, delta_g_e_aug, delta_g_v_aug, device):
    aug_g = MHEdgeDropping(org_g, delta_g_e_aug, device)
    aug_g = MHNodeDropping(aug_g, delta_g_v_aug, device)
    return aug_g


@th.no_grad()
def mh_aug(args, org_g, prev_aug_g, model, dataloader, device):
    # Get only local partitioned graph
    print(org_g)
    exit(100)

    delta_g_e = 1 - prev_aug_g.num_edges() / org_g.num_edges()
    delta_g_e_aug = truncnorm.rvs(0, 1, loc=delta_g_e, scale=args.sigma_delta_e)

    delta_g_v = 1 - prev_aug_g.num_nodes() / org_g.num_nodes()
    delta_g_v_aug = truncnorm.rvs(0, 1, loc=delta_g_v, scale=args.sigma_delta_v)

    cur_aug_g = drop_node_edge(org_g, delta_g_e_aug, delta_g_v_aug, device)

    agg_model = SimpleAGG
    h_loss = HLoss()

    model.eval()

    num_seeds = 0
    num_inputs = 0
    batch_cnt = 0
    ent_sum = 0

    for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
        batch_cnt += 1

        # Slice feature and label.
        batch_inputs = org_g.ndata["features"][input_nodes]
        batch_labels = org_g.ndata["labels"][seeds].long()
        num_seeds += len(blocks[-1].dstdata[dgl.NID])
        num_inputs += len(blocks[0].srcdata[dgl.NID])

        # Move to target device.
        blocks = [block.to(device) for block in blocks]
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        # Get prediction to calculate ent.
        batch_pred = model(blocks, batch_inputs)

        max_ent = h_loss(th.full((1, batch_pred.shape[1]), 1 / batch_pred.shape[1])).item()
        ent = h_loss(batch_pred.detach(), True) / max_ent
        ent_sum += ent

    ent_mean = ent_sum / batch_cnt
    org_ego = aggregate(org_g, agg_model)

    delta_g_e_ = 1 - (aggregate(cur_aug_g, agg_model) / org_ego).squeeze(1)
    delta_g_aug_e_ = 1 - (aggregate(cur_aug_g, agg_model) / org_ego).squeeze(1)
    delta_g_v_ = 1 - (aggregate(cur_aug_g, agg_model) / org_ego).squeeze(1)
    delta_g_aug_v_ = 1 - (aggregate(cur_aug_g, agg_model) / org_ego).squeeze(1)

    p = (args.lam1_e * log_normal(delta_g_e_, args.mu_e, args.a_e * ent_mean + args.b_e) +
         args.lam1_v * log_normal(delta_g_v_, args.mu_v, args.a_v * ent_mean + args.b_v))
    p_aug = (args.lam1_e * log_normal(delta_g_aug_e_, args.mu_e, args.a_e * ent_mean + args.b_e) +
             args.lam1_v * log_normal(delta_g_aug_v_, args.mu_v, args.a_v * ent_mean + args.b_v))

    q = (np.log(truncnorm.pdf(delta_g_e, 0, 1, loc=delta_g_e_aug, scale=args.sigma_delta_e)) +
         args.lam2_e * betaln(org_g.num_edges() - org_g.num_edges() * delta_g_e + 1,
                              org_g.num_edges() * delta_g_e + 1) +
         np.log(truncnorm.pdf(delta_g_v, 0, 1, loc=delta_g_v_aug, scale=args.sigma_delta_v)) +
         args.lam2_v * betaln(org_g.num_nodes() - org_g.num_nodes() * delta_g_v + 1, org_g.num_nodes() * delta_g_v + 1))
    q_aug = (np.log(truncnorm.pdf(delta_g_e_aug, 0, 1, loc=delta_g_e, scale=args.sigma_delta_e)) +
             args.lam2_e * betaln(org_g.num_edges() - org_g.num_edges() * delta_g_e_aug + 1,
                                  org_g.num_edges() * delta_g_e_aug + 1) +
             np.log(truncnorm.pdf(delta_g_v_aug, 0, 1, loc=delta_g_v, scale=args.sigma_delta_v)) +
             args.lam2_v * betaln(org_g.num_nodes() - org_g.num_nodes() * delta_g_v_aug + 1,
                                  org_g.num_nodes() * delta_g_v_aug + 1))

    acceptance = ((th.sum(p_aug) - th.sum(p)) - (q_aug - q))

    if np.log(random.random()) < acceptance:
        if delta_g_e + delta_g_v < delta_g_e_aug + delta_g_v_aug:
            return cur_aug_g, True
        else:
            return cur_aug_g, False
    else:
        return None
