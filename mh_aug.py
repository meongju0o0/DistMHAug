import random

import dgl
import torch as th

import numpy as np
from scipy.stats import truncnorm
from scipy.special import betaln

from augmentation.masking import MHMasking
from training.loss import HLoss
from training.model import SimpleAGG
from common.calc import log_normal


@th.no_grad()
def aggregate(block, agg_model, x):
    s_vec = agg_model(block, x)
    return s_vec


@th.no_grad()
def mh_aug(args, g, model, dataloader, device):
    # Get only local partitioned graph

    org_num_edges = g.local_partition.num_edges()
    org_num_nodes = g.local_partition.num_nodes()

    prev_num_edges = g.edata["prev_emask"].local_partition.sum()
    prev_num_nodes = g.ndata["prev_nmask"].local_partition.sum()

    delta_g_e = 1 - prev_num_edges / org_num_edges
    delta_g_e_aug = truncnorm.rvs(0, 1, loc=delta_g_e, scale=args.sigma_delta_e)

    delta_g_v = 1 - prev_num_nodes / org_num_nodes
    delta_g_v_aug = truncnorm.rvs(0, 1, loc=delta_g_v, scale=args.sigma_delta_v)

    MHMasking(g, delta_g_e_aug, delta_g_v_aug, device)

    agg_model = SimpleAGG(num_hop=2)
    h_loss = HLoss()

    model.eval()

    ones = g.ndata["ones"]

    num_seeds = 0
    num_inputs = 0
    batch_cnt = 0
    ent_sum = 0

    org_ego_sum = 0
    delta_g_e_sum_ = 0
    delta_g_aug_e_sum_ = 0
    delta_g_v_sum_ = 0
    delta_g_aug_v_sum_ = 0

    for step, src_blocks in enumerate(dataloader):
        batch_cnt += 1

        org = src_blocks["org"]
        org_input_nodes = org[0]
        org_blocks = org[2]

        prev = src_blocks["prev"]
        prev_input_nodes = prev[0]
        prev_blocks = prev[2]

        cur = src_blocks["cur"]
        cur_input_nodes = cur[0]
        cur_blocks = cur[2]

        # Slice feature and label.
        batch_inputs = g.ndata["features"][org_input_nodes]
        num_seeds += len(org_blocks[-1].dstdata[dgl.NID])
        num_inputs += len(org_blocks[0].srcdata[dgl.NID])

        # Move to target device.
        org_blocks = [block.to(device) for block in org_blocks]
        batch_inputs = batch_inputs.to(device)

        # Get prediction to calculate ent.
        batch_pred = model(org_blocks, batch_inputs)

        max_ent = h_loss(th.full((1, batch_pred.shape[1]), 1 / batch_pred.shape[1])).item()
        ent = h_loss(batch_pred.detach(), True) / max_ent
        ent_sum += ent

        batch_org_ego = aggregate(org_blocks, agg_model, g.ndata["org_nmask"][org_input_nodes])
        org_ego_sum += batch_org_ego

        delta_g_e_sum_ += 1 - (aggregate(prev_blocks, agg_model, ones[prev_input_nodes]) / batch_org_ego).squeeze(1)
        delta_g_aug_e_sum_ += 1 - (aggregate(cur_blocks, agg_model, ones[cur_input_nodes]) / batch_org_ego).squeeze(1)
        delta_g_v_sum_ += 1 - (aggregate(prev_blocks, agg_model, ones[prev_input_nodes]) / batch_org_ego).squeeze(1)
        delta_g_aug_v_sum_ += 1 - (aggregate(cur_blocks, agg_model, ones[cur_input_nodes]) / batch_org_ego).squeeze(1)

    ent_mean = ent_sum / batch_cnt

    delta_g_e_mean_ = delta_g_e_sum_ / batch_cnt
    delta_g_aug_e_mean_ = delta_g_aug_e_sum_ / batch_cnt
    delta_g_v_mean_ = delta_g_v_sum_ / batch_cnt
    delta_g_aug_v_mean_ = delta_g_aug_v_sum_ / batch_cnt

    p = (args.lam1_e * log_normal(delta_g_e_mean_, args.mu_e, args.a_e * ent_mean + args.b_e) +
         args.lam1_v * log_normal(delta_g_v_mean_, args.mu_v, args.a_v * ent_mean + args.b_v))
    p_aug = (args.lam1_e * log_normal(delta_g_aug_e_mean_, args.mu_e, args.a_e * ent_mean + args.b_e) +
             args.lam1_v * log_normal(delta_g_aug_v_mean_, args.mu_v, args.a_v * ent_mean + args.b_v))

    q = (np.log(truncnorm.pdf(delta_g_e, 0, 1, loc=delta_g_e_aug, scale=args.sigma_delta_e)) +
        args.lam2_e * betaln(org_num_edges - org_num_edges * delta_g_e + 1, org_num_edges * delta_g_e + 1) +
        np.log(truncnorm.pdf(delta_g_v, 0, 1, loc=delta_g_v_aug, scale=args.sigma_delta_v)) +
        args.lam2_v * betaln(org_num_nodes - org_num_nodes * delta_g_v + 1, org_num_nodes * delta_g_v + 1))
    q_aug = (np.log(truncnorm.pdf(delta_g_e_aug, 0, 1, loc=delta_g_e, scale=args.sigma_delta_e)) +
        args.lam2_e * betaln(org_num_edges - org_num_edges * delta_g_e_aug + 1, org_num_edges * delta_g_e_aug + 1) +
        np.log(truncnorm.pdf(delta_g_v_aug, 0, 1, loc=delta_g_v, scale=args.sigma_delta_v)) +
        args.lam2_v * betaln(org_num_nodes - org_num_nodes * delta_g_v_aug + 1, org_num_nodes * delta_g_v_aug + 1))

    acceptance = ((th.sum(p_aug) - th.sum(p)) - (q_aug - q))

    if np.log(random.random()) < acceptance:
        if delta_g_e + delta_g_v < delta_g_e_aug + delta_g_v_aug:
            return g, True
        else:
            return g, False
    else:
        return None
