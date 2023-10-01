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

def mh_algorithm(args, org_g, prev_aug_g, model, loss_fcn, dataloader, device):
    delta_g_e = 1 - prev_aug_g.num_edges() / org_g.num_edges()
    delta_g_e_aug = truncnorm.rvs(0, 1, loc=delta_g_e, sigma=args.sigma_delta_e)

    delta_g_v = 1 - prev_aug_g.num_nodes() / org_g.num_nodes()
    delta_g_v_aug = truncnorm.rvs(0, 1, loc=delta_g_v, sigma=args.sigma_delta_v)

    cur_aug_g = augment(org_g, delta_g_e_aug, delta_g_v_aug)

    model.eval()

    num_seeds = 0
    num_inputs = 0
    step_cnt = 0
    loss_sum = 0

    with th.no_grad():
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            step_cnt += 1
            # Slice feature and label.
            batch_inputs = org_g.ndata["features"][input_nodes]
            batch_labels = org_g.ndata["labels"][seeds].long()
            num_seeds += len(blocks[-1].dstdata[dgl.NID])
            num_inputs += len(blocks[0].srcdata[dgl.NID])

            # Move to target device.
            blocks = [block.to(device) for block in blocks]
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            # Compute loss and prediction.
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            loss_sum += loss

        loss_mean = loss_sum / step_cnt
