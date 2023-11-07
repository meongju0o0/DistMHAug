import argparse
import socket
import time

import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from training.evaluation import compute_acc, evaluate
from training.model import DistSAGE
from training.loss import HLoss, XeLoss, JensenShannon

from mh_aug import mh_aug
from common.set_graph import SetGraph
from common.load_batch import AugDataLoader
from common.config import CONFIG
from common.calc import one_hot_encode


def init(shape, dtype):
    return th.ones(size=shape, dtype=dtype)


def run(args, device, data):
    """
    Train and evaluate DistSAGE.

    Parameters
    ----------
    args : argparse.Args
        Arguments for train and evaluate.
    device : torch.Device
        Target device for train and evaluate.
    data : Packed Data
        This includes train/val/test IDs, feature dimension,
        number of classes, graph.
    """

    # Initial var declare and copy for augmentation training
    train_nid, val_nid, test_nid, in_feats, n_classes, g = data

    num_edges = g.num_edges()
    num_nodes = g.num_nodes()

    g.ndata["prev_features"] = g.ndata["features"][0:num_nodes]
    g.ndata["cur_features"] = g.ndata["features"][0:num_nodes]

    g.ndata["ones"] = dgl.distributed.DistTensor((num_nodes, 1), th.float32,
                                                 name='mpv', init_func=init)  # mpv: message passing value

    g.edata['org_emask'] = dgl.distributed.DistTensor((num_edges, 1), th.float32,
                                                      name='org_emask', init_func=init)
    g.edata['prev_emask'] = dgl.distributed.DistTensor((num_edges, 1), th.float32,
                                                       name='prev_emask', init_func=init)
    g.edata['cur_emask'] = dgl.distributed.DistTensor((num_edges, 1), th.float32,
                                                      name='cur_emask', init_func=init)

    g.ndata['org_nmask'] = dgl.distributed.DistTensor((num_nodes, 1), th.float32,
                                                      name='org_nmask', init_func=init)
    g.ndata['prev_nmask'] = dgl.distributed.DistTensor((num_nodes, 1), th.float32,
                                                       name='prev_nmask', init_func=init)
    g.ndata['cur_nmask'] = dgl.distributed.DistTensor((num_nodes, 1), th.float32,
                                                      name='cur_nmask', init_func=init)

    # Declare DataLoader
    dataloader = AugDataLoader(g, train_nid, args,
                               batch_size=args.batch_size, shuffle=False, drop_last=False, device=device)

    # Declare Training Methods
    model = DistSAGE(
        in_feats,
        args.num_hidden,
        n_classes,
        args.num_layers,
        F.relu,
        args.dropout,
    )
    model = model.to(device)
    if args.num_gpus == 0:
        model = th.nn.parallel.DistributedDataParallel(model)
    else:
        model = th.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)

    # Declare Loss Functions
    hard_xe_loss_op = nn.CrossEntropyLoss()
    soft_xe_loss_op = XeLoss()
    h_loss_op = HLoss()
    js_loss_op = JensenShannon()

    # Declare Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    # Training loop.
    batch_time = []  # time check per batch
    epoch = 0  # epoch count
    epoch_time = []  # time check per epoch
    test_acc = 0.0  # get accuracy per epoch

    while epoch < args.num_epochs:
        epoch += 1
        tic = time.time()
        # Various time statistics.
        sample_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        start = time.time()
        step_time = []

        with model.join():
            while True:
                print(f"{g.rank()}: Trying Metropolis-Hastings Augmentation...")
                cur_g, kl_loss_opt = mh_aug(args, g, model, dataloader, device)
                if kl_loss_opt is not None:
                    print("Metropolis-Hastings Augmentation Accepted!!!")
                    break

            for step, src_and_blocks in enumerate(dataloader):
                # input_nodes: src nodes, i.e. whole MFG's nodes
                # seeds: dst nodes
                # blocks: Message Flow Graph

                org = src_and_blocks["org"]
                org_input_nodes = org[0]
                org_dst_nodes = org[1]
                org_blocks = org[2]

                prev = src_and_blocks["prev"]
                prev_input_nodes = prev[0]
                prev_dst_nodes = prev[1]
                prev_blocks = prev[2]

                cur = src_and_blocks["cur"]
                cur_input_nodes = cur[0]
                cur_dst_nodes = cur[1]
                cur_blocks = cur[2]

                # Declare time variable to calculate computing time
                tic_step = time.time()
                sample_time += tic_step - start

                # Slice feature and label.
                org_batch_inputs = g.ndata["features"][org_input_nodes]
                prev_batch_inputs = g.ndata["features"][prev_input_nodes]
                cur_batch_inputs = g.ndata["features"][cur_input_nodes]

                org_batch_labels = g.ndata["labels"][org_dst_nodes].long()
                prev_batch_labels = g.ndata["labels"][prev_dst_nodes].long()

                num_seeds += len(org_blocks[-1].dstdata[dgl.NID])
                num_inputs += len(org_blocks[0].srcdata[dgl.NID])

                # Move to target device.
                org_blocks = [block.to(device) for block in org_blocks]
                prev_blocks = [block.to(device) for block in prev_blocks]
                cur_blocks = [block.to(device) for block in cur_blocks]

                org_batch_inputs = org_batch_inputs.to(device)
                prev_batch_inputs = prev_batch_inputs.to(device)
                cur_batch_inputs = cur_batch_inputs.to(device)

                org_batch_labels = org_batch_labels.to(device)
                prev_batch_labels = prev_batch_labels.to(device)

                # Compute loss and prediction.
                start = time.time()

                batch_pred = model(org_blocks, org_batch_inputs)
                batch_prev_pred = model(prev_blocks, prev_batch_inputs)
                batch_cur_pred = model(cur_blocks, cur_batch_inputs)

                forward_end = time.time()

                one_hot_prev_batch_labels = one_hot_encode(prev_batch_labels, n_classes)

                loss_XE = hard_xe_loss_op(batch_prev_pred, prev_batch_labels)
                if args.option_loss == 0:
                    loss_KL = soft_xe_loss_op(batch_prev_pred, one_hot_prev_batch_labels)
                else:
                    if kl_loss_opt:
                        loss_KL = js_loss_op(batch_prev_pred.detach(), batch_cur_pred)
                    else:
                        loss_KL = js_loss_op(batch_prev_pred, batch_cur_pred.detach())
                loss_H = h_loss_op(batch_pred)

                total_loss = loss_XE + args.kl * loss_KL + args.h * loss_H

                optimizer.zero_grad()
                total_loss.backward()

                # Calculate computing time
                compute_end = time.time()
                forward_time += forward_end - start
                backward_time += compute_end - forward_end

                optimizer.step()
                update_time += time.time() - compute_end

                step_t = time.time() - tic_step
                step_time.append(step_t)
                batch_time.append(len(org_blocks[-1].dstdata[dgl.NID]) / step_t)

                acc = compute_acc(batch_pred, org_batch_labels)
                gpu_mem_alloc = (
                    th.cuda.max_memory_allocated() / 1000000
                    if th.cuda.is_available()
                    else 0
                )

                sample_speed = np.mean(batch_time[-args.log_every:])
                mean_step_time = np.mean(step_time[-args.log_every:])

                print(
                    f"Part {g.rank()} | Epoch {epoch:05d} | Step {step:05d}"
                    f" | Loss {total_loss.item():.4f} | Train Acc {acc:.4f}"
                    f" | Speed (samples/sec) {sample_speed:.4f}"
                    f" | GPU {gpu_mem_alloc:.1f} MB | "
                    f"Mean step time {mean_step_time:.3f} s"
                )

                start = time.time()

        toc = time.time()
        print(
            f"Part {g.rank()}, Epoch Time(s): {toc - tic:.4f}, "
            f"sample+data_copy: {sample_time:.4f}, forward: {forward_time:.4f},"
            f" backward: {backward_time:.4f}, update: {update_time:.4f}, "
            f"#seeds: {num_seeds}, #inputs: {num_inputs}"
        )
        epoch_time.append(toc - tic)

        if epoch % args.eval_every == 0 or epoch == args.num_epochs:
            start = time.time()
            val_acc, test_acc = evaluate(
                model.module,
                g,
                g.ndata["features"],
                g.ndata["labels"],
                val_nid,
                test_nid,
                args.batch_size_eval,
                device,
            )
            print(
                f"Part {g.rank()}, Val Acc {val_acc:.4f}, "
                f"Test Acc {test_acc:.4f}, time: {time.time() - start:.4f}"
                )

    return epoch_time, test_acc


def main(args):
    """
    Main function.
    """
    host_name = socket.gethostname()
    print(f"{host_name}: Initializing DistDGL.")
    dgl.distributed.initialize(args.ip_config)
    print(f"{host_name}: Initializing PyTorch process group.")
    th.distributed.init_process_group(backend=args.backend)
    print(f"{host_name}: Initializing DistGraph.")
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    print(f"Rank of {host_name}: {g.rank()}")

    if args.num_gpus == 0:
        device = th.device("cpu")
    else:
        dev_id = g.rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))

    # Get data.
    data = SetGraph(g, args).__call__()

    # Train and evaluate.
    epoch_time, test_acc = run(args, device, data)

    print(
        f"Summary of node classification(GraphSAGE): GraphName "
        f"{args.graph_name} | TrainEpochTime(sum) {np.sum(epoch_time):.4f} "
        f"| TestAccuracy {test_acc:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed GraphSAGE")
    parser.add_argument("--graph_name", type=str,
        help="graph name")
    parser.add_argument("--ip_config", type=str,
        help="The file for IP configuration")
    parser.add_argument("--part_config", type=str,
        help="The path to the partition config file")
    parser.add_argument("--n_classes", type=int, default=0,
        help="the number of classes")
    parser.add_argument("--backend", type=str, default="gloo",
        help="pytorch distributed backend")
    parser.add_argument("--num_gpus", type=int, default=2,
        help="the number of GPU device. Use 0 for CPU training")

    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,25")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--decay", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--option_loss", type=int, default=0)
    parser.add_argument("--local_rank", type=int, help="get rank of the process")
    parser.add_argument("--pad-data", default=False, action="store_true",
        help="Pad train nid to the same length across machine, to ensure num of batches to be the same.")
    args = parser.parse_args()

    for key, value in CONFIG.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    print(f"Arguments: {args}")
    main(args)
