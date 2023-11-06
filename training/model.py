import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import numpy as np
import torch as th
import torch.nn as nn


class SAGEConvSUM(dglnn.SAGEConv):
    def __init__(self, in_feats, n_classes):
        super().__init__(in_feats, n_classes,
                         aggregator_type="mean", feat_drop=0, bias=False, norm=None, activation=None)

    def reset_parameters(self):
        """
        Reset weight parameters as a one
        """
        nn.init.ones_(self.fc_neigh.weight)

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = th.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            graph.srcdata["h"] = (
                self.fc_neigh(feat_src) if lin_before_mp else feat_src
            )
            graph.update_all(msg_fn, fn.sum("m", "neigh"))
            h_neigh = graph.dstdata["neigh"]
            if not lin_before_mp:
                h_neigh = self.fc_neigh(h_neigh)

        rst = self.fc_self(h_self) + h_neigh

        return rst


class SimpleAGG(nn.Module):
    """
    Simple Aggregation Model to Calculate ego-graph's changing rate

    Parameters
    ----------
    num_hop : int
        Depth of Aggregation
    """

    def __init__(self, num_hop, in_feats=1, n_classes=1, dropout=0):
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(num_hop):
            self.layers.append(SAGEConvSUM(in_feats, n_classes))

        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, x):
        """
        Forward function.

        Parameters
        ----------
        blocks : List[DGLBlock]
            Sampled blocks.
        x : DistTensor
            Feature data.
        """
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.dropout(h)
        return h


class DistSAGE(nn.Module):
    """
    SAGE model for distributed train and evaluation.

    Parameters
    ----------
    in_feats : int
        Feature dimension.
    n_hidden : int
        Hidden layer dimension.
    n_classes : int
        Number of classes.
    n_layers : int
        Number of layers.
    activation : callable
        Activation function.
    dropout : float
        Dropout value.
    """

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        for _ in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        """
        Forward function.

        Parameters
        ----------
        blocks : List[DGLBlock]
            Sampled blocks.
        x : DistTensor
            Feature data.
        """
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Distributed layer-wise inference with the GraphSAGE model on full
        neighbors.

        Parameters
        ----------
        g : DistGraph
            Input Graph for inference.
        x : DistTensor
            Node feature data of input graph.

        Returns
        -------
        DistTensor
            Inference results.
        """
        # Split nodes to each trainer.
        nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )

        for i, layer in enumerate(self.layers):
            # Create DistTensor to save forward results.
            if i == len(self.layers) - 1:
                out_dim = self.n_classes
                name = "h_last"
            else:
                out_dim = self.n_hidden
                name = "h"
            y = dgl.distributed.DistTensor(
                (g.num_nodes(), out_dim),
                th.float32,
                name,
                persistent=True,
            )
            print(f"|V|={g.num_nodes()}, inference batch size: {batch_size}")

            # `-1` indicates all inbound edges will be inlcuded, namely, full
            # neighbor sampling.
            sampler = dgl.dataloading.NeighborSampler([-1])
            dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                # Copy back to CPU as DistTensor requires data reside on CPU.
                y[output_nodes] = h.cpu()

            x = y
            # Synchronize trainers.
            g.barrier()
        return x
