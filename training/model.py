import dgl
import dgl.nn.pytorch as dglnn
import numpy as np
import torch as th
import torch.nn as nn
import tqdm


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
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, aggregator_type="add", feat_drop=0, bias=False))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset weight parameters as a one
        """
        for layer in self.layers:
            nn.init.ones_(layer.weight)

    def forward(self, graph):
        """
        Forward function

        Parameters
        ----------
        graph : DGLGraph
            Graph to aggregate

        Returns
        -------
        Aggregated Value
        """
        h = graph.ndata["one"] # one: the feature to calculate changing rate
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)
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

    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
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

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
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
