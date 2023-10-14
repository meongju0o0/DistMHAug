import copy

import torch as th
import dgl

def mh_drop(org_g, delta_g_e, delta_g_v, aug_g, device):
    temp = org_g.local_partition.cpu()  # Returns only local partition to cpu
    drop_edge = dgl.transforms.DropEdge(delta_g_e)
    drop_node = dgl.transforms.FeatMask(delta_g_v, node_feat_names=["features"])

    temp = drop_edge(temp)
    temp = drop_node(temp)

    aug_g.ndata = temp.ndata
    aug_g.edata = temp.edata


class MHDropping:
    def __init__(self, g, delta_g_e, delta_g_v):
        self.g = g.local_partition.cpu() # Returns only local partition and copy to cpu
        self.drop_edge = dgl.transforms.DropEdge(delta_g_e)
        self.drop_node = dgl.transforms.FeatMask(delta_g_v, node_feat_names=["features"])


class MHEdgeDropping(MHDropping):
    def __init__(self, org_g, delta_g_e, delta_g_v):
        super().__init__(org_g, delta_g_e, delta_g_v)
        self.drop_edge = dgl.transforms.DropEdge(delta_g_e)

    def __call__(self):
        self.drop_edge(self.g)
        return self.drop_edge(self.g)


class MHNodeDropping(MHDropping):
    def __init__(self, org_g, delta_g_e, delta_g_v):
        super().__init__(org_g, delta_g_e, delta_g_v)
        self.drop_node = dgl.transforms.FeatMask(delta_g_v, node_feat_names=["features"])

    def __call__(self):
        return self.drop_node(self.g)
