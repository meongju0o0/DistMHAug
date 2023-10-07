import copy

import torch as th
import dgl


class MHDropping:
    def __init__(self, org_g):
        self.aug_g = copy.deepcopy(org_g)
        self.aug_g.to(th.device('cuda:0'))


class MHEdgeDropping(MHDropping):
    def __init__(self, org_g, delta_g_e):
        super().__init__(org_g)
        self.drop_edge = dgl.transforms.DropEdge(delta_g_e)

    def __call__(self):
        return self.drop_edge(self.aug_g)


class MHNodeDropping(MHDropping):
    def __init__(self, org_g, delta_g_v):
        super().__init__(org_g)
        self.drop_node = dgl.transforms.FeatMask(delta_g_v, node_feat_names=["features"])

    def __call__(self):
        return self.drop_node(self.aug_g)
