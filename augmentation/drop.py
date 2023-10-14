import dgl


class MHDropping:
    def __init__(self, org_g, delta_g_e, delta_g_v, device):
        self.delta_g_e = delta_g_e
        self.delta_g_v = delta_g_v
        self.device = device
        self.aug_g = org_g.local_partition.cpu()  # Returns only local partition and copy to cpu

    def __call__(self):
        self.aug_g = self._mh_edge_dropping()
        self.aug_g = self._mh_node_dropping()
        self.aug_g.to(self.device)

    def _mh_edge_dropping(self):
        drop_edge = dgl.transforms.DropEdge(self.delta_g_e)
        return drop_edge(self.aug_g)

    def _mh_node_dropping(self):
        drop_node = dgl.transforms.FeatMask(self.delta_g_v, node_feat_names=["features"])
        return drop_node(self.aug_g)
