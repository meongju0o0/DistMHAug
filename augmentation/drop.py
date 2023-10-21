import dgl


class MHMasking:
    def __init__(self, g, delta_g_e, delta_g_v, device):
        self.g = g
        self.delta_g_e = delta_g_e
        self.delta_g_v = delta_g_v
        self.device = device

    def __call__(self):
        self.local_aug_g = self._mh_edge_dropping()
        self.local_aug_g = self._mh_node_dropping()
        self.local_aug_g.to(self.device)

    def _mh_edge_dropping(self):
        drop_edge = dgl.transforms.DropEdge(self.delta_g_e)
        return drop_edge(self.local_aug_g)

    def _mh_node_dropping(self):
        drop_node = dgl.transforms.FeatMask(self.delta_g_v, node_feat_names=["features"])
        return drop_node(self.local_aug_g)
