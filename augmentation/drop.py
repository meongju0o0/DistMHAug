import torch as th
import dgl


class MHMasking:
    def __init__(self, g, delta_g_e, delta_g_v, device):
        self.g = g
        self.num_nodes = g.num_nodes()
        self.num_edges = g.num_edges()
        self.delta_g_e = delta_g_e
        self.delta_g_v = delta_g_v
        self.device = device

    def __call__(self):
        self._mh_edge_masking()
        self._mh_node_masking()

    def _mh_edge_masking(self):
        num_edge_drop = self.num_edges - int(self.num_edges * self.delta_g_e)
        masking_eids = th.randperm(self.num_edges, device=self.device)[:num_edge_drop]
        self.g.ndata["org_emask"][masking_eids] = 0
        self.g.ndata["cur_emask"] = self.g.ndata["org_emask"][masking_eids]
        self.g.ndata["org_emask"][:] = 1

    def _mh_node_masking(self):
        num_node_drop = int(self.num_nodes * self.delta_g_v)
        masking_nids = th.randperm(self.num_nodes, device=self.device)[:num_node_drop]
        self.g.edata["org_nmask"][masking_nids] = 0
        self.g.ndata["cur_nmask"] = self.g.ndata["org_nmask"][masking_nids]
        self.g.ndata["org_nmask"][:] = 1
