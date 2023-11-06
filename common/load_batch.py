import dgl


class AugDataLoader:
    def __init__(self, g, train_nid, args, batch_size, shuffle=False, drop_last=False, device=None):
        self.g = g
        self.drop_last = drop_last
        self.device = device

        self.fanout = [int(fanout) for fanout in args.fan_out.split(",")]

        self.samplers = [dgl.dataloading.NeighborSampler(self.fanout, prob=None),
                         dgl.dataloading.NeighborSampler(self.fanout, prob="prev_emask"),
                         dgl.dataloading.NeighborSampler(self.fanout, prob="cur_emask")]

        self.org_dataloader = dgl.dataloading.DistNodeDataLoader(
            g, train_nid, self.samplers[0],
            batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, device="cpu")

    def __iter__(self):
        return self._generator()

    def _generator(self):
        for step, (src_nodes, dst_nodes, blocks) in enumerate(self.org_dataloader):
            org_src_nodes, org_dst_nodes, org_blocks = src_nodes, dst_nodes, blocks
            prev_src_nodes, prev_dst_nodes, prev_blocks = self.samplers[1].sample(self.g, dst_nodes)
            cur_src_nodes, cur_dst_nodes, cur_blocks = self.samplers[2].sample(self.g, dst_nodes)

            yield {"org": [org_src_nodes, org_dst_nodes, org_blocks],
                   "prev": [prev_src_nodes, prev_dst_nodes, prev_blocks],
                   "cur": [cur_src_nodes, cur_dst_nodes, cur_blocks]}
