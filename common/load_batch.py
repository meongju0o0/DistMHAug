import dgl


class AugDataLoader:
    def __init__(self, g, train_nid, args, batch_size, shuffle=False, drop_last=False, device=None):
        self.g = g
        self.drop_last = drop_last
        self.device = device

        fanout = [int(fanout) for fanout in args.fan_out.split(",")]

        self.samplers = [dgl.dataloading.NeighborSampler(fanout, prob=None),
                         dgl.dataloading.NeighborSampler(fanout, prob="prev_emask"),
                         dgl.dataloading.NeighborSampler(fanout, prob="cur_emask")]

        self.org_dataloader = dgl.dataloading.DistNodeDataLoader(
            g, train_nid, self.samplers[0],
            batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, device="cpu")

    def __iter__(self):
        return self._generator()

    def _generator(self):
        for step, (src_nodes, dst_nodes, blocks) in enumerate(self.org_dataloader):
            org_src_nodes = src_nodes
            org_dst_nodes = dst_nodes
            org_blocks = blocks

            prev_src_nodes = None
            prev_dst_nodes = None
            prev_blocks = None

            cur_src_nodes = None
            cur_dst_nodes = None
            cur_blocks = None

            prev_dataloader = dgl.dataloading.DistNodeDataLoader(
                self.g, dst_nodes, self.samplers[1],
                batch_size=dst_nodes.size(dim=0), shuffle=False, drop_last=self.drop_last, device="cpu")

            cur_dataloader = dgl.dataloading.DistNodeDataLoader(
                self.g, dst_nodes, self.samplers[2],
                batch_size=dst_nodes.size(dim=0), shuffle=False, drop_last=self.drop_last, device="cpu")

            for src_nodes, dst_nodes, blocks in prev_dataloader:
                prev_src_nodes = src_nodes
                prev_dst_nodes = dst_nodes
                prev_blocks = blocks

            for src_nodes, dst_nodes, blocks in cur_dataloader:
                cur_src_nodes = src_nodes
                cur_dst_nodes = dst_nodes
                cur_blocks = blocks

            for idx, org_block in enumerate(org_blocks):
                org_blocks[idx] = org_block.to(self.device)

            for idx, prev_block in enumerate(prev_blocks):
                prev_blocks[idx] = prev_block.to(self.device)

            for idx, cur_block in enumerate(cur_blocks):
                cur_blocks[idx] = cur_block.to(self.device)

            yield {"org": [org_src_nodes, org_dst_nodes, org_blocks],
                   "prev": [prev_src_nodes, prev_dst_nodes, prev_blocks],
                   "cur": [cur_src_nodes, cur_dst_nodes, cur_blocks]}
