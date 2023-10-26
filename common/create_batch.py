import dgl


class AugDataLoader:
    def __init__(self, g, train_nid, args, batch_size, shuffle=True, drop_last=False):
        self.g = g
        fanout = [int(fanout) for fanout in args.fan_out.split(",")]

        samplers = [dgl.dataloading.NeighborSampler(fanout, prob=None),
                    dgl.dataloading.NeighborSampler(fanout, prob="prev_emask"),
                    dgl.dataloading.NeighborSampler(fanout, prob="cur_emask")]

        self.org_dataloader = dgl.dataloading.DistNodeDataLoader(
            g, train_nid, samplers[0], batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        self.prev_dataloader = dgl.dataloading.DistNodeDataLoader(
            g, train_nid, samplers[1], batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        self.cur_dataloader = dgl.dataloading.DistNodeDataLoader(
            g, train_nid, samplers[2], batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

        self.src_nodes_list = []
        self.dst_nodes_list = []
        self.blocks = {"org_blocks": [], "prev_blocks": [], "cur_blocks": []}

        self.limit = 0
        self.idx = 0

    def __iter__(self):
        self.limit = 0
        self.idx = 0

        # DistNodeDataLoader로는 여러개 block을 불러올 수 없음
        for src_nodes, dst_nodes, org_blocks, prev_blocks, cur_blocks in self.dataloader:
            self.limit += 1

            self.src_nodes_list.append(src_nodes)
            self.dst_nodes_list.append(dst_nodes)

            self.blocks["org_blocks"].append(org_blocks)
            self.blocks["prev_blocks"].append(prev_blocks)
            self.blocks["cur_blocks"].append(cur_blocks)

        return self

    def __next__(self):
        if self.idx < self.limit:
            self.idx += 1
            return (self.src_nodes_list[self.idx], self.src_nodes_list[self.idx],
                    self.blocks["org_blocks"][self.idx],
                    self.blocks["prev_blocks"][self.idx],
                    self.blocks["cur_blocks"][self.idx])
        else:
            raise StopIteration
