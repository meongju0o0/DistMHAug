import dgl


class AugDataLoader:
    def __init__(self, g, train_nid, args, batch_size, shuffle=False, drop_last=False):
        self.g = g
        self.drop_last = drop_last
        fanout = [int(fanout) for fanout in args.fan_out.split(",")]

        self.samplers = [dgl.dataloading.NeighborSampler(fanout, prob=None),
                         dgl.dataloading.NeighborSampler(fanout, prob="prev_emask"),
                         dgl.dataloading.NeighborSampler(fanout, prob="cur_emask")]

        self.org_dataloader = dgl.dataloading.DistNodeDataLoader(
            g, train_nid, self.samplers[0], batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

        self.src_nodes_list = {"org_src_nodes": [], "prev_src_nodes": [], "cur_src_nodes": []}
        self.dst_nodes_list = {"org_dst_nodes": [], "prev_dst_nodes": [], "cur_dst_nodes": []}
        self.blocks = {"org_blocks": [], "prev_blocks": [], "cur_blocks": []}

        self.limit = 0
        self.idx = 0


    def __iter__(self):
        for src_nodes, dst_nodes, blocks in self.org_dataloader:
            self.limit += 1
            self.src_nodes_list["org_src_nodes"].append(src_nodes)
            self.dst_nodes_list["org_dst_nodes"].append(dst_nodes)
            self.blocks["org_blocks"].append(blocks)

            prev_dataloader = dgl.dataloading.DistNodeDataLoader(
                self.g, src_nodes, self.samplers[1],
                batch_size=src_nodes.size(dim=0), shuffle=False, drop_last=self.drop_last)

            cur_dataloader = dgl.dataloading.DistNodeDataLoader(
                self.g, src_nodes, self.samplers[2],
                batch_size=src_nodes.size(dim=0), shuffle=False, drop_last=self.drop_last)

            for src_nodes, dst_nodes, blocks in prev_dataloader:
                self.src_nodes_list["prev_src_nodes"].append(src_nodes)
                self.dst_nodes_list["prev_dst_nodes"].append(dst_nodes)
                self.blocks["prev_blocks"].append(blocks)

            for src_nodes, dst_nodes, blocks in cur_dataloader:
                self.src_nodes_list["cur_src_nodes"].append(src_nodes)
                self.dst_nodes_list["cur_dst_nodes"].append(dst_nodes)
                self.blocks["cur_blocks"].append(blocks)

        return self


    def __next__(self):
        if self.idx < self.limit:
            self.idx += 1

            org_src_list = self.src_nodes_list["org_src_nodes"][self.idx]
            org_blocks = self.blocks["org_blocks"][self.idx]
            prev_src_list = self.src_nodes_list["prev_src_nodes"][self.idx]
            prev_blocks = self.blocks["prev_blocks"][self.idx]
            cur_src_list = self.src_nodes_list["cur_src_nodes"][self.idx]
            cur_blocks = self.blocks["cur_blocks"][self.idx]

            return {"org": [org_src_list, org_src_list, org_blocks],
                    "prev": [prev_src_list, prev_src_list, prev_blocks],
                    "cur": [cur_src_list, cur_src_list, cur_blocks]}
        else:
            raise StopIteration
