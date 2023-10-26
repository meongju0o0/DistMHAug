import torch as th
import dgl


class AugDataLoader:
    def __init__(self, g, train_nid, args, batch_size, shuffle=True, drop_last=False):
        self.g = g

        sampler = dgl.dataloading.NeighborSampler([int(fanout) for fanout in args.fan_out.split(",")])
        self.dataloader = dgl.dataloading.DistNodeDataLoader(
            g, train_nid, sampler, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

        self.src_nodes_list = []
        self.dst_nodes_list = []
        self.blocks = {"org_blocks": [], "prev_blocks": [], "cur_blocks": []}

        self.limit = 0
        self.idx = 0

    def __iter__(self, g):
        self.limit = 0
        self.idx = 0

        self.g = g

        for src_nodes, dst_nodes, blocks in self.dataloader:
            self.limit += 1

            self.src_nodes_list.append(src_nodes)
            self.dst_nodes_list.append(dst_nodes)

            print(src_nodes)

            self.blocks["org_blocks"].append(blocks)
            self.blocks["prev_blocks"].append(
                dgl.to_block(self.g, dst_nodes=dst_nodes, src_nodes=src_nodes, include_dst_in_src=True))
            self.blocks["cur_blocks"].append(
                dgl.to_block(self.g, dst_nodes=dst_nodes, src_nodes=src_nodes, include_dst_in_src=True))

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
