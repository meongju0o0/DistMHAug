import dgl
from dgl import backend as F
from dgl.base import NID, EID
from dgl import DGLGraph


class MyNeighborSampler(dgl.dataloading.NeighborSampler):
    def __init__(self, fanout, prob=(None,None,None)):
        super().__init__(fanout)
        self.prob = prob

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        org_blocks = []
        prev_blocks = []
        cur_blocks = []

        if self.fused:
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                            cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    org_block = g.sample_neighbors_fused(seed_nodes, fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob[0],
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping)
                    prev_block = g.sample_neighbors_fused(seed_nodes, fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob[1],
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping)
                    cur_block = g.sample_neighbors_fused(seed_nodes, fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob[2],
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping)

                    seed_nodes = org_block.srcdata[NID]
                    org_blocks.insert(0, org_block)
                    prev_blocks.insert(0, prev_block)
                    cur_blocks.insert(0, cur_block)
                    print("*************")
                    print(seed_nodes)
                    print("*************")
                    print(output_nodes)
                    print("*************")
                    print(org_blocks)
                    print("*************")

                return seed_nodes, output_nodes, org_blocks, prev_blocks, cur_blocks

        else:
            for fanout in reversed(self.fanouts):
                org_frontier = g.sample_neighbors(seed_nodes, fanout,
                    edge_dir=self.edge_dir,
                    prob=self.prob[0],
                    replace=self.replace,
                    output_device=self.output_device,
                    exclude_edges=exclude_eids)
                prev_frontier = g.sample_neighbors(seed_nodes, fanout,
                    edge_dir=self.edge_dir,
                    prob=self.prob[1],
                    replace=self.replace,
                    output_device=self.output_device,
                    exclude_edges=exclude_eids)
                cur_frontier = g.sample_neighbors(seed_nodes, fanout,
                    edge_dir=self.edge_dir,
                    prob=self.prob[2],
                    replace=self.replace,
                    output_device=self.output_device,
                    exclude_edges=exclude_eids)

                org_eid = org_frontier.edata[EID]
                org_block = dgl.to_block(org_frontier, seed_nodes)
                org_block.edata[EID] = org_eid
                seed_nodes = org_block.srcdata[NID]
                org_blocks.insert(0, org_block)

                prev_eid = prev_frontier.edata[EID]
                prev_block = dgl.to_block(prev_frontier, seed_nodes)
                prev_block.edata[EID] = prev_eid
                seed_nodes = prev_block.srcdata[NID]
                prev_blocks.insert(0, prev_block)

                cur_eid = cur_frontier.edata[EID]
                cur_block = dgl.to_block(cur_frontier, seed_nodes)
                cur_block.edata[EID] = cur_eid
                seed_nodes = cur_block.srcdata[NID]
                cur_blocks.insert(0, cur_block)

            return seed_nodes, output_nodes, org_blocks, prev_blocks, cur_blocks


class AugDataLoader:
    def __init__(self, g, train_nid, args, batch_size, shuffle=True, drop_last=False):
        self.g = g

        sampler = MyNeighborSampler([int(fanout) for fanout in args.fan_out.split(",")],
                                    prob=(None, "prev_emask", "cur_emask"))
        # DistNodeDataLoader로는 여러개 block을 불러올 수 없음
        self.dataloader = dgl.dataloading.DistNodeDataLoader(
            g, train_nid, sampler, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

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
