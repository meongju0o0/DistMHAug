from concurrent.futures import ThreadPoolExecutor

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

        self.executor = ThreadPoolExecutor(max_workers=2)


    def __iter__(self):
        return self._generator()


    @staticmethod
    def _load_data(dataloader):
        items = []
        for item in dataloader:
            items.append(item)
        return items


    def _generator(self):
        for step, (src_nodes, dst_nodes, blocks) in enumerate(self.org_dataloader):
            org_src_nodes = src_nodes
            org_dst_nodes = dst_nodes
            org_blocks = blocks

            prev_dataloader = dgl.dataloading.DistNodeDataLoader(
                self.g, dst_nodes, self.samplers[1],
                batch_size=dst_nodes.size(dim=0), shuffle=False, drop_last=self.drop_last)

            cur_dataloader = dgl.dataloading.DistNodeDataLoader(
                self.g, dst_nodes, self.samplers[2],
                batch_size=dst_nodes.size(dim=0), shuffle=False, drop_last=self.drop_last)

            future_prev = self.executor.submit(self._load_data, prev_dataloader)
            future_cur = self.executor.submit(self._load_data, cur_dataloader)

            prev_data = future_prev.result()
            cur_data = future_cur.result()

            prev_src_nodes, prev_dst_nodes, prev_blocks = prev_data[0]
            cur_src_nodes, cur_dst_nodes, cur_blocks = cur_data[0]

            yield {"org": [org_src_nodes, org_dst_nodes, org_blocks],
                   "prev": [prev_src_nodes, prev_dst_nodes, prev_blocks],
                   "cur": [cur_src_nodes, cur_dst_nodes, cur_blocks]}


    def __del__(self):
        self.executor.shutdown(wait=True)
