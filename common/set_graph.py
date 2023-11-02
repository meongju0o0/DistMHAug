import numpy as np
import torch as th
import dgl


class SetGraph:
    def __init__(self, g, args):
        self.args = args
        self.g =g
        self.train_nid, self.val_nid, self.test_nid = None, None, None
        self.in_feats, self.n_classes = None, None


    def __call__(self):
        self._train_test_split()
        self._get_classes()
        return self._pack_data()


    def _train_test_split(self):
        pb = self.g.get_partition_book()

        if "trainer_id" in self.g.ndata:
            self.train_nid = dgl.distributed.node_split(
                self.g.ndata["train_mask"],
                pb,
                force_even=True,
                node_trainer_ids=self.g.ndata["trainer_id"],
            )
            self.val_nid = dgl.distributed.node_split(
                self.g.ndata["val_mask"],
                pb,
                force_even=True,
                node_trainer_ids=self.g.ndata["trainer_id"],
            )
            self.test_nid = dgl.distributed.node_split(
                self.g.ndata["test_mask"],
                pb,
                force_even=True,
                node_trainer_ids=self.g.ndata["trainer_id"],
            )

        else:
            self.train_nid = dgl.distributed.node_split(self.g.ndata["train_mask"], pb, force_even=True)
            self.val_nid = dgl.distributed.node_split(self.g.ndata["val_mask"], pb, force_even=True)
            self.test_nid = dgl.distributed.node_split(self.g.ndata["test_mask"], pb, force_even=True)
        local_nid = pb.partid2nids(pb.partid).detach().numpy()

        num_train_local = len(np.intersect1d(self.train_nid.numpy(), local_nid))
        num_val_local = len(np.intersect1d(self.val_nid.numpy(), local_nid))
        num_test_local = len(np.intersect1d(self.test_nid.numpy(), local_nid))
        print(
            f"part {self.g.rank()}, train: {len(self.train_nid)} (local: {num_train_local}), "
            f"val: {len(self.val_nid)} (local: {num_val_local}), "
            f"test: {len(self.test_nid)} (local: {num_test_local})"
        )

        del local_nid


    def _get_classes(self):
        n_classes = self.args.n_classes

        if n_classes == 0:
            labels = self.g.ndata["labels"][np.arange(self.g.num_nodes())]
            self.n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
            del labels

        if SetGraph.cnt == 1:
            print(f"Number of classes: {n_classes}")


    def _pack_data(self):
        self.in_feats = self.g.ndata["features"].shape[1]
        data = self.train_nid, self.val_nid, self.test_nid, self.in_feats, self.n_classes, self.g
        return data
