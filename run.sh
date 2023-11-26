#!/bin/bash

python3 -m augmentation.masking
python3 -m common.calc common.config common.set_graph common.load_batch
python3 -m training.evaluation training.loss training.model

python3 /mnt/shared/development/dgl/juyeong/launch.py \
--workspace /mnt/shared/development/dgl/juyeong \
--num_trainers 1 \
--num_samplers 0 \
--num_servers 1 \
--part_config /mnt/shared/data/partitioned/parts4/ogb-product/ogb-product.json \
--ip_config ip_config.txt \
"python3 node_classification.py --graph_name ogb-product --ip_config ip_config.txt --num_epochs 30 --batch_size 1000"

# cora: /mnt/shared/development/dgl/juyeong/data/4partition/cora/cora.json
# citeseer: /mnt/shared/development/dgl/juyeong/data/4partition/citeseer/citeseer.json
# ogbn-products: /mnt/shared/data/partitioned/parts4/ogb-product/ogb-product.json
