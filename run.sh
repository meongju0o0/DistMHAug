#!/bin/bash
python3 -m augmentation.masking
python3 -m common.calc common.config common.set_graph common.load_batch
python3 -m training.evaluation training.loss training.model
python3 /mnt/shared/development/dgl/juyeong/launch.py --workspace /mnt/shared/development/dgl/juyeong --num_trainers 1 --num_samplers 0 --num_servers 1 --part_config /mnt/shared/development/dgl/juyeong/data/citeseer.json --ip_config ip_config.txt "python3 node_classification.py --graph_name citeseer --ip_config ip_config.txt --num_epochs 2000 --batch_size 100"
