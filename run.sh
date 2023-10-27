#!/bin/bash
python -m augmentation.masking
python -m common.calc common.config common.set_graph common.load_batch
python -m training.evaluation training.loss training.model
python /workspace/launch.py --workspace /workspace --num_trainers 1 --num_samplers 0 --num_servers 1 --part_config data/ogbn-products.json --ip_config ip_config.txt "python node_classification.py --graph_name ogbn-products --ip_config ip_config.txt --num_epochs 30 --batch_size 1000"
