#!/bin/bash
python3 -m distpartitioning.array_readwriter
python3 -m files
python3 random_partition.py --in_dir /mnt/shared/development/dgl/juyeong/dataset --out_dir /mnt/shared/development/dgl/juyeong/data
