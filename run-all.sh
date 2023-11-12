#!/bin/env bash

chmod +x run-citeseer.sh run-cora.sh run-ogbn-papers100M.sh run-ogbn-products.sh

./run-citeseer.sh && \
./run-cora.sh && \
./run-ogbn-papers100M.sh && \
./run-ogbn-products.sh
