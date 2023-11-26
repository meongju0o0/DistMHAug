#!/bin/bash

chmod +x run-cora.sh run-citeseer.sh run-ogbn-products.sh

./run-cora.sh && \
./run-citeseer.sh && \
./run-ogbn-products.sh
