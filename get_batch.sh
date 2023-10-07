#!/bin/sh
srun --gres=gpu:1 --cpus-per-gpu=4 --mem-per-gpu=10G --account ugrad_ce --partition debug_ce_ugrad --pty bash
