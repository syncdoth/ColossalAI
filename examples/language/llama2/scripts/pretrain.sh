#!/bin/bash

################
#Load your environments and modules here
################

HOSTFILE=$(realpath hosts.txt)

cd ..

export OMP_NUM_THREADS=8

colossalai run --nproc_per_node 8 --hostfile $HOSTFILE pretrain.py \
    -p 3d -g -x -b 2048 --mbs 1024 --tp 4 --pp 2 --zero 1 # -p 3d_cpu
