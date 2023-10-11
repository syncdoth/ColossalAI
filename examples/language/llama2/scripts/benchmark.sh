#!/bin/bash

################
#Load your environments and modules here
################

HOSTFILE=$(realpath hosts-4node.txt)

cd ..

export OMP_NUM_THREADS=8
export NCCL_P2P_LEVEL=PIX

# gemini
# colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
#     --model_name llama -c 7b \
#     -p gemini -g -x -b 8 > llama-7b-x-b8.bench

# colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
#     --model_name llama -c 7b \
#     -p gemini -g -x -b 16 > llama-7b-x-b16.bench

# # colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
# #     --model_name retnet -c ../../../retnet/configs/retnet-7b \
# #     -p gemini -g -b 8 > retnet-7b-b8.bench

# colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
#     --model_name retnet -c ../../../retnet/configs/retnet-7b \
#     -p gemini -g -b 16 > retnet-7b-b16.bench

# # 3D
# colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
#     --model_name llama -c 7b \
#     -p 3d -g -x -b 32 --tp 4 --pp 2 --mbs 4 > llama-7b-x-3d-b32.bench

# colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
#     --model_name llama -c 7b \
#     -p 3d -g -x -b 64 --tp 4 --pp 2 --mbs 4 > llama-7b-x-3d-b64.bench

# colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
#     --model_name llama -c 7b \
#     -p 3d -g -x -b 128 --tp 4 --pp 2 --mbs 4 > llama-7b-x-3d-b128.bench

# colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
#     --model_name retnet -c ../../../retnet/configs/retnet-7b \
#     -p 3d -g -x -b 32 --tp 4 --pp 2 --mbs 4 > retnet-7b-3d-b32.bench

# colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
#     --model_name retnet -c ../../../retnet/configs/retnet-7b \
#     -p 3d -g -x -b 64 --tp 4 --pp 2 --mbs 4 > retnet-7b-3d-b64.bench


# colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
#     --model_name retnet -c ../../../retnet/configs/retnet-7b \
#     -p 3d -g -x -b 32 --tp 2 --pp 4 --mbs 4 > retnet-7b-3d-b32-tp2-pp4.bench

# colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
#     --model_name retnet -c ../../../retnet/configs/retnet-7b \
#     -p 3d -g -x -b 16 --tp 1 --pp 4 --mbs 4 > retnet-7b-3d-b16-tp1-pp4.bench

# colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
#     --model_name retnet -c ../../../retnet/configs/retnet-7b \
#     -p 3d -g -x -b 32 --tp 1 --pp 4 --mbs 4 > retnet-7b-3d-b32-tp1-pp4.bench

# colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
#     --model_name retnet -c ../../../retnet/configs/retnet-7b \
#     -p 3d -g -x -b 16 --tp 1 --pp 2 --mbs 4 > retnet-7b-3d-b16-tp1-pp2.bench


# colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
#     --model_name retnet -c ../../../retnet/configs/retnet-7b \
#     -p 3d -g -x -b 32 --tp 1 --pp 2 --mbs 4 --zero 1 > retnet-7b-3d-b32-tp1-pp2-z1.bench


colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
    --model_name retnet -c ../../../retnet/configs/retnet-7b \
    -p 3d -g -x -b 64 --tp 1 --pp 2 --mbs 4 --zero 1 > retnet-7b-3d-b64-tp1-pp2-z1.bench


colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
    --model_name retnet -c ../../../retnet/configs/retnet-7b \
    -p 3d -g -x -b 32 --tp 1 --pp 2 --mbs 2 --zero 1 > retnet-7b-3d-b32-tp1-pp2-z1-m2.bench

