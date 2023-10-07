#!/bin/bash

################
#Load your environments and modules here
################

HOSTFILE=$(realpath hosts.txt)

cd ..

export OMP_NUM_THREADS=8
export NCCL_P2P_LEVEL=PIX
export HF_DATASETS_OFFLINE=0

# sudo -E env "PATH=$PATH"
colossalai run --nproc_per_node 8 --hostfile $HOSTFILE pretrain.py \
    --config ../../../retnet/configs/retnet-xl \
    --model_name retnet \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --plugin hybrid_parallel \
    --num_epochs 10 \
    --batch_size 2048 \
    --micro_batch_size 1024 \
    --block_size 2048 \
    --max_iters 1000000 \
    --lr 3e-4 \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --grad_checkpoint \
    --mixed_precision bf16 \
    --save_interval 5000 \
    --save_dir /nfs/data_mount/checkpoints \
    --run_name retnet-1.3b-glu-tok-llama-3d-parallel \
    --tp 4 --pp 2 --zero_stage 1 --offload
    # --load CKPT PATH
    # epoch -> max iter TODO
    # --dataset wikipedia # TODO
