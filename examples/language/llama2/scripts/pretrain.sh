#!/bin/bash

################
pw=$1
#Load your environments and modules here
################

HOSTFILE=$(realpath hosts-16node.txt)

cd ..

export OMP_NUM_THREADS=8
export NCCL_P2P_LEVEL=PIX
export HF_DATASETS_OFFLINE=0

size=13b

sudo -E env "PATH=$PATH" colossalai run --nproc_per_node 8 --hostfile $HOSTFILE --extra_launch_args sudo_password=$pw pretrain.py \
    --config ../../../retnet/configs/retnet-$size \
    --model_name retnet \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --plugin hybrid_parallel \
    --batch_size 2048 \
    --block_size 4096 \
    --max_iters 1000000 \
    --lr 3e-4 \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --grad_checkpoint \
    --mixed_precision bf16 \
    --save_interval 1000 \
    --save_dir /nfs/data_mount/checkpoints \
    --run_name retnet-$size-rw-emb_llama \
    --tp 1 --pp 2 --zero_stage 1 --num_pp_mbs 32 \
    --datasets refinedweb --dataset_weights 1 \
    --init_embedding_from_llama
    # --load CKPT PATH --offload
