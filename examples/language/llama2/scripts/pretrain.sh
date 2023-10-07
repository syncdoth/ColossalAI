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
    --config ../../../retnet/configs/retnet-3b \
    --model_name retnet \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --plugin hybrid_parallel \
    --batch_size 2048 \
    --micro_batch_size 2048 \
    --num_pp_mbs 4 \
    --block_size 2048 \
    --max_iters 1000000 \
    --lr 3e-4 \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --grad_checkpoint \
    --mixed_precision bf16 \
    --save_interval 5000 \
    --save_dir /nfs/data_mount/checkpoints \
    --run_name retnet-3b-glu-tok-llama-data-rw-3d-parallel \
    --tp 4 --pp 4 --zero_stage 1 --offload \
    --datasets refinedweb --dataset_weights 1
    # --load CKPT PATH
    # epoch -> max iter TODO
    # --dataset wikipedia # TODO
