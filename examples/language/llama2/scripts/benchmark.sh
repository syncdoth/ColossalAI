#!/bin/bash

################
#Load your environments and modules here
################

skip_gemini=1
size=13b

HOSTFILE=$(realpath hosts.txt)
SAVEDIR=scripts/results-32node
cd ..
mkdir -p $SAVEDIR

export OMP_NUM_THREADS=8
export NCCL_P2P_LEVEL=PIX

###################################### gemini ##################################
if [ $skip_gemini != 1 ]; then
    echo "####################################"
    echo "benchmarking gemini for llama-$size..."
    echo "####################################"

    colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
        --model_name llama -c $size \
        -p gemini -g -x -b 16 -l 2048 > $SAVEDIR/llama-$size-gemini-x-b16-l2048.bench

    echo "####################################"
    echo "benchmarking gemini for retnet-$size..."
    echo "####################################"

    colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
        --model_name retnet -c ../../../retnet/configs/retnet-$size \
        -p gemini -g -x -b 16 -l 2048 > $SAVEDIR/retnet-$size-gemini-b16-l2048.bench
fi

####################################### 3D #####################################

function run() {
    steps=10
    model=$1
    p=$2
    bs=$3
    tp=$4
    pp=$5
    z=$6
    mbs=$7
    seqlen=$8

    if [[ $seqlen = '' ]]; then
        seqlen=4096
    fi

    if [ $model = retnet ]; then
        cfg=../../../retnet/configs/retnet-$size
    else
        cfg=$size
    fi

    echo "############################################################"
    echo "benchmarking for $model-$size..."
    echo "plugin = $p"
    echo "cfg: bs=$bs, tp=$tp, pp=$pp, zero=$z, mbs=$mbs, steps=$steps"
    echo "############################################################"

    colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py \
        --model_name $model -c $cfg \
        -p $p -s $steps -g -x \
        -b $bs --tp $tp --pp $pp --mbs $mbs -l $seqlen \
        --zero $z > $SAVEDIR/$model-$size-$p-b$bs-tp$tp-pp$pp-z$z-mbs$mbs-l$seqlen.bench
}

# Find best PP
for bs in 16 32 64; do
    for pp in 2 4; do
        for z in 0 1; do
            run retnet 3d $bs 1 $pp $z 4
        done
    done
done

run retnet 3d 16 1 2 1 16 8192
run retnet 3d 32 1 2 1 32 8192

# Try TP
# run retnet 3d 128 2 2 1 16
# run retnet 3d 128 4 2 1 16
# run retnet 3d 128 4 2 1 8
# run retnet 3d 64 4 2 1 8
# run retnet 3d 64 4 2 1 4
