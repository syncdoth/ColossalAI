# torch
echo "install torch"
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# conda
echo "install other conda packages"
conda install pandas matplotlib -y
conda install xformers -c xformers=0.0.22 -y

# pip install
echo "install pip packages"
pip install transformers==4.31 datasets sentencepiece timm ninja packaging tensorboard==2.14.0 wandb
# flash attn
echo "install flash attn"
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn==2.0.5 --no-build-isolation

# apex install
echo "install apex"
cd $HOME/installs/apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# finally, colossalai
echo "install colossalai"
cd $HOME/ColossalAI
CUDA_EXT=1 pip install -e .
