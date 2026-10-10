export CUDA_VISIBLE_DEVICES=0,1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=/usr/local/cuda
export CUDNN_HOME=/data/gaoya/agent-data/envs/cosmos_predict25_official/lib/python3.10/site-packages/nvidia/cudnn
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:/home/gaoya/miniconda3/envs/flux/lib/python3.10/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
for cosmos_cuda_lib in /home/gaoya/miniconda3/envs/flux/lib/python3.10/site-packages/nvidia/*/lib; do
    [[ "$cosmos_cuda_lib" == */cudnn/lib ]] || export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$cosmos_cuda_lib"
done
unset cosmos_cuda_lib
export PATH=/data/gaoya/agent-data/envs/cosmos_predict25_official/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=.:packages/cosmos-oss
export HF_HOME=/data/gaoya/agent-data/cache/huggingface
export IMAGINAIRE_OUTPUT_ROOT=/data/gaoya/agent-data/outputs_v1/cosmos_predict25/20261004_official_acn75
export WANDB_ENTITY=875222004-gy
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PIP_CACHE_DIR=/data/gaoya/agent-data/cache/pip_cosmos_predict25_official
export XDG_CACHE_HOME=/data/gaoya/agent-data/cache/cosmos_predict25_official
export TORCH_EXTENSIONS_DIR=/data/gaoya/agent-data/cache/cosmos_predict25_official/torch_extensions
