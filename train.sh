# export PATH=/usr/local/cuda-11.1/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/opt/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=3 python main.py \
        --config ./config/run_sdf.yaml \
        --mode train \
        --rep sdf 