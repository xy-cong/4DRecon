#!/bin/bash
export LD_LIBRARY_PATH=$HOME/opt/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
# 使用seq命令生成序列
for value in $(seq 9999 1000 10000)
do
   # 执行你的命令，使用$value作为continue_from的值
   CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/run.yaml --mode interp --rep sdf --split train --continue_from $value
done