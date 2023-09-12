
# Commands

## Train the network
```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/seq_hand_x_0_0_0/arap/hand_x3/sdf_seq_hand_x3_ARAP_SE4k_SLSLine_w1e-3.yaml --mode train --rep sdf
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/run.yaml --mode train --rep sdf
```

## Interpolation
```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/seq_hand_x_0_0_0/arap/hand_x3/sdf_seq_hand_x3_ARAP_SE4k_SLSLine_w1e-3.yaml --mode interp --rep sdf --split train --continue_from 3999

CUDA_VISIBLE_DEVICES=3 python main.py --config ./config/run.yaml --mode interp --rep sdf --split train --continue_from 3999
```

## Visualize interpolated shapes
```
cd ./visualization
python vis_sdf2d.py
```

## libssl.so.1.1 error:

```
export LD_LIBRARY_PATH=$HOME/opt/lib:$LD_LIBRARY_PATH
```


