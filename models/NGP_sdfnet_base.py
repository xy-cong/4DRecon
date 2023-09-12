import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn

class SDFNetwork(nn.Module):
    def __init__(self,
                 xyz_encoding="hashgrid",
                 time_encoding="frequency",
                 num_layers=3,
                 skips=[],
                 hidden_dim=64,
                 freq=2,
                 clip_sdf=None,
                 ):
        super().__init__()
        # import ipdb; ipdb.set_trace()

        # self.transformer = nn.Transformer(d_model=32 + 2*freq, nhead=8, num_encoder_layers=6)
        print("xyz_encoding, time_encoding", xyz_encoding, time_encoding)
        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.clip_sdf = clip_sdf
        self.flag = 0
        
        assert self.skips == [], 'TCNN does not support concatenating inside, please use skips=[].'

        if xyz_encoding == "hashgrid" and time_encoding == 'hashgrid':
            self.flag = 0
            self.xyzt_encoder = tcnn.Encoding(
                n_input_dims=4,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": 19,
                    "base_resolution": 16,
                    "per_level_scale": 1.3819,
                },
            )

            time_add_dim = 0

        elif xyz_encoding == "hashgrid" and time_encoding == 'frequency':
            self.flag = 1
            self.xyz_encoder = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": 19,
                    "base_resolution": 16,
                    "per_level_scale": 1.3819,
                },
            )

            self.time_encoder = tcnn.Encoding(
                n_input_dims=1,
                encoding_config={
                    "otype": "Frequency",
                    "n_frequencies": freq,
                },
            )

            time_add_dim = freq*2
        else:
            raise NotImplementedError

        # self.backbone = tcnn.Network(
        #     n_input_dims=32 + time_add_dim, # 32 + encoding_dim(t) : to be estimated!!!!
        #     n_output_dims=1,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": hidden_dim,
        #         "n_hidden_layers": num_layers - 1,
        #     },
        # )

        self.backbone = nn.Sequential(
                nn.Linear(32 + time_add_dim, 64), nn.LayerNorm(64), 
                nn.GELU(),
                nn.Linear(64, 64), nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, 1)
            )

    
    def forward(self, time, x):
        # x: [B, 3]
        # import ipdb; ipdb.set_trace()
        x = (x + 1) / 2 # to [0, 1]
        if self.flag == 0:
            input = torch.cat([x, time], dim=1)
            input = self.xyzt_encoder(input)
        elif self.flag == 1:
            xyz = self.xyz_encoder(x)
            time = self.time_encoder(time)
            input = torch.cat([xyz, time], dim=1)

        transformer_feature = input
        # transformer_feature = self.transformer.encoder(input.unsqueeze(0))[0]

        h = self.backbone(transformer_feature)

        if self.clip_sdf != 'None':
            h = h.clamp(-self.clip_sdf, self.clip_sdf)

        return h
    
