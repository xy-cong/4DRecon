import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn

class SDF_Base(nn.Module):
    def __init__(self,
                 config,
                 clip_sdf=None,
                 ):
        super().__init__()
        # import ipdb; ipdb.set_trace()
        self.clip_sdf = clip_sdf

        config_encoder = config.xyz_encoder
        if config_encoder.type == "hashgrid":
            self.xyz_encoder = tcnn.Encoding(
                n_input_dims=config_encoder.input_dim,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": config_encoder.n_levels,
                    "n_features_per_level": config_encoder.n_features_per_level,
                    "log2_hashmap_size": 19,
                    "base_resolution": 16,
                    "per_level_scale": 1.3819,
                },
            )
        else:
            raise NotImplementedError

        input_dim = config_encoder.n_levels*config_encoder.n_features_per_level
        hidden_dim = config_encoder.hidden_dim
        out_dim = config_encoder.out_dim
        self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), 
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim)
            )

    
    def forward(self, x):
        # x: [B, 3]
        # import ipdb; ipdb.set_trace()
        x = (x + 1) / 2 # to [0, 1]
        encoding = self.xyz_encoder(x)

        h = self.backbone(encoding)

        if self.clip_sdf != None:
            h = h.clamp(-self.clip_sdf, self.clip_sdf)

        return h, encoding
    
    
class SDF_warpnet(nn.Module):
    def __init__(self,
                 config,
                 clip_sdf=None,
                 ):
        super().__init__()
        # import ipdb; ipdb.set_trace()

        self.clip_sdf = clip_sdf

        config_xyz_encoder = config.xyz_encoder
        config_time_encoder = config.time_encoder
        if config_xyz_encoder.type == "hashgrid" and config_time_encoder.type == 'hashgrid':
            self.xyzt_encoder = tcnn.Encoding(
                n_input_dims=config_xyz_encoder.input_dim + config_time_encoder.input_dim,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": 19,
                    "base_resolution": 16,
                    "per_level_scale": 1.3819,
                },
            )
        else:
            raise NotImplementedError

        input_dim = config_xyz_encoder.n_levels*config_xyz_encoder.n_features_per_level
        hidden_dim = config_xyz_encoder.hidden_dim
        out_dim = config_xyz_encoder.out_dim
        self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), 
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim)
            )

    def forward(self, time, x):
        # x: [B, 3]
        # import ipdb; ipdb.set_trace()
        x = (x + 1) / 2 # to [0, 1]
        input = torch.cat([x, time], dim=1)
        encoding = self.xyzt_encoder(input)

        dxdydz = self.backbone(encoding)

        if self.clip_sdf != None:
            dxdydz = dxdydz.clamp(-self.clip_sdf, self.clip_sdf)

        return dxdydz, encoding


class SDF_Warping(nn.Module):
    def __init__(self,
                 config,
                 clip_sdf=None,
                 ):
        super().__init__()
        # import ipdb; ipdb.set_trace()
        self.conanical_net = SDF_Base(
            config.canonical_net
        )
        self.warp_net = SDF_warpnet(
            config.warp_net
        )

        input_dim = config.canonical_net.xyz_encoder.n_levels * config.canonical_net.xyz_encoder.n_features_per_level + \
                    config.warp_net.xyz_encoder.n_levels * config.warp_net.xyz_encoder.n_features_per_level
        hidden_dim = config.residual_net.hidden_dim
        out_dim = config.residual_net.out_dim
        self.residual_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), 
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim)
            )
        

        self.clip_sdf = clip_sdf
        self.flag = 0

    def forward(self, time, x, is_canonical=False):
        # x: [B, 3]
        # import ipdb; ipdb.set_trace()
        xyz = (x+1)/2
        if is_canonical:
            sdf, _ = self.conanical_net(xyz)
        else:
            dxdydz, encoding_warp = self.warp_net(time, xyz)
            xyz = xyz + dxdydz
            sdf, encoding_canonical = self.conanical_net(xyz)

            encoding = torch.cat([encoding_canonical, encoding_warp], dim=1)
            residual = self.residual_net(encoding)
            sdf = sdf + residual
        if self.clip_sdf != None:
            sdf = sdf.clamp(-self.clip_sdf, self.clip_sdf)
        return sdf
