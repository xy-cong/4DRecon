#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from models.encoder import get_encoder
import tinycudann as tcnn


class SdfDecoder_Modify(nn.Module):
    def __init__(
        self,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        xyz_encoder=None,
        use_xyz_encoder=True,
        lat_dim=256,
        **kwargs
    ):
        super().__init__()
        self.flag = 0
        if use_xyz_encoder:
            self.xyz_encoder, input_xyz_dim = get_encoder(xyz_encoder)
        else:
            self.xyz_encoder = None
            input_xyz_dim = 3
        dims = [input_xyz_dim + lat_dim] + list(dims)
        output_dim = 1

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.Layers = nn.ModuleList()

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - (input_xyz_dim)
            else:
                out_dim = dims[layer + 1]

            if layer in self.norm_layers:
                Layer = nn.Sequential(
                    nn.Linear(dims[layer], out_dim, bias=False), nn.LayerNorm(out_dim), nn.GELU()
                )
            else:
                Layer = nn.Sequential(
                    nn.Linear(dims[layer], out_dim), nn.GELU()
                )
            self.Layers.append(Layer)

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.output_Layer = nn.Linear(dims[-1], output_dim)
        # self.th = nn.Tanh()

    def forward(self, lat_vec, xyz):
        """
        Args:
            lat_vecs: (N, L)
            xyz: (N, query_dim)
        Returns:
            x: (N,)
        """
        # import ipdb; ipdb.set_trace()
        xyz = (xyz + 1) / 2 # to [0, 1]
        if self.xyz_encoder is not None:
            xyz_encoding = self.xyz_encoder(xyz)
        else:
            xyz_encoding = xyz
        x = torch.cat([lat_vec, xyz_encoding], dim=-1)
        x_input = xyz_encoding
        # x = torch.cat([time_MLP, xyz_MLP], dim=-1) # (N, L+query_dim)
        # x_input = torch.cat([time_encoding, xyz_encoding], dim=-1)

        for layer in range(0, self.num_layers-1):
            Layer = self.Layers[layer]
            if layer in self.latent_in:
                x = torch.cat([x, x_input], 1)
            x = Layer(x)
            # if layer < self.num_layers - 2:
            #     if self.dropout is not None and layer in self.dropout:
            #         x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.output_Layer(x)
        # x = self.th(x)

        return x
    


def get_freq_reg_mask(pos_enc_length, current_iter, total_reg_iter, max_visible=None, type='submission'):
    '''
    Returns a frequency mask for position encoding in NeRF.

    Args:
    pos_enc_length (int): Length of the position encoding.
    current_iter (int): Current iteration step.
    total_reg_iter (int): Total number of regularization iterations.
    max_visible (float, optional): Maximum visible range of the mask. Default is None. 
        For the demonstration study in the paper.

    Correspond to FreeNeRF paper:
        L: pos_enc_length
        t: current_iter
        T: total_iter

    Returns:
    jnp.array: Computed frequency or visibility mask.
    '''
    if max_visible is None:
    # default FreeNeRF
        if current_iter < total_reg_iter:
            freq_mask = np.zeros(pos_enc_length)  # all invisible
            ptr = pos_enc_length / 3 * current_iter / total_reg_iter + 1 
            ptr = ptr if ptr < pos_enc_length / 3 else pos_enc_length / 3
            int_ptr = int(ptr)
            freq_mask[: int_ptr * 3] = 1.0  # assign the integer part
            freq_mask[int_ptr * 3 : int_ptr * 3 + 3] = (ptr - int_ptr)  # assign the fractional part
            return np.clip(np.array(freq_mask), 1e-8, 1-1e-8)  # for numerical stability
        else:
            return np.ones(pos_enc_length)
    else:
        # For the ablation study that controls the maximum visible range of frequency spectrum
        freq_mask = np.zeros(pos_enc_length)
        freq_mask[: int(pos_enc_length * max_visible)] = 1.0
        return np.array(freq_mask)