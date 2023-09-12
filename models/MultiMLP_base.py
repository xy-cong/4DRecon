#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from models.encoder import get_encoder
import tinycudann as tcnn

class SdfDecoder_MultiMLP(nn.Module):
    def __init__(
        self,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        xyz_encoder=None,
        use_xyz_encoder=True,
        time_encoder=None,
        use_time_encoder=True,
        time_embedding_dim=8,
        Multi_MLP_dim=64,
        Levels=[5, 20],
        **kwargs
    ):
        super().__init__()
        self.Levels = Levels
        self.Levels_intervals = [
            [j / Levels[i] for j in range(Levels[i] + 1)] for i in range(len(Levels))
        ]
        assert len(Levels) == 2
        # import ipdb; ipdb.set_trace()
        if use_time_encoder and use_xyz_encoder:
            self.xyz_embedding, xyz_embedding_dim = get_encoder(xyz_encoder)

            self.time_embeddings = [
                nn.Embedding(Levels[i]+1, time_embedding_dim) for i in range(len(Levels))
            ]

        else:
            raise NotImplementedError

        
        Multi_MLP_base = nn.Sequential(
                    nn.utils.weight_norm(nn.Linear(time_embedding_dim+xyz_embedding_dim, Multi_MLP_dim)),
                    nn.ReLU(),
                    nn.Linear(Multi_MLP_dim, Multi_MLP_dim)
                )
        # Level 1
        self.Multi_MLPs= [
            [Multi_MLP_base for i in range(Levels[j]+1)] for j in range(len(Levels))
        ]
        
        input_dim = Multi_MLP_dim*len(Levels)
        dims = [input_dim] + list(dims)
        output_dim = 1

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.Layers = nn.ModuleList()

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - (input_dim)
            else:
                out_dim = dims[layer + 1]

            if layer in self.norm_layers:
                Layer = nn.Sequential(
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim, bias=False)),nn.ReLU()
                )
            else:
                Layer = nn.Sequential(
                    nn.Linear(dims[layer], out_dim), nn.ReLU()
                )
            self.Layers.append(Layer)

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.output_Layer = nn.Linear(dims[-1], output_dim)
        # self.th = nn.Tanh()

    def forward(self, time, xyz):
        """
        Args:
            lat_vecs: (N, L)
            xyz: (N, query_dim)
        Returns:
            x: (N,)
        """
        # import ipdb; ipdb.set_trace()
        device = time.device
        assert(time.shape[0] == xyz.shape[0])
        assert(len(time.shape) == len(xyz.shape) == 2)
        xyz = (xyz + 1) / 2 # to [0, 1]
        
        xyz_embedding = self.xyz_embedding(xyz)

        time_level_idx = []
        interp_weigths = []
        time_cpu = time.cpu().numpy()

        for time in time_cpu:
            idxs = []
            weights = []
            for level in range(len(self.Levels)):
                for idx in range(len(self.Levels_intervals[level])-1):
                    if time >= self.Levels_intervals[level][idx] and time < self.Levels_intervals[level][idx+1]:
                        idxs.append(idx)
                        weights.append((self.Levels_intervals[level][idx+1] - time) / (self.Levels_intervals[level][idx+1] - self.Levels_intervals[level][idx]))
                        break
            time_level_idx.append(idxs)
            interp_weigths.append(weights)
        time_idxs = torch.tensor(time_level_idx).to(device)

        time_
        for idxs in time_idxs:


        for i in range(len(self.Levels)):
            for idx in range(len(self.Levels_intervals[i])-1):
                if time
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

    