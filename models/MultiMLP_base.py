#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from models.encoder import get_encoder
# import tinycudann as tcnn
from einops import rearrange, repeat

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

        
            self.time_embeddings_0 = nn.Embedding(Levels[0]+1, time_embedding_dim)
            self.time_embeddings_1 = nn.Embedding(Levels[1]+1, time_embedding_dim)
        
            self.time_embeddings = [
                self.time_embeddings_0,
                self.time_embeddings_1
            ]
        else:
            raise NotImplementedError

        
        # Multi_MLP_base = nn.Sequential(
        #             nn.utils.weight_norm(nn.Linear(time_embedding_dim+xyz_embedding_dim, Multi_MLP_dim)),
        #             nn.ReLU(),
        #             nn.Linear(Multi_MLP_dim, Multi_MLP_dim)
        #         )
        # # Level 1
        # self.Multi_MLPs= [
        #     [Multi_MLP_base for i in range(Levels[j]+1)] for j in range(len(Levels))
        # ]
        
        # input_dim = Multi_MLP_dim*len(Levels)
        input_dim = xyz_embedding_dim + time_embedding_dim*2
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
        
        device = time.device

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
                    if idx == len(self.Levels_intervals[level])-2:
                        idxs.append(idx+1)
                        weights.append(np.array([0]).astype(np.float32))
            time_level_idx.append(idxs)
            interp_weigths.append(weights)

        time_idxs = torch.tensor(time_level_idx).to(device)
        interp_weigths = torch.from_numpy(np.array(interp_weigths)).to(device)
        # import ipdb; ipdb.set_trace()
        time_encodings = []
        for id in range(time_idxs.shape[0]):
            idxs = time_idxs[id]
            interp_weigth = interp_weigths[id]
            time_encs = []

            if idxs[0] == self.Levels[0] and idxs[1] == self.Levels[1]:
                time_encoding_1 = self.time_embeddings[0](idxs[0].to(device))
                time_encoding_2 = self.time_embeddings[1](idxs[1].to(device))
                
                time_encoding = torch.cat((time_encoding_1, time_encoding_2), dim=0)
                time_encodings.append(time_encoding[None, :])
                continue

            for level in range(len(self.Levels)):
                idx = idxs[level]
                time_encoding_1 = self.time_embeddings[level](idx.to(device))
                time_encoding_2 = self.time_embeddings[level](idx.to(device)+1)
                time_enc = interp_weigth[level]*time_encoding_1 + (1-interp_weigth[level])*time_encoding_2
                time_encs.append(time_enc)

            time_encoding = torch.cat(time_encs, dim=0)
            time_encodings.append(time_encoding[None, :])
        

        time_encodings = torch.cat(time_encodings, dim=0).to(device)
        
        time_encodings = repeat(time_encodings, 'B d -> (B S) d', S=xyz.shape[1])

        xyz = rearrange(xyz, 'B S d -> (B S) d')
        xyz = (xyz + 1) / 2 # to [0, 1]
        xyz_embedding = self.xyz_embedding(xyz)

        x = torch.cat([time_encodings, xyz_embedding], dim=1)
        x_input = x

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

    