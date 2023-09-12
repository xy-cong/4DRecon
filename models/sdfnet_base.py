#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from models.encoder import get_encoder
import tinycudann as tcnn

class SdfDecoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        query_dim=None,
        **kwargs
    ):
        super().__init__()
        # import ipdb; ipdb.set_trace()
        dims = [latent_size + query_dim] + list(dims)
        output_dim = 1

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.Layers = nn.ModuleList()

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]

            if layer in self.norm_layers:
                Layer = nn.Sequential(
                    nn.Linear(dims[layer], out_dim), nn.LayerNorm(out_dim), nn.GELU()
                )
            else:
                Layer = nn.Sequential(
                    nn.Linear(dims[layer], out_dim), nn.GELU()
                )
            self.Layers.append(Layer)

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.output_Layer = nn.Linear(dims[-1], output_dim)
        self.th = nn.Tanh()

    def forward(self, lat_vecs, xyz):
        """
        Args:
            lat_vecs: (N, L)
            xyz: (N, query_dim)
        Returns:
            x: (N,)
        """
        # import ipdb; ipdb.set_trace()
        assert(lat_vecs.shape[0] == xyz.shape[0])
        assert(len(lat_vecs.shape) == len(xyz.shape) == 2)

        x = torch.cat([lat_vecs, xyz], dim=-1) # (N, L+query_dim)
        x_input = x

        for layer in range(0, self.num_layers-1):
            Layer = self.Layers[layer]
            if layer in self.latent_in:
                x = torch.cat([x, x_input], 1)
            x = Layer(x)
            if layer < self.num_layers - 2:
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.output_Layer(x)
        x = self.th(x)

        return x

class SdfDecoder_Encoder(nn.Module):
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
        **kwargs
    ):
        super().__init__()
        # import ipdb; ipdb.set_trace()
        self.flag = 0
        if use_xyz_encoder and not use_time_encoder:
            self.xyz_encoder, input_xyz_dim = get_encoder(xyz_encoder)
            input_time_dim = 1
            self.flag = 0
        if use_time_encoder and not use_xyz_encoder:
            self.time_encoder, input_time_dim = get_encoder(time_encoder)
            input_xyz_dim = 3
            self.flag = 1
        if use_time_encoder and use_xyz_encoder:
            self.xyz_encoder, input_xyz_dim = get_encoder(xyz_encoder)
            self.time_encoder, input_time_dim = get_encoder(time_encoder)
            self.flag = 2
        
        time_MLP_out_dim = 32
        self.time_MLP = nn.Sequential(
                nn.Linear(input_time_dim, 64), nn.LayerNorm(64), 
                nn.GELU(),
                nn.Linear(64, 64), nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, time_MLP_out_dim)
            )

        dims = [time_MLP_out_dim + input_xyz_dim] + list(dims)
        output_dim = 1

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.Layers = nn.ModuleList()

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]

            if layer in self.norm_layers:
                Layer = nn.Sequential(
                    nn.Linear(dims[layer], out_dim), nn.LayerNorm(out_dim), nn.GELU()
                )
            else:
                Layer = nn.Sequential(
                    nn.Linear(dims[layer], out_dim), nn.GELU()
                )
            self.Layers.append(Layer)

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.output_Layer = nn.Linear(dims[-1], output_dim)
        self.th = nn.Tanh()

    def forward(self, time, xyz, lat_vecs=None):
        """
        Args:
            lat_vecs: (N, L)
            xyz: (N, query_dim)
        Returns:
            x: (N,)
        """
        # import ipdb; ipdb.set_trace()
        assert(time.shape[0] == xyz.shape[0])
        assert(len(time.shape) == len(xyz.shape) == 2)
        xyz = (xyz + 1) / 2 # to [0, 1]
        if self.flag == 0:
            xyz_encoding = self.xyz_encoder(xyz)
            time_encoding = time
        elif self.flag == 1:
            xyz_encoding = xyz
            time_encoding = self.time_encoder(time)
        elif self.flag == 2:
            xyz_encoding = self.xyz_encoder(xyz)
            time_encoding = self.time_encoder(time)
            time_encoding = self.time_MLP(time_encoding)
        x = torch.cat([time_encoding, xyz_encoding], dim=-1) # (N, L+query_dim)
        x_input = x

        for layer in range(0, self.num_layers-1):
            Layer = self.Layers[layer]
            if layer in self.latent_in:
                x = torch.cat([x, x_input], 1)
            x = Layer(x)
            if layer < self.num_layers - 2:
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.output_Layer(x)
        x = self.th(x)

        return x


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
        time_encoder=None,
        use_time_encoder=True,
        **kwargs
    ):
        super().__init__()
        # import ipdb; ipdb.set_trace()
        self.flag = 0
        if use_xyz_encoder and not use_time_encoder:
            self.xyz_encoder, input_xyz_dim = get_encoder(xyz_encoder)
            # self.xyz_MLP = tcnn.Network(
            #     n_input_dims=input_xyz_dim,
            #     n_output_dims=xyz_encoder.MLP_dim,
            #     network_config={
            #         "otype": "FullyFusedMLP",
            #         "activation": "ReLU",
            #         "output_activation": "None",
            #         "n_neurons": xyz_encoder.hidden_dim,
            #         "n_hidden_layers": xyz_encoder.num_layers - 1,
            #     },
            # )
            # MLP_xyz_dim = xyz_encoder.MLP_dim
            # input_xyz_dim = 3
            MLP_xyz_dim = input_xyz_dim
            if time_encoder.type == 'embedding':
                self.time_emb = torch.nn.Embedding(3, time_encoder.embedding_dim)
                MLP_time_dim = time_encoder.embedding_dim
                input_time_dim = time_encoder.embedding_dim
            else:
                MLP_time_dim = 1
                input_time_dim = 1
            self.flag = 0
        if use_time_encoder and not use_xyz_encoder:
            self.time_encoder, input_time_dim = get_encoder(time_encoder)
            self.time_MLP = tcnn.Network(
                n_input_dims=input_time_dim,
                n_output_dims=time_encoder.MLP_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": time_encoder.hidden_dim,
                    "n_hidden_layers": time_encoder.num_layers - 1,
                },
            )
            MLP_time_dim = time_encoder.MLP_dim
            input_xyz_dim = 3
            self.flag = 1
        if use_time_encoder and use_xyz_encoder:
            self.xyz_encoder, input_xyz_dim = get_encoder(xyz_encoder)
            if xyz_encoder.use_MLP:
                self.xyz_MLP = nn.Sequential(
                    nn.Linear(input_xyz_dim, xyz_encoder.hidden_dim), nn.LayerNorm(xyz_encoder.hidden_dim), 
                    nn.GELU(),
                    nn.Linear(xyz_encoder.hidden_dim, xyz_encoder.hidden_dim), nn.LayerNorm(xyz_encoder.hidden_dim),
                    nn.GELU(),
                    nn.Linear(xyz_encoder.hidden_dim, xyz_encoder.MLP_dim)
                )
                MLP_xyz_dim = xyz_encoder.MLP_dim
                input_xyz_dim = MLP_xyz_dim
                self.use_XYZ_MLP = True
            # input_xyz_dim = 3
            else:
                MLP_xyz_dim = input_xyz_dim
                self.use_XYZ_MLP = False

            # self.time_encoder, input_time_dim = get_encoder(time_encoder)
            input_time_dim = 1
            # self.time_MLP = tcnn.Network(
            #     n_input_dims=1,
            #     n_output_dims=time_encoder.MLP_dim,
            #     network_config={
            #         "otype": "FullyFusedMLP",
            #         "activation": "ReLU",
            #         "output_activation": "None",
            #         "n_neurons": time_encoder.hidden_dim,
            #         "n_hidden_layers": time_encoder.num_layers - 1,
            #     },
            # )
            self.time_MLP = nn.Sequential(
                nn.Linear(input_time_dim, time_encoder.hidden_dim), nn.LayerNorm(time_encoder.hidden_dim), 
                nn.GELU(),
                nn.Linear(time_encoder.hidden_dim, time_encoder.hidden_dim), nn.LayerNorm(time_encoder.hidden_dim),
                nn.GELU(),
                nn.Linear(time_encoder.hidden_dim, time_encoder.MLP_dim)
            )
            MLP_time_dim = time_encoder.MLP_dim
            self.flag = 2

        dims = [MLP_time_dim + MLP_xyz_dim] + list(dims)
        output_dim = 1

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.Layers = nn.ModuleList()

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - (MLP_time_dim+input_xyz_dim)
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

    def forward(self, time, xyz):
        """
        Args:
            lat_vecs: (N, L)
            xyz: (N, query_dim)
        Returns:
            x: (N,)
        """
        # import ipdb; ipdb.set_trace()
        assert(time.shape[0] == xyz.shape[0])
        assert(len(time.shape) == len(xyz.shape) == 2)
        xyz = (xyz + 1) / 2 # to [0, 1]
        if self.flag == 0:
            xyz_encoding = self.xyz_encoder(xyz)
            # xyz_MLP = self.xyz_MLP(xyz_encoding)
            time_encoding = time
            # under_int = time.int()
            # upper_int = under_int + 1
            # under_encoding = self.time_emb(under_int)[:, 0, :]
            # upper_encoding = self.time_emb(upper_int)[:, 0, :]
            # coe_under = (upper_int - time)
            # coe_upper = (time - under_int)
            # time_encoding = coe_under * under_encoding + coe_upper * upper_encoding
            x = torch.cat([time_encoding, xyz_encoding], dim=-1)
            # x = xyz_encoding
            x_input = x
        elif self.flag == 1:
            xyz_encoding = xyz
            time_encoding = self.time_encoder(time)
            time_MLP = self.time_MLP(time_encoding)
        elif self.flag == 2:
            if self.use_XYZ_MLP:
                xyz_encoding = self.xyz_encoder(xyz)
                xyz_encoding = self.xyz_MLP(xyz_encoding)
            else:
                xyz_encoding = self.xyz_encoder(xyz)
            # xyz_MLP = self.xyz_MLP(xyz_encoding)
            # time_encoding = self.time_encoder(time)
            time_encoding = time
            time_MLP = self.time_MLP(time_encoding)
            x = torch.cat([time_MLP, xyz_encoding], dim=-1)
            x_input = x
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
    

class SdfDecoder_freenerf(nn.Module):
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
        freq_mask_max_epoch=3000,
        **kwargs
    ):
        super().__init__()
        # import ipdb; ipdb.set_trace()
        self.freq_mask_max_epoch = freq_mask_max_epoch
        self.flag = 0
        if use_xyz_encoder and not use_time_encoder:
            self.xyz_encoder, input_xyz_dim = get_encoder(xyz_encoder)
            MLP_xyz_dim = input_xyz_dim
            MLP_time_dim = 1
            input_time_dim = 1
            self.flag = 0
        else:
            raise NotImplementedError

        dims = [MLP_time_dim + MLP_xyz_dim] + list(dims)
        output_dim = 1

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.Layers = nn.ModuleList()

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - (MLP_time_dim+input_xyz_dim)
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

    def forward(self, time, xyz, epoch):
        """
        Args:
            lat_vecs: (N, L)
            xyz: (N, query_dim)
        Returns:
            x: (N,)
        """
        # import ipdb; ipdb.set_trace()
        assert(time.shape[0] == xyz.shape[0])
        assert(len(time.shape) == len(xyz.shape) == 2)
        
        xyz = (xyz + 1) / 2 # to [0, 1]
        if self.flag == 0:
            # import ipdb; ipdb.set_trace()
            xyz_encoding = self.xyz_encoder(xyz)
            mask = torch.from_numpy(
                np.tile(get_freq_reg_mask(xyz_encoding.shape[1], epoch.detach().cpu().item(), self.freq_mask_max_epoch), (xyz_encoding.shape[0], 1))
                ).to(xyz_encoding.device).to(xyz_encoding.dtype)
            xyz_encoding = xyz_encoding * mask
            time_encoding = time
            x = torch.cat([time_encoding, xyz_encoding], dim=-1)
            # x = xyz_encoding
            x_input = x
        else:
            raise NotImplementedError

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
    

class SdfDecoder_Weightnorm(nn.Module):
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
        **kwargs
    ):
        super().__init__()
        # import ipdb; ipdb.set_trace()
        self.flag = 0
        if use_time_encoder and use_xyz_encoder:
            self.xyz_encoder, input_xyz_dim = get_encoder(xyz_encoder)
            if xyz_encoder.use_MLP:
                self.xyz_MLP = nn.Sequential(
                    nn.utils.weight_norm(nn.Linear(input_xyz_dim, xyz_encoder.hidden_dim)),
                    nn.ReLU(),
                    nn.utils.weight_norm(nn.Linear(xyz_encoder.hidden_dim, xyz_encoder.hidden_dim)),
                    nn.ReLU(),
                    nn.Linear(xyz_encoder.hidden_dim, xyz_encoder.MLP_dim)
                )
                MLP_xyz_dim = xyz_encoder.MLP_dim
                input_xyz_dim = MLP_xyz_dim
                self.use_XYZ_MLP = True
            # input_xyz_dim = 3
            else:
                MLP_xyz_dim = input_xyz_dim
                self.use_XYZ_MLP = False

            if time_encoder.use_MLP and time_encoder.use_freq:
                self.time_encoder, input_time_dim = get_encoder(time_encoder)

                self.time_MLP = nn.Sequential(
                    nn.utils.weight_norm(nn.Linear(input_time_dim, time_encoder.hidden_dim)), 
                    nn.ReLU(),
                    nn.utils.weight_norm(nn.Linear(time_encoder.hidden_dim, time_encoder.hidden_dim)),
                    nn.ReLU(),
                    nn.Linear(time_encoder.hidden_dim, time_encoder.MLP_dim)
                )
                
                MLP_time_dim = time_encoder.MLP_dim
                self.time_flag = 0
            elif time_encoder.use_MLP and not time_encoder.use_freq:
                input_time_dim = 1
                self.time_MLP = nn.Sequential(
                    nn.utils.weight_norm(nn.Linear(input_time_dim, time_encoder.hidden_dim)), 
                    nn.ReLU(),
                    nn.utils.weight_norm(nn.Linear(time_encoder.hidden_dim, time_encoder.hidden_dim)),
                    nn.ReLU(),
                    nn.Linear(time_encoder.hidden_dim, time_encoder.MLP_dim)
                )
                MLP_time_dim = time_encoder.MLP_dim
                self.time_flag = 1
            elif not time_encoder.use_MLP and time_encoder.use_freq:
                self.time_encoder, input_time_dim = get_encoder(time_encoder)
                MLP_time_dim = input_time_dim
                self.time_flag = 2

            self.flag = 2
        else:
            self.xyz_encoder, input_xyz_dim = get_encoder(xyz_encoder)
            MLP_xyz_dim = input_xyz_dim
            self.use_XYZ_MLP = False

            input_time_dim = 1
            MLP_time_dim = 1
            self.flag = 1

        dims = [MLP_time_dim + MLP_xyz_dim] + list(dims)
        output_dim = 1

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.Layers = nn.ModuleList()

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - (MLP_time_dim+input_xyz_dim)
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
        assert(time.shape[0] == xyz.shape[0])
        assert(len(time.shape) == len(xyz.shape) == 2)
        xyz = (xyz + 1) / 2 # to [0, 1]
        if self.flag == 2:
            if self.use_XYZ_MLP:
                xyz_encoding = self.xyz_encoder(xyz)
                xyz_encoding = self.xyz_MLP(xyz_encoding)
            else:
                xyz_encoding = self.xyz_encoder(xyz)
            if self.time_flag == 0:
                time_encoding = self.time_encoder(time)
                time_encoding = self.time_MLP(time_encoding)
            elif self.time_flag == 1:
                time_encoding = time
                time_encoding = self.time_MLP(time)
            elif self.time_flag == 2:
                time_encoding = self.time_encoder(time)
            else:
                time_encoding = time

            x = torch.cat([time_encoding, xyz_encoding], dim=-1)
            x_input = x
        elif self.flag == 1:
            xyz_encoding = self.xyz_encoder(xyz)
            time_encoding = time
            x = torch.cat([time_encoding, xyz_encoding], dim=-1)
            x_input = x
        else:
            raise NotImplementedError
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