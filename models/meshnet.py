import torch
import torch.nn as nn
import torch.nn.functional as F

from models.meshnet_base import MeshDecoder

class MeshNet(nn.Module):
    def __init__(self,
                 config,
                 edge_index,
                 down_transform,
                 up_transform,
                ):
        super().__init__()
        self.config = config
        self.decoder = MeshDecoder(edge_index=edge_index,
                                   down_transform=down_transform,
                                   up_transform=up_transform,
                                   **config.model.mesh)

    def forward(self, lat_vecs, data=None):
        '''
        Args:
            lat_vecs: (B, latent_dim)
        '''
        end_points = {}

        mesh_out_pred = self.decoder(lat_vecs) # (B, N, 3), normalized coordinates
        end_points["mesh_out_pred"] = mesh_out_pred
        # mesh_verts_pred = mesh_out_pred * self.std + self.mean # (B, N, 3)

        return end_points



