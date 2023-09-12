# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional

def compute_neigh(faces):
    '''
    Args:
        faces: (m, 3)
    '''
    neigh = torch.cat( (faces[:, [0, 1]], faces[:, [0, 2]], faces[:, [1, 2]]), 0)
    return neigh.long()


def arap_exact(vert_diff_t, vert_diff_0, neigh, n_vert):
    device = vert_diff_t.device
    S_neigh = torch.bmm(vert_diff_t.unsqueeze(2), vert_diff_0.unsqueeze(1))

    S = torch.zeros([n_vert, 3, 3], device=device, dtype=torch.float32)

    S = torch.index_add(S, 0, neigh[:, 0], S_neigh)
    S = torch.index_add(S, 0, neigh[:, 1], S_neigh)

    U, _, V = torch.svd(S.cpu(), compute_uv=True)

    U = U.to(device)
    V = V.to(device)

    R = torch.bmm(U, V.transpose(1, 2))

    Sigma = torch.ones((R.shape[0], 1, 3), device=device, dtype=torch.float32)
    Sigma[:, :, 2] = torch.det(R).unsqueeze(1)

    R = torch.bmm(U * Sigma, V.transpose(1, 2))

    return R


def arap_energy_exact(vert_t, vert_0, neigh, lambda_reg_len=1e-6):
    n_vert = vert_t.shape[0]

    vert_diff_t = vert_t[neigh[:, 0], :] - vert_t[neigh[:, 1], :]
    vert_diff_0 = vert_0[neigh[:, 0], :] - vert_0[neigh[:, 1], :]

    R_t = arap_exact(vert_diff_t, vert_diff_0, neigh, n_vert) # (V, 3, 3)

    R_neigh_t = 0.5 * (
        torch.index_select(R_t, 0, neigh[:, 0])
        + torch.index_select(R_t, 0, neigh[:, 1])
    ) # (E, 3, 3)

    vert_diff_0_rot = torch.bmm(R_neigh_t, vert_diff_0.unsqueeze(2)).squeeze()
    acc_t_neigh = vert_diff_t - vert_diff_0_rot

    E_arap = acc_t_neigh.norm() ** 2 + lambda_reg_len * (vert_t - vert_0).norm() ** 2

    return E_arap


if __name__ == "__main__":
    import trimesh
    mesh = trimesh.load("/scratch/cluster/yanght/Projects/ShapeCorres/Dataset/Human/Human_hybrid/raw/mesh_nml/test/100080/100080.obj")
    faces = torch.from_numpy(mesh.faces).cuda()
    verts_0 = torch.from_numpy(mesh.vertices).float().cuda()
    verts_t = verts_0 + torch.rand(verts_0.shape).cuda() * 0.01

    neigh = compute_neigh(faces)
    E_arap = arap_energy_exact(verts_t, verts_0, neigh)

    from IPython import embed; embed()

    print("main of arap_potential.py")
