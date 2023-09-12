from pysdf import SDF

# Load some mesh (don't necessarily need trimesh)
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData, PlyElement
import pyrender
import skimage
import torch
# import tqdm
import os

from topologylayer.nn import BatchedLevelSetLayer3D
from pytorch3d.loss import chamfer_distance

def get_ndpd(sdfs: torch.Tensor, min_persistence=0.0):
    # import ipdb; ipdb.set_trace()
    # sdf.shape: (B, meshgrid_size, meshgrid_size, meshgrid_size)
    dgms, _ = topology_PD_loss(sdfs) 
    res = []
    for dgm in dgms:
        inds = torch.all(torch.all(torch.isfinite(dgm), dim=2), dim=0)
        fh = dgm[:, inds, ...]
        diag = fh[:, :, 0] - fh[:, :, 1]
        res.append(fh[:, torch.any(diag> min_persistence, axis=0), ...])
    return res

def filter_diag(dgm, min_persistence=0.0):
    # import ipdb; ipdb.set_trace()
    diag = dgm[:, :, 0] - dgm[:, :, 1]
    return dgm[:, torch.any(diag > min_persistence, axis=0), :]

if __name__ == '__main__':
    base_dir = "/home/xiaoyan/3D/4DRecon/4DRep_DeepSDF3D/data/DeformingThings4D/animals/bear3EP_attack1"
    global_ply_dir = os.path.join(base_dir, "global_w_normal_obj")
    global_ply_paths = sorted(os.listdir(global_ply_dir))
    meshgrid_size = 128
    x = np.linspace(-1, 1, meshgrid_size)
    y = np.linspace(-1, 1, meshgrid_size)
    z = np.linspace(-1, 1, meshgrid_size)
    X, Y, Z = np.meshgrid(x, y, z)
    meshgrid = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)

    print('begin')
    topology_PD_loss = BatchedLevelSetLayer3D(size=(meshgrid_size,meshgrid_size,meshgrid_size), sublevel=False)

    # 创建一个 2x2 的子图网格
    # m = 3
    # n = 3
    # fig, axs = plt.subplots(m, n, figsize=(20, 20))
    meshgrid_sdfs = []
    for i in range(26):
        print(i)
        global_ply_path = global_ply_paths[i]
        # import ipdb; ipdb.set_trace()
        ply = trimesh.load(os.path.join(global_ply_dir, global_ply_path))
        f = SDF(ply.vertices, ply.faces); # (num_vertices, 3) and (num_faces, 3)
        meshgrid_sdf = f(meshgrid).reshape(meshgrid_size, meshgrid_size, meshgrid_size)
        meshgrid_sdfs.append(meshgrid_sdf)

    meshgrid_sdfs = np.stack(meshgrid_sdfs, axis=0)
    
    dgms = get_ndpd(torch.from_numpy(meshgrid_sdfs))
    
    # dgms = [filter_diag(dgm) for dgm in dgms]
    h0 = dgms[0].cpu().numpy()
    h1 = dgms[1].cpu().numpy()
    h2 = dgms[2].cpu().numpy()
    np.save("/home/xiaoyan/3D/4DRecon/4DRep_DeepSDF3D/data/DeformingThings4D/animals/bear3EP_attack1/GT_pd/GT_pd_h0.npy", h0)
    np.save("/home/xiaoyan/3D/4DRecon/4DRep_DeepSDF3D/data/DeformingThings4D/animals/bear3EP_attack1/GT_pd/GT_pd_h1.npy", h1)
    np.save("/home/xiaoyan/3D/4DRecon/4DRep_DeepSDF3D/data/DeformingThings4D/animals/bear3EP_attack1/GT_pd/GT_pd_h2.npy", h2)
    # axs[i//3, i%3].scatter(h0[0,:,0], h0[0,:,1], c='r', s=5, label='h0')
    # axs[i//3, i%3].scatter(h1[0,:,0], h1[0,:,1], c='b', s=5, label='h1')
    # axs[i//3, i%3].scatter(h2[0,:,0], h2[0,:,1], c='g', s=5, label='h2')
    # axs[i//3, i%3].legend()

    # plt.legend()
    plt.savefig("view_pd.png")

