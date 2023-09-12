from pysdf import SDF

# Load some mesh (don't necessarily need trimesh)
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData, PlyElement
from mesh_to_sdf import mesh_to_voxels
from mesh_to_sdf import sample_sdf_near_surface

import skimage

meshgrid_size = 64
x = np.linspace(-1, 1, meshgrid_size)
y = np.linspace(-1, 1, meshgrid_size)
z = np.linspace(-1, 1, meshgrid_size)
X, Y, Z = np.meshgrid(x, y, z)
meshgrid = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)


o = trimesh.load('/home/xiaoyan/3D/4DRecon/4DRep_DeepSDF3D/data/DeformingThings4D/animals/bear3EP_attack1/plys_global_w_normal/00000.ply')
f = SDF(o.vertices, o.faces); # (num_vertices, 3) and (num_faces, 3)
random_surface_points = f.sample_surface(10000)

# ------------------------------------------- mesh_to_sdf ------------------------------------------- #
# voxels = mesh_to_voxels(mesh, 64, pad=True)
points, sdf = sample_sdf_near_surface(o, number_of_points=250000)
# ------------------------------------------- mesh_to_sdf ------------------------------------------- #

# Compute some SDF values (negative outside);
# takes a (num_points, 3) array, converts automatically
# origin_sdf = f([0, 0, 0])
# sdf_multi_point = f([[0, 0, 0],[1,1,1],[0.1,0.2,0.2]])
meshgrid_sdf = f(meshgrid)[..., None]
ret = np.concatenate((meshgrid, meshgrid_sdf), axis=1)
np.save("bear3EP_attack1_frame0.npy", ret)


# # 创建一个新的图形
# fig = plt.figure()

# # 添加3D子图
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(random_surface_points[:,0], random_surface_points[:,1], random_surface_points[:, 2], c='r', s=1)
# # plt.scatter(meshgrid[:,0], meshgrid[:,1], c=meshgrid_sdf[:,0], s=1)
# plt.savefig("bear3EP_attack1_frame0.png")


# 将点的数据转换为PLY格式
vertex = np.array([tuple(point) for point in random_surface_points],
                  dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

# 创建PLY元素
vertex_element = PlyElement.describe(vertex, 'vertex')

# 保存为PLY文件
PlyData([vertex_element], text=True).write('output.ply')

import ipdb; ipdb.set_trace()
