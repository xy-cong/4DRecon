#!/usr/bin/env python
# coding=utf-8

import os
import igl
import trimesh
import numpy as np
from collections import defaultdict
import pywavefront
import matplotlib.pyplot as plt

def extract_boundary(faces):
    """
    Args:
        faces: (F, 3)
    """
    assert (np.min(faces) == 0), "obj mesh face vertex index starts from 1"
    edge_dict = defaultdict(float)
    for fid, face in enumerate(faces):
        for sid, tid in zip([0, 1, 2], [1, 2, 0]):
            assert((face[sid], face[tid]) not in edge_dict)
            edge_dict[(face[sid], face[tid])] = 1
            if (face[tid], face[sid]) in edge_dict:
                edge_dict[(face[sid], face[tid])] += 1
                edge_dict[(face[tid], face[sid])] += 1
    
    contour_list = []
    for key, val in edge_dict.items():
        if val == 1:
            contour_list.append(key)

    # Emperically find that this way directly gives the output
    contour_sort_list = sorted(contour_list, key=lambda x: x[0])[::-1]
    for i in range(1, len(contour_sort_list)):
        assert(contour_sort_list[i][0] == contour_sort_list[i - 1][1])
    return contour_sort_list


def sample_2d_points_near_boundary(V, E, N_inp, band_width=0.05):
    """ sample N points near the boundary
    Args:
        V: (n, 2)
        E: (m, 2)
        N_inp: number of points to be sampled, the returned num_of_points is N_inp // m * m
    """
    m = E.shape[0]
    V = V[:, :2]
    assert(V.shape[1] == E.shape[1] == 2)
    n_per_edge = int(N_inp / m)

    Pe_list = []
    Se_list = []

    e_vec = V[E[:, 1]] - V[E[:, 0]] # (m, 2)
    e_tan = e_vec / np.linalg.norm(e_vec, axis=-1, keepdims=True)
    e_tan = np.stack((-e_tan[:, 1], e_tan[:, 0]), axis=-1) # (m, 2)
    
    rand_vec = np.random.rand(m, n_per_edge) * 2 - 0.5 # (m, n_per_edge), in range [-0.5, 1.5]
    rand_tan = (np.random.rand(m, n_per_edge) * 2 - 1) * band_width # (m, n_per_edge)

    Pe = V[E[:, 0]][:, None, :] + e_vec[:, None, :] * rand_vec[:, :, None] + e_tan[:, None, :] * rand_tan[:, :, None] # (m, n_per_edge, 2)
    Pe = Pe.reshape(m * n_per_edge, 2)

    return Pe, Pe.shape[0]


def sample_2d_points(V, E, N=100000, near_bd_ratio=0.9, band_width=0.05):
    """
    Args:
        V: (N, 2), in range [-1, 1]
    """
    N_near_bd = int(N * near_bd_ratio)
    P_near_bd, N_near_bd = sample_2d_points_near_boundary(V, E, N_near_bd, band_width=band_width)

    N_uniform = N - N_near_bd
    P_uniform = np.random.rand(N_uniform, 2) * 2 - 1

    P = np.concatenate((P_near_bd, P_uniform), axis=0)
    return P



def compute_2d_sdf(P, V, F, E):
    assert(F.shape[1] == 3)
    assert(P.shape[1] == 2)
    assert(V.shape[1] == 2)
    P = np.concatenate((P, np.zeros_like(P[:, 0:1])), axis=-1)
    V = np.concatenate((V, np.zeros_like(V[:, 0:1])), axis=-1)
    D, _, _ = igl.point_mesh_squared_distance(P, V, E)
    S, _, _ = igl.signed_distance(P, V, F, return_normals=False)
    sign = (S > 1e-6).astype(np.float32) * 2 - 1
    sdf = np.sqrt(D) * sign
    return sdf



def read_obj(filename):
    scene = pywavefront.Wavefront(filename, collect_faces=True)
    vertices = np.array(scene.vertices)
    return vertices

def visible_contour(vertices, viewpoint):
    # 计算每个顶点相对于观察点的角度
    angles = np.arctan2(vertices[:,1] - viewpoint[1], vertices[:,0] - viewpoint[0])
    # 找到最左边和最右边的顶点
    left_vertex = vertices[np.argmin(angles)]
    right_vertex = vertices[np.argmax(angles)]
    return left_vertex, right_vertex

def plot_contour(vertices, viewpoint, left_vertex, right_vertex):
    plt.figure()
    plt.plot(vertices[:,0], vertices[:,1], 'b-')  # 轮廓线
    plt.plot([viewpoint[0], left_vertex[0]], [viewpoint[1], left_vertex[1]], 'r-')  # 观察点到最左边的射线
    plt.plot([viewpoint[0], right_vertex[0]], [viewpoint[1], right_vertex[1]], 'r-')  # 观察点到最右边的射线
    plt.plot(viewpoint[0], viewpoint[1], 'go')  # 观察点
    plt.show()


if __name__ == '__main__':

    mesh = trimesh.load('../data/simple_data/raw/mesh/0_0_0_0.obj', process=False)

    contour_sort_list = extract_boundary(np.asarray(mesh.faces))

    # 读取.obj文件
    vertices = read_obj('../data/simple_data/raw/mesh/0_0_0_0.obj')
    import ipdb; ipdb.set_trace()
    contour_index = [point[0] for point in contour_sort_list]
    contour_index.append(contour_index[0])
    vertices = vertices[contour_index]
    # 定义观察点
    viewpoint = np.array([0, -2])
    # 计算可见轮廓
    left_vertex, right_vertex = visible_contour(vertices, viewpoint)
    # 可视化
    plot_contour(vertices, viewpoint, left_vertex, right_vertex)


