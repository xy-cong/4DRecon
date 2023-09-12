"""
给定partial views, 生成frame_surface_pts_normals and near_surface_pts_normals
"""

import numpy as np
from scipy.spatial import KDTree
import os
import sys
import math
import glob
import random
# import matplotlib  
# matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData, PlyElement


import open3d as o3d

def calculate_project_point(center_pt, normal, target_pt):
    """
    计算target_pt在center_pt的tangent平面上的投影点
    """
    normal = normal / np.linalg.norm(normal)  # 单位化法向量

    # 计算移动的距离
    d = np.dot(normal, center_pt - target_pt)

    # 计算q在切平面上的投影
    target_proj_pt = target_pt + d * normal
    
    return target_proj_pt


def find_k_nearest_pts_by_KDtree(query_pt, ret_num=100, bgn=20):
    """
    在all_pts中找到离query_pt最近的ret_num个点
    但要注意的是有一个bgn参数, 表示从all_pts离query_pt最近的的第bgn个点开始找, 更robust
    总之, 最后返回的是 [ bgn, bgn+ret_num ] 这个区间的点
    """

    # 创建k-d树 
    # kdtree = KDTree(all_pts)
    distances, indices = kdtree.query(query_pt, ret_num+bgn)

    # 这将返回最近的k个点的距离和在A中的索引
    # nearest_points = all_pts[indices[bgn:]]

    # return nearest_points
    return indices[bgn:]


def calculate_signed_angle(vec1, vec2, normal_vector):
    """
    计算两个向量之间的夹角，并根据旋转方向确定正负
    """
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cos_angle, -1, 1)) # 避免由于浮点误差导致的问题

    # 使用向量叉乘来确定旋转方向
    cross_product = np.cross(vec1, vec2)
    direction = np.dot(cross_product, normal_vector)

    if direction < 0:
        angle = -angle

    return angle # 返回角度（弧度制）

def sort_vectors_by_angle(vectors, normal_vector, base_vector_index=0):
    """
    以第一个向量为基准，计算其他所有向量关于这个向量的夹角，并按照夹角大小排序，得到一个排序后的新向量数组
    base_vector_index: 选取基准向量的索引: 目前使用的是random.randint(0,len(K_nearest_project_pts))-1, 但实际上取0足够了,反正是随机的
    """
    base_vector = vectors[base_vector_index] # 选取基准向量
    angles = [calculate_signed_angle(base_vector, v, normal_vector) for v in vectors] # 计算所有向量与基准向量的夹角
    import ipdb; ipdb.set_trace()
    sorted_indices = np.argsort(angles) # 获取排序后的索引
    sorted_vectors = vectors[sorted_indices]
    return sorted_vectors

def clip_angle_between_minus_pi_and_pi(angle):
    assert (angle>= -np.pi*2 and angle <= 2*np.pi)
    if angle > np.pi:
        return angle - 2*np.pi
    elif angle <= -np.pi:
        return angle + 2*np.pi
    else:
        return angle

def calculate_sorted_angles(vectors, normal_vector, base_vector_index=0):
    """
    以第一个向量为基准，计算其他所有向量关于这个向量的夹角，并按照夹角大小排序, 返回顺序的夹角
    base_vector_index: 选取基准向量的索引: 目前使用的是random.randint(0,len(K_nearest_project_pts))-1, 但实际上取0足够了,反正是随机的
    """
    base_vector = vectors[base_vector_index] # 选取基准向量
    angles = [calculate_signed_angle(base_vector, v, normal_vector) for v in vectors] # 计算所有向量与基准向量的夹角
    # import ipdb; ipdb.set_trace()
    sorted_angles = sorted(angles)
    return sorted_angles


def find_max_gap(A):
    # import ipdb; ipdb.set_trace()
    gap_0_and_lastone = A[-1] - A[0]
    gap = (np.array(A[1:]) - np.array(A[:-1])).tolist()  # 使用numpy数组计算所有相邻元素的差值
    gap.append(gap_0_and_lastone)
    gap = [abs(clip_angle_between_minus_pi_and_pi(angle)) for angle in gap]
    max_gap = np.max(gap)  # 使用numpy的max函数找到最大的差值
    return max_gap

def calculate_b_for_pt(all_pts, center_pt, normal, idx, sigma=np.pi/4):
    """
    计算center_pt的b
    """
    if np.linalg.norm(normal) == 0:
        return np.exp(-8)
    K_nearest_pts = K_neighbor_points[idx]
    K_nearest_project_pts = [calculate_project_point(center_pt, normal, pt) for pt in K_nearest_pts]
    K_nearest_project_vectors = K_nearest_project_pts - center_pt
    sorted_angles = calculate_sorted_angles(K_nearest_project_vectors, normal, base_vector_index=random.randint(0,len(K_nearest_project_pts)-1))
    max_gap = find_max_gap(sorted_angles)
    b = np.exp( - (max_gap ** 2) / (2*(sigma**2)))
    return b

def sample_pts_based_on_interior_confidence(all_pts, normals, confidences, d_max=0.05, m_t=10):
    """
    all_pts: scan 点
    normalL: scan 点对应的法向量
    confidence: 置信度, paper中的b, b越高意味着这个点是boundary的概率越低, 在附近可以采样的范围大, 反之...
    d_max: 可以采样的最大边界offset, max_sdf
    m_t: 每个样本点附近采多少点
    """
    ret = np.zeros((all_pts.shape[0], m_t, 8)) # (n, 8) x,y,z,n_x,n_y,n_z,sdf,confidence
    # import ipdb; ipdb.set_trace()
    for idx in range(len(all_pts)):
        pt = all_pts[idx]
        normal = normals[idx]
        if np.linalg.norm(normal):
            normal = normal / np.linalg.norm(normal)  # 单位化法向量
        confidence = confidences[idx] # (-1,1)
        psi = np.random.uniform(0,1,m_t) # (-1,1)
        sdfs_sample = psi*confidence*d_max # (-d_max, d_max)
        pts_sample = pt + np.outer(sdfs_sample, normal)
        ret[idx, :, 0:3] = pts_sample
        ret[idx, :, 3:6] = normal
        ret[idx, :, 6] = sdfs_sample
        ret[idx, :, 7] = confidence
    return ret


def get_pointcloud_files(path):
    files = list()
    for f in glob.glob(path + '/*.ply'):
        files.append(f)
    return files

def get_pointcloud_from_file(path, filename):
    cloud = o3d.io.read_point_cloud(path + '/' + filename)
    return cloud

def get_transformations_from_file(path, filename):
    with open(path + '/' + filename) as f:
        lines = (line for line in f)
        source = np.loadtxt(lines, delimiter=' ', skiprows=1, dtype='str')
        source = np.delete(source, 0, 1)  #remove camera
        
        filenames = source[:,0]
        source = source[filenames.argsort()]
        filenames = np.sort(filenames)
        
        translations = list()
        for row in source[:,1:4]:
            translations.append(np.reshape(row, [3,1]).astype(np.float32))
        
        quaternions = list()
        for row in source[:,4:]:
            quaternions.append(np.reshape(row, [4,1]).astype(np.float32))    
        
    return filenames, translations, quaternions

def quaternion_rotation_matrix(Q):

    # Extract the values from Q
    q0 = Q[3]
    q1 = Q[0]
    q2 = Q[1]
    q3 = Q[2]
    
    # calculate unit quarternion
    magnitude = math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
    
    q0 = q0 / magnitude
    q1 = q1 / magnitude
    q2 = q2 / magnitude
    q3 = q3 / magnitude
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
    
    rot_matrix = np.transpose(rot_matrix)
                            
    return rot_matrix

def sample_vis(save_path, pts_normals):
    
    # 将数据分解为点和法线
    points = pts_normals[:, :3]
    normals = pts_normals[:, 3:]

    # 创建一个包含点和法线的结构化数组
    vertex = np.zeros(pts_normals.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])

    # 将数据填充到结构化数组中
    for i in range(pts_normals.shape[0]):
        vertex[i] = (points[i, 0], points[i, 1], points[i, 2], normals[i, 0], normals[i, 1], normals[i, 2])

    # 创建一个PlyElement实例
    vertex_element = PlyElement.describe(vertex, 'vertex')

    # 将PlyElement实例保存为PLY文件
    PlyData([vertex_element], text=True).write(save_path)


def save_partial_frames(frame, surface_pts_normals):
    np.save(os.path.join(path, f'partial_frames/{frame:04d}.npy'), surface_pts_normals)
    # 将数据分解为点和法线
    points = surface_pts_normals[:, :3]
    normals = surface_pts_normals[:, 3:]

    # 创建一个包含点和法线的结构化数组
    vertex = np.zeros(surface_pts_normals.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])

    # 将数据填充到结构化数组中
    for i in range(surface_pts_normals.shape[0]):
        vertex[i] = (points[i, 0], points[i, 1], points[i, 2], normals[i, 0], normals[i, 1], normals[i, 2])

    # 创建一个PlyElement实例
    vertex_element = PlyElement.describe(vertex, 'vertex')

    # 将PlyElement实例保存为PLY文件
    PlyData([vertex_element], text=True).write(os.path.join(path, f'partial_frames_vis/{frame:04d}.ply'))
    
if __name__=="__main__": # $python visualization_bunny.py bunny/data

    path = sys.argv[1]

    filenames = sorted(os.listdir(os.path.join(path, 'plys_global_w_normal')))
    num_frame = len(filenames)
    viewpoints_dir = os.path.join(path, 'viewpoint')
    viewpoints = sorted(os.listdir(viewpoints_dir))
    viewpoints = viewpoints + viewpoints[360:300:-1] + viewpoints[180:120:-1]
    frame_viewpoints = []
    views_per_frame = len(viewpoints) // num_frame
    for i in range(num_frame):
        frame_viewpoints.append(viewpoints[i*views_per_frame:(i+1)*views_per_frame])
    
    surface_points_normals = []

    for frame in range(num_frame):
        frame_viewpoint = frame_viewpoints[frame] # (views_per_frame, )
        frame_points = []
        frame_normals = []
        for viewpoint in frame_viewpoint:
            cloud = o3d.io.read_point_cloud(os.path.join(viewpoints_dir, viewpoint, f'partial_plys/{frame:05d}.ply'))
            frame_points.append(np.asarray(cloud.points))
            frame_normals.append(np.asarray(cloud.normals))
        # import ipdb; ipdb.set_trace()
        frame_points = np.concatenate(frame_points, axis=0)
        frame_normals = np.concatenate(frame_normals, axis=0)
        frame_pts_normals = np.concatenate((frame_points, frame_normals), axis=1)
        frame_pts_normals = np.unique(frame_pts_normals, axis=0)
        surface_points_normals.append(frame_pts_normals)
        save_partial_frames(frame, frame_pts_normals)

    # import ipdb; ipdb.set_trace()
    for idx in range(len(surface_points_normals)):
        print(idx)
        test_point_cloud = surface_points_normals[idx][:, :3]
        test_normal_cloud = surface_points_normals[idx][:, 3:6]
        kdtree = KDTree(test_point_cloud)
        # 查询每个点的最近的101个点
        distances, indices = kdtree.query(test_point_cloud, k=121)

        # 使用找到的索引来获取邻居的坐标，丢弃第一个邻居（即自身）
        K_neighbor_points = test_point_cloud[indices[:, 21:]]

        test_point_cloud_b = [calculate_b_for_pt(test_point_cloud, pt, normal, i) for i, (pt, normal) in enumerate(zip(test_point_cloud, test_normal_cloud))]
        test_point_cloud_b = np.array(test_point_cloud_b)

        # import ipdb; ipdb.set_trace()
        save_cloud = sample_pts_based_on_interior_confidence(test_point_cloud, test_normal_cloud, test_point_cloud_b, m_t=30)
        np.save(path + '/near_surface_samples_30/' + f'{idx:04d}.npy', save_cloud)
        sample_vis(path + '/near_surface_samples_30_vis/' + f'{idx:04d}.ply', save_cloud.reshape(-1, 8)[:,:6])
