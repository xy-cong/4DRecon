import numpy as np
from scipy.spatial import KDTree
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

if __name__=="__main__": # $python visualization_bunny.py bunny/data

    path = sys.argv[1]
    
    # load transformations and filenames from file
    filenames, translations, quaternions = get_transformations_from_file(path, 'bun.conf')
    
    curr_transformation = np.zeros([3,4])

    point_clouds = []
    normal_clouds = []
    for curr_filename, curr_quaternion, curr_translation in zip(filenames, quaternions, translations):  # go through input files
        
        curr_cloud = get_pointcloud_from_file(path, curr_filename)
        
        # convert cloud to numpy
        curr_point_cloud = np.asarray(curr_cloud.points)
        curr_normal_cloud = np.asarray(curr_cloud.normals)
        
        # compute rotation matrix from quaternions
        curr_rotation_matr = quaternion_rotation_matrix(curr_quaternion)
        curr_rotation_matr = np.squeeze(curr_rotation_matr)
        curr_translation = np.squeeze(curr_translation)
        
        # create transformation matrix
        curr_transformation[:,0:3] = curr_rotation_matr
        curr_transformation[:,3] = curr_translation
        
        # transform current cloud
        for i in range(curr_point_cloud.shape[0]):
            # apply rotation
            curr_point = np.matmul(curr_rotation_matr, np.transpose(curr_point_cloud[i,:]))
            # apply translation
            curr_point = curr_point + curr_translation 

            curr_normal = np.matmul(curr_rotation_matr, np.transpose(curr_normal_cloud[i,:]))
            
            curr_point_cloud[i] = curr_point

            curr_normal_cloud[i] = curr_normal
            
        # add current cloud to list of clouds
        point_clouds.append(curr_point_cloud)
        normal_clouds.append(curr_normal_cloud)

        # # 假设 `surface_pts` 是顶点的 numpy 数组, shape = (N, 3)
        # # 并且 `normals` 是对应的法向量的 numpy 数组, shape = (N, 3)

        # # 转换顶点和法向量为可以作为一维数组处理的元组，然后将所有这些元组放入一个列表中
        # vertices_normals = [tuple(pt) + tuple(normal) for pt, normal in zip(curr_point_cloud, curr_normal_cloud)]

        # vertex_normal_element = PlyElement.describe(np.array(vertices_normals, dtype=[
        #     ('x', 'f4'),
        #     ('y', 'f4'),
        #     ('z', 'f4'),
        #     ('nx', 'f4'),
        #     ('ny', 'f4'),
        #     ('nz', 'f4')
        # ]), 'vertex')

        # # 创建PlyData对象并写入文件
        # ply_data = PlyData([vertex_normal_element], text=True)
        # ply_data.write(f'bunny_normal/global_data/{curr_filename}')
    import ipdb; ipdb.set_trace()
    point_clouds_num = [point_cloud.shape[0] for point_cloud in point_clouds]
    point_clouds_np = np.concatenate(point_clouds, axis=0)
    point_clouds_np -= np.mean(point_clouds_np, axis=0, keepdims=True)
    coord_max = np.amax(point_clouds_np)
    coord_min = np.amin(point_clouds_np)

    point_clouds_np = (point_clouds_np - coord_min) / (coord_max - coord_min)
    point_clouds_np -= 0.5
    point_clouds_np *= 2.

    point_clouds =[]
    bgn = 0
    end = point_clouds_num[0]
    for i in range(1, len(point_clouds_num)):
        point_clouds.append(point_clouds_np[bgn:end, :])
        bgn = end
        end += point_clouds_num[i]
    point_clouds.append(point_clouds_np[bgn:, :])

    for curr_filename, curr_point_cloud, curr_normal_cloud in zip(filenames, point_clouds, normal_clouds):
        vertices_normals = [tuple(pt) + tuple(normal) for pt, normal in zip(curr_point_cloud, curr_normal_cloud)]

        vertex_normal_element = PlyElement.describe(np.array(vertices_normals, dtype=[
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('nx', 'f4'),
            ('ny', 'f4'),
            ('nz', 'f4')
        ]), 'vertex')

        # 创建PlyData对象并写入文件
        ply_data = PlyData([vertex_normal_element], text=True)
        ply_data.write(f'bunny_normal/global_data/{curr_filename}') 


def calculate_project_point(center_pt, normal, target_pt):
    """
    计算target_pt在center_pt的tangent平面上的投影点
    """
    normal = normal / normal.linalg.norm(np)  # 单位化法向量

    # 计算移动的距离
    d = np.dot(normal, center_pt - target_pt)

    # 计算q在切平面上的投影
    target_proj_pt = target_pt + d * normal
    
    return target_proj_pt


def find_k_nearest_pts_by_KDtree(all_pts, query_pt, ret_num=100, bgn=20):
    """
    在all_pts中找到离query_pt最近的ret_num个点
    但要注意的是有一个bgn参数, 表示从all_pts离query_pt最近的的第bgn个点开始找, 更robust
    总之, 最后返回的是 [ bgn, bgn+ret_num ] 这个区间的点
    """

    # 创建k-d树 
    kdtree = KDTree(all_pts)
    distances, indices = kdtree.query(query_pt, ret_num+bgn)

    # 这将返回最近的k个点的距离和在A中的索引
    nearest_points = all_pts[indices[bgn:]]

    return nearest_points


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
    base_vector_index: 选取基准向量的索引: 目前使用的是random.choice(len(K_nearest_project_pts)), 但实际上取0足够了,反正是随机的
    """
    base_vector = vectors[base_vector_index] # 选取基准向量
    angles = [calculate_signed_angle(base_vector, v, normal_vector) for v in vectors] # 计算所有向量与基准向量的夹角
    sorted_indices = np.argsort(angles) # 获取排序后的索引
    sorted_vectors = vectors[sorted_indices] # 获取排序后的向量数组
    return sorted_vectors

def find_max_gap(A):
    assert type(A) == np.ndarray
    gap = np.array(A[1:]) - np.array(A[:-1])  # 使用numpy数组计算所有相邻元素的差值
    max_gap = np.max(gap)  # 使用numpy的max函数找到最大的差值
    return max_gap

def calculate_b_for_pt(all_pts, center_pt, normal, sigma=np.pi/4):
    """
    计算center_pt的b
    """
    K_nearest_value = 100
    robust_thresh = 20
    K_nearest_pts = find_k_nearest_pts_by_KDtree(all_pts, center_pt, K_nearest_value, robust_thresh)
    K_nearest_project_pts = [calculate_project_point(center_pt, normal, pt) for pt in K_nearest_pts]
    K_nearest_project_vectors = K_nearest_project_pts - center_pt
    sorted_vectors = sort_vectors_by_angle(K_nearest_project_vectors, normal, base_vector_index=random.choice(len(K_nearest_project_pts)))
    max_gap = find_max_gap(sorted_vectors)
    b = np.exp(- (max_gap ** 2) / (2*(sigma**2)))
    return b

