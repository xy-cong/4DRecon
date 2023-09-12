"""
from anime to .ply(vertices & faces)
"""

import os
import numpy as np
from plyfile import PlyData, PlyElement
import glob
import open3d as o3d

def anime_read(filename):
    """
    filename: path of .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: vertice data of the 1st frame (3D positions in x-y-z-order)
        face_data: riangle face data of the 1st frame
        offset_data: 3D offset data from the 2nd to the last frame
    """
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    '''check data consistency'''
    if len(offset_data) != (nf - 1) * nv * 3:
        raise ("data inconsistent error!", filename)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data

def write_ply(filename, vertices, faces):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(vertices)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("element face {}\n".format(len(faces)))
        f.write("property list uchar int vertex_index\n")
        f.write("end_header\n")

        for v in vertices:
            f.write("{} {} {}\n".format(v[0], v[1], v[2]))

        for face in faces:
            f.write("{} ".format(len(face)) + " ".join([str(i) for i in face]) + "\n")

def calculate_area_weighted_normals(vertices, faces):
    normals = np.zeros(vertices.shape, dtype=vertices.dtype)
    for face in faces:
        v = vertices[face]
        normal = np.cross(v[1] - v[0], v[2] - v[0])
        area = np.linalg.norm(normal) / 2
        normal /= np.linalg.norm(normal)
        normals[face] += normal * area
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    return normals


def write_ply_pt_normal_face(filename, vertices, normals, faces):
    vertex_data = np.empty(len(vertices), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')
    ])
    vertex_data['x'] = vertices[:, 0]
    vertex_data['y'] = vertices[:, 1]
    vertex_data['z'] = vertices[:, 2]
    vertex_data['nx'] = normals[:, 0]
    vertex_data['ny'] = normals[:, 1]
    vertex_data['nz'] = normals[:, 2]

    # Create the face data
    face_data = np.empty(len(faces), dtype=[
        ('vertex_indices', 'i4', (3,)),
    ])
    face_data['vertex_indices'] = faces

    # Create the PlyElement instances
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    face_element = PlyElement.describe(face_data, 'face')

    # Write to a PLY file
    PlyData([vertex_element, face_element], text=True).write(filename)

def sample_pts_based_on_interior_confidence(all_pts, normals, confidences, d_max=0.1, m_t=10):
    """
    all_pts: scan 点
    normalL: scan 点对应的法向量
    confidence: 置信度, paper中的b, b越高意味着这个点是boundary的概率越低, 在附近可以采样的范围大, 反之...
    d_max: 可以采样的最大边界offset, max_sdf
    m_t: 每个样本点附近采多少点
    """
    ret = np.zeros((all_pts.shape[0], m_t, 8)) # (n, 8) x,y,z,n_x,n_y,n_z,sdf,confidence
    import ipdb; ipdb.set_trace()
    for idx in range(len(all_pts)):
        pt = all_pts[idx]
        normal = normals[idx]
        if np.linalg.norm(normal):
            normal = normal / np.linalg.norm(normal)  # 单位化法向量
        confidence = confidences[idx]
        psi = np.random.normal(0,1,m_t)
        sdfs_sample = psi*confidence*d_max
        pts_sample = pt + np.outer(sdfs_sample, normal)
        ret[idx, :, 0:3] = pts_sample
        ret[idx, :, 3:6] = normal
        ret[idx, :, 6] = sdfs_sample
        ret[idx, :, 7] = confidence
    return ret

def sample_pts_near_surface(all_pts, normals, d_max=0.1, m_t=10):
    """
    all_pts: scan 点
    normalL: scan 点对应的法向量
    d_max: 可以采样的最大边界offset, max_sdf
    m_t: 每个样本点附近采多少点
    """
    ret = np.zeros((all_pts.shape[0], m_t, 7)) # (n, 8) x,y,z,n_x,n_y,n_z,sdf
    # import ipdb; ipdb.set_trace()
    for idx in range(len(all_pts)):
        pt = all_pts[idx]
        normal = normals[idx]
        if np.linalg.norm(normal):
            normal = normal / np.linalg.norm(normal)  # 单位化法向量
        psi = np.random.normal(0,1,m_t)
        sdfs_sample = psi*d_max
        pts_sample = pt + np.outer(sdfs_sample, normal)
        ret[idx, :, 0:3] = pts_sample
        ret[idx, :, 3:6] = normal
        ret[idx, :, 6] = sdfs_sample
    return ret.reshape(-1, 7)

if __name__ == '__main__':
    anime_file = "DeformingThings4D/animals/bear3EP_attack1/bear3EP_attack1.anime"
    ply_save_dir = "DeformingThings4D/animals/bear3EP_attack1/plys_w_normal"
    # normal_save_dir = "DeformingThings4D/animals/bear3EP_attack1/normals"
    # m_t = 5
    # near_surface_pts_save_dir = f"/home/xiaoyan/3D/4DRep_DeepSDF3D/data/DeformingThings4D/animals/bear3EP_attack1/near_surface_pts_mt_{mt}"
    nf, nv, nt, vert_data, face_data, offset_data = anime_read(anime_file)
    # offset_data = np.concatenate ( [ np.zeros( (1, offset_data.shape[1], offset_data.shape[2]) ), offset_data], axis=0)
    # # import ipdb; ipdb.set_trace()
    filenames = sorted(glob.glob(os.path.join(ply_save_dir, "*.ply")))
    point_clouds = []
    normal_clouds = []
    for filename in filenames:
        print(filename)
        # vert_data_frame = vert_data + offset_data[frame]
        # # normals_frame = calculate_area_weighted_normals(vert_data_frame, face_data)
        # write_ply(os.path.join(ply_save_dir, f"{frame:05d}.ply"), vert_data_frame, face_data)
        # # write_ply_pt_normal_face(os.path.join(ply_save_dir, f"{frame:03d}.ply"), vert_data_frame, normals_frame, face_data)
        # # near_surface_pts = sample_pts_near_surface(vert_data_frame, normals_frame, d_max=0.1, m_t=m_t)
        # # np.save(os.path.join(normal_save_dir, f"{frame:03d}.npy"), normals_frame)
    
        cloud = o3d.io.read_point_cloud(filename)
        point_clouds.append(np.asarray(cloud.points))
        normal_clouds.append(np.asarray(cloud.normals))
    # import ipdb; ipdb.set_trace()
    point_clouds_num = point_clouds[0].shape[0]
    point_clouds_np = np.concatenate(point_clouds, axis=0)
    point_clouds_np -= np.mean(point_clouds_np, axis=0, keepdims=True)
    coord_max = np.amax(point_clouds_np)
    coord_min = np.amin(point_clouds_np)

    point_clouds_np = (point_clouds_np - coord_min) / (coord_max - coord_min)
    point_clouds_np -= 0.5
    point_clouds_np *= 2.

    point_clouds =[]
    for idx in range(len(filenames)):
        bgn = idx*point_clouds_num
        end = (idx+1)*point_clouds_num
        point_clouds.append(point_clouds_np[bgn:end, :])

    # point_clouds.append(point_clouds_np[bgn:, :])

    for curr_filename, curr_point_cloud, curr_normal_cloud in zip(filenames, point_clouds, normal_clouds):
        filename = curr_filename.split('/')[-1]
        print(filename)

        # 首先，我们需要将顶点和法线数据结合到一起
        # import ipdb; ipdb.set_trace()
        vertices_normals = [tuple(pt) + tuple(normal) for pt, normal in zip(curr_point_cloud, curr_normal_cloud)]

        vertex_normal_element = PlyElement.describe(np.array(vertices_normals, dtype=[
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('nx', 'f4'),
            ('ny', 'f4'),
            ('nz', 'f4')
        ]), 'vertex')

        # 然后，我们处理面数据
        faces = face_data.copy().tolist()
        for i in range(len(faces)):
            faces[i] = (faces[i],)

        ply_face = np.array(faces, dtype=[('vertex_indices', 'i4', (3,))])
        face_element = PlyElement.describe(ply_face, 'face')

        # 创建 PlyData 对象
        ply_data = PlyData([vertex_normal_element, face_element], text=False)


        # # 创建PlyData对象并写入文件
        # ply_data = PlyData([vertex_normal_element], text=True)
        ply_data.write(f'DeformingThings4D/animals/bear3EP_attack1/plys_global_w_normal/{filename}') 
