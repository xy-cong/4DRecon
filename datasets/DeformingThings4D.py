import os
import os.path as osp
import glob
import pickle

import torch
import open3d as o3d
import numpy as np
from omegaconf import OmegaConf
from loguru import logger
from sklearn.decomposition import PCA
import trimesh
import pysdf

import sys
sys.path.append('../')
from pyutils import *


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]
def anime_read( filename):
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

def sample_pts_based_on_interior_confidence(all_pts, normals, confidences, d_max=0.1, m_t=10):
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

def normalize_mashed(mesh_paths):
    """
    globally normalize meshes to [-1, 1]
    based on min/max of all meshes
    """
    ret_meshes = []
    mesh_vertices = []
    for mesh_path in mesh_paths:
        mesh = trimesh.load(mesh_path, force='mesh')

        # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
        vs = np.asarray(mesh.vertices)
        mesh_vertices.append(vs)

    mesh_vertices = np.concatenate(mesh_vertices, axis=0)
    vmin = mesh_vertices.min(0)
    vmax = mesh_vertices.max(0)
    v_center = (vmin + vmax) / 2
    v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
    for mesh_path in mesh_paths:
        mesh = trimesh.load(mesh_path, force='mesh')
        vs = mesh.vertices
        vs = (vs - v_center[None, :]) * v_scale
        mesh.vertices = vs
        ret_meshes.append(mesh)
    return ret_meshes


class DeformingThings4D_Partial(torch.utils.data.Dataset):

    def __init__(self, 
                 mode,
                 rep,
                 data_dir,
                 exp_name,
                 max_frames,
                 num_samples,
                 sample_mode,
                 near_surface_num_samples,
                 off_num_samples,
                 data_term, 
                 **kwargs):
        '''
        Args:
            mesh_dir: raw mesh dir
            sdf_dir: raw sdf dir
            num_samples: num of samples used to train sdfnet
        '''
        super().__init__()

        self.rep = rep
        self.mode = mode
        if self.mode == 'train':
            split = 'train'
        elif self.mode == 'test':
            split = 'test'
        else: 
            raise ValueError('invalid mode')
        self.exp_name = exp_name
        
        # plys_w_normal = sorted(glob.glob(os.path.join(data_dir, exp_name, 'plys_global_w_normal', '*.ply')))
        # self.num_data = min(max_frames, len(plys_w_normal)) 

        # viewpoints_dir = os.path.join(data_dir, exp_name, 'viewpoint')
        # viewpoints = sorted(os.listdir(viewpoints_dir))
        # viewpoints = viewpoints + viewpoints[360:300:-1] + viewpoints[180:120:-1]
        # frame_viewpoints = []
        # views_per_frame = len(viewpoints) // self.num_data
        # for i in range(self.num_data):
        #     frame_viewpoints.append(viewpoints[i*views_per_frame:(i+1)*views_per_frame])
        
        # self.surface_points_normals = []
    
        # for frame in range(self.num_data):
        #     frame_viewpoint = frame_viewpoints[frame] # (views_per_frame, )
        #     frame_points = []
        #     frame_normals = []
        #     for viewpoint in frame_viewpoint:
        #         cloud = o3d.io.read_point_cloud(os.path.join(viewpoints_dir, viewpoint, f'partial_plys/{frame:05d}.ply'))
        #         frame_points.append(np.asarray(cloud.points))
        #         frame_normals.append(np.asarray(cloud.normals))
        #     # import ipdb; ipdb.set_trace()
        #     frame_points = np.concatenate(frame_points, axis=0)
        #     frame_normals = np.concatenate(frame_normals, axis=0)
        #     frame_pts_normals = np.concatenate((frame_points, frame_normals), axis=1)
        #     frame_pts_normals = np.unique(frame_pts_normals, axis=0)
        #     self.surface_points_normals.append(frame_pts_normals)
            # self.__vis__(frame, frame_pts_normals)

        # ---------------------------------------------------------------------  after data preprocessing ---------------------------------------------------------------------
        self.surface_points_normals = []
        partial_frames_dir = sorted(glob.glob(os.path.join(data_dir, exp_name, 'partial_frames', '*.npy')))
        self.num_data = len(partial_frames_dir)
        for partial_frames in partial_frames_dir:
            frame_pts_normals = np.load(partial_frames)
            self.surface_points_normals.append(frame_pts_normals)
        # import ipdb; ipdb.set_trace()
        logger.info(f"dataset mode = {mode}")
        logger.info(f"dataset split = {split}")
        logger.info(f"dataset len = {self.num_data}\n")

        self.data_dir = data_dir
        self.num_samples = num_samples
        self.sample_mode = sample_mode
        self.near_surface_num_samples = near_surface_num_samples
        self.off_num_samples = off_num_samples
        self.data_term = data_term

        if self.sample_mode:
            samples_dir = os.path.join(data_dir, exp_name, 'near_surface_samples_30')
            self.near_surface_points_normals_sdfs = []
            for frame in range(self.num_data):
                samples = np.load(os.path.join(samples_dir, f'{frame:04d}.npy'))[:, :, :-1].reshape(-1, 7)
                self.near_surface_points_normals_sdfs.append(samples)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        
        data_dict = {}
        surface_pts_normal_sdf = self.surface_points_normals[idx]
        surface_pts_normal_sdf = np.concatenate((surface_pts_normal_sdf, np.zeros((surface_pts_normal_sdf.shape[0], 1))), axis=1)
    
        # import ipdb; ipdb.set_trace()
        # surface_pts_normal_sdf = np.concatenate((surface_points, surface_normals, np.zeros((surface_points.shape[0], 1))), axis=1)
        select_surface_ids = np.random.choice(surface_pts_normal_sdf.shape[0], self.num_samples)
        surface_pts_normal_sdf = surface_pts_normal_sdf[select_surface_ids, :]
        if self.sample_mode:
            near_surface_point_normal_sdf = self.near_surface_points_normals_sdfs[idx]
            assert surface_pts_normal_sdf.shape[1] == near_surface_point_normal_sdf.shape[1]
            select_near_surface_ids = np.random.choice(near_surface_point_normal_sdf.shape[0], self.near_surface_num_samples*self.num_samples)
            near_surface_point_normal_sdf = near_surface_point_normal_sdf[select_near_surface_ids, :]
            pts_normal_sdf = np.concatenate((surface_pts_normal_sdf, near_surface_point_normal_sdf), axis=0)
        else:
            pts_normal_sdf = surface_pts_normal_sdf

        sdf_pts = pts_normal_sdf[:, 0:3]
        sdf_normals = pts_normal_sdf[:, 3:6]
        sdf_pts_sdf = pts_normal_sdf[:, 6:]

        off_surface_normals = np.ones((self.off_num_samples, 3)) * -1.0
        normals = np.concatenate((sdf_normals, off_surface_normals), axis=0)
        off_surface_points = np.random.uniform(-1, 1, size=(self.off_num_samples, 3)).astype(np.float32)
        points_samples = np.concatenate((sdf_pts, off_surface_points), axis=0)
        off_surface_points_sdf = -1 * np.ones((self.off_num_samples, 1))
        sdf_samples = np.concatenate((sdf_pts_sdf, off_surface_points_sdf), axis=0)

        data_dict['point_samples'] = points_samples.astype(np.float32)
        data_dict['sdf_gt'] = sdf_samples.astype(np.float32)
        data_dict['normal_gt'] = normals.astype(np.float32)
        data_dict['idx'] = idx
        data_dict['time'] = np.array( ( idx ) / (self.num_data - 1)).astype(np.float32) # (1, )

        return data_dict

    def __vis__(self, frame, surface_pts_normals):

        from plyfile import PlyData, PlyElement
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
        PlyData([vertex_element], text=True).write(f'data/test_vis/{frame:04d}.ply')

class DeformingThings4D_Complete(torch.utils.data.Dataset):

    def __init__(self, 
                 mode,
                 rep,
                 data_dir,
                 exp_name,
                 num_samples,
                 sample_mode,
                 near_surface_num_samples,
                 off_num_samples,
                 data_term,
                 **kwargs):
        '''
        Args:
            mesh_dir: raw mesh dir
            sdf_dir: raw sdf dir
            num_samples: num of samples used to train sdfnet
        '''
        super().__init__()

        self.rep = rep
        self.mode = mode
        if self.mode == 'train':
            split = 'train'
        elif self.mode == 'test':
            split = 'test'
        else: 
            raise ValueError('invalid mode')
        self.exp_name = exp_name
        # import ipdb; ipdb.set_trace()
        plys_w_normal = sorted(glob.glob(os.path.join(data_dir, exp_name, 'plys_global_w_normal', '*.ply')))
        self.num_data = len(plys_w_normal)
        self.surface_points = []
        self.surface_normals = []
        # import ipdb; ipdb.set_trace()
        for ply in plys_w_normal:
            cloud = o3d.io.read_point_cloud(ply)
            self.surface_points.append(np.asarray(cloud.points))
            self.surface_normals.append(np.asarray(cloud.normals))
        
        logger.info(f"dataset mode = {mode}")
        logger.info(f"dataset split = {split}")
        logger.info(f"dataset len = {self.num_data}\n")
        logger.info(f"dataset points num per frame = {self.surface_points[0].shape[0]}\n")

        self.data_dir = data_dir
        self.num_samples = num_samples
        self.sample_mode = sample_mode
        self.near_surface_num_samples = near_surface_num_samples
        self.off_num_samples = off_num_samples
        self.data_term = data_term

        if self.sample_mode:
            samples_dir = os.path.join(data_dir, exp_name, 'samples_complete_10')
            self.near_surface_points_normals_sdfs = []
            for frame in range(self.num_data):
                samples = np.load(os.path.join(samples_dir, f'{frame:05d}.npy'))[:, :, :-1].reshape(-1, 7)
                self.near_surface_points_normals_sdfs.append(samples)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        
        data_dict = {}
        surface_points = self.surface_points[idx]
        surface_normals = self.surface_normals[idx]
    
        # import ipdb; ipdb.set_trace()
        surface_pts_normal_sdf = np.concatenate((surface_points, surface_normals, np.zeros((surface_points.shape[0], 1))), axis=1)
        select_surface_ids = np.random.choice(surface_pts_normal_sdf.shape[0], self.num_samples)
        surface_pts_normal_sdf = surface_pts_normal_sdf[select_surface_ids, :]
        if self.sample_mode:
            near_surface_point_normal_sdf = self.near_surface_points_normals_sdfs[idx]
            assert surface_pts_normal_sdf.shape[1] == near_surface_point_normal_sdf.shape[1]
            select_near_surface_ids = np.random.choice(near_surface_point_normal_sdf.shape[0], self.near_surface_num_samples*self.num_samples)
            near_surface_point_normal_sdf = near_surface_point_normal_sdf[select_near_surface_ids, :]
            pts_normal_sdf = np.concatenate((surface_pts_normal_sdf, near_surface_point_normal_sdf), axis=0)
        else:
            pts_normal_sdf = surface_pts_normal_sdf

        sdf_pts = pts_normal_sdf[:, 0:3]
        sdf_normals = pts_normal_sdf[:, 3:6]
        sdf_pts_sdf = pts_normal_sdf[:, 6:]

        off_surface_normals = np.ones((self.off_num_samples, 3)) * -1.0
        normals = np.concatenate((sdf_normals, off_surface_normals), axis=0)
        off_surface_points = np.random.uniform(-1, 1, size=(self.off_num_samples, 3)).astype(np.float32)
        points_samples = np.concatenate((sdf_pts, off_surface_points), axis=0)
        off_surface_points_sdf = -1 * np.ones((self.off_num_samples, 1))
        sdf_samples = np.concatenate((sdf_pts_sdf, off_surface_points_sdf), axis=0)

        data_dict['point_samples'] = points_samples.astype(np.float32)
        data_dict['sdf_gt'] = sdf_samples.astype(np.float32)
        data_dict['normal_gt'] = normals.astype(np.float32)
        data_dict['idx'] = idx
        data_dict['time'] = np.array( ( idx ) / (self.num_data - 1)).astype(np.float32) # (1, )
        
        # self.__vis__(filename, sdf_pts[:, :])
        return data_dict

class DeformingThings4D_GT(torch.utils.data.Dataset):

    def __init__(self, 
                 mode,
                 rep,
                 data_dir,
                 exp_name,
                 num_samples,
                 sample_mode,
                 near_surface_num_samples,
                 off_num_samples,
                 data_term,
                 **kwargs):
        '''
        Args:
            mesh_dir: raw mesh dir
            sdf_dir: raw sdf dir
            num_samples: num of samples used to train sdfnet
        '''
        super().__init__()

        self.rep = rep
        self.mode = mode
        if self.mode == 'train':
            split = 'train'
        elif self.mode == 'test':
            split = 'test'
        else: 
            raise ValueError('invalid mode')
        self.exp_name = exp_name
        # import ipdb; ipdb.set_trace()
        plys_w_normal = sorted(glob.glob(os.path.join(data_dir, exp_name, 'plys_global_w_normal', '*.ply')))
        self.num_data = len(plys_w_normal)
        self.surface_points = []
        self.surface_normals = []
        # import ipdb; ipdb.set_trace()
        for ply in plys_w_normal:
            cloud = o3d.io.read_point_cloud(ply)
            self.surface_points.append(np.asarray(cloud.points))
            self.surface_normals.append(np.asarray(cloud.normals))
        
        logger.info(f"dataset mode = {mode}")
        logger.info(f"dataset split = {split}")
        logger.info(f"dataset len = {self.num_data}\n")
        logger.info(f"dataset points num per frame = {self.surface_points[0].shape[0]}\n")

        self.data_dir = data_dir
        self.num_samples = num_samples
        self.sample_mode = sample_mode
        self.near_surface_num_samples = near_surface_num_samples
        self.off_num_samples = off_num_samples
        self.data_term = data_term

        if self.sample_mode:
            samples_dir = os.path.join(data_dir, exp_name, 'samples_300k')
            self.near_surface_points_sdfs = []
            for frame in range(self.num_data):
                samples = np.load(os.path.join(samples_dir, f'{frame:05d}.npy'))
                self.near_surface_points_sdfs.append(samples)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        
        data_dict = {}
        surface_points = self.surface_points[idx]
        surface_pts_sdf = np.concatenate((surface_points, np.zeros((surface_points.shape[0], 1))), axis=1)
        surface_normals = self.surface_normals[idx]
    
        select_surface_ids = np.random.choice(surface_pts_sdf.shape[0], self.num_samples)
        surface_pts_sdf = surface_pts_sdf[select_surface_ids, :]
        if self.sample_mode:
            near_surface_point_sdf = self.near_surface_points_sdfs[idx]
            assert surface_pts_sdf.shape[1] == near_surface_point_sdf.shape[1]
            select_near_surface_ids = np.random.choice(near_surface_point_sdf.shape[0], self.near_surface_num_samples)
            near_surface_point_sdf = near_surface_point_sdf[select_near_surface_ids, :]
            pts_sdf = np.concatenate((surface_pts_sdf, near_surface_point_sdf), axis=0)
        else:
            pts_sdf = surface_pts_sdf

        sdf_pts = pts_sdf[:, 0:3]
        sdf_pts_sdf = pts_sdf[:, 3:]

        off_surface_points = np.random.uniform(-1, 1, size=(self.off_num_samples, 3)).astype(np.float32)
        points_samples = np.concatenate((sdf_pts, off_surface_points), axis=0)
        off_surface_points_sdf = -1 * np.ones((self.off_num_samples, 1))
        sdf_samples = np.concatenate((sdf_pts_sdf, off_surface_points_sdf), axis=0)

        data_dict['point_samples'] = points_samples.astype(np.float32)
        data_dict['sdf_gt'] = sdf_samples.astype(np.float32)
        data_dict['surface_num'] = self.num_samples
        data_dict['normal_gt'] = surface_normals.astype(np.float32)
        data_dict['idx'] = idx
        data_dict['time'] = np.array( ( idx ) / (self.num_data - 1)).astype(np.float32) # (1, )
        
        # self.__vis__(filename, sdf_pts[:, :])
        return data_dict
    
class DeformingThings4D_some_frames(torch.utils.data.Dataset):

    def __init__(self, 
                 mode,
                 rep,
                 data_dir,
                 exp_name,
                 num_samples,
                 sample_mode,
                 data_term,
                 frames,
                 near_surface_num_samples=0,
                 off_num_samples=0,
                 **kwargs):
        '''
        Args:
            mesh_dir: raw mesh dir
            sdf_dir: raw sdf dir
            num_samples: num of samples used to train sdfnet
        '''
        super().__init__()

        self.rep = rep
        self.mode = mode
        self.frames = frames
        if self.mode == 'train':
            split = 'train'
        elif self.mode == 'test':
            split = 'test'
        else: 
            raise ValueError('invalid mode')
        self.exp_name = exp_name
        if len(self.frames) == 2:
            bgn, end = self.frames
            self.frames = list(range(bgn, end+1))
        self.num_data = len(self.frames)
        # import ipdb; ipdb.set_trace()
        plys_w_normal = sorted(glob.glob(os.path.join(data_dir, exp_name, 'plys_global_w_normal', '*.ply')))
        self.surface_points = []
        self.surface_normals = []
        # import ipdb; ipdb.set_trace()
        for ply in plys_w_normal:
            cloud = o3d.io.read_point_cloud(ply)
            self.surface_points.append(np.asarray(cloud.points))
            self.surface_normals.append(np.asarray(cloud.normals))
        
        logger.info(f"dataset mode = {mode}")
        logger.info(f"dataset split = {split}")
        logger.info(f"dataset len = {self.num_data}\n")
        logger.info(f"dataset points num per frame = {self.surface_points[0].shape[0]}\n")

        self.data_dir = data_dir
        self.num_samples = num_samples
        self.sample_mode = sample_mode
        self.near_surface_num_samples = near_surface_num_samples
        self.off_num_samples = off_num_samples
        self.data_term = data_term

        if self.sample_mode:
            samples_dir = os.path.join(data_dir, exp_name, 'samples_300k')
            self.near_surface_points_sdfs = []
            for frame in range(len(plys_w_normal)):
                samples = np.load(os.path.join(samples_dir, f'{frame:05d}.npy'))
                self.near_surface_points_sdfs.append(samples)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if self.__len__() == 1:
            time = np.array( ( 0 )).astype(np.float32)
        else:
            time = np.array( ( idx ) / (self.__len__() - 1)).astype(np.float32)
        idx = self.frames[idx]
        # import ipdb; ipdb.set_trace()
        data_dict = {}
        surface_points = self.surface_points[idx]
        surface_pts_sdf = np.concatenate((surface_points, np.zeros((surface_points.shape[0], 1))), axis=1)
        surface_normals = self.surface_normals[idx]
    
        select_surface_ids = np.random.choice(surface_pts_sdf.shape[0], self.num_samples)
        surface_pts_sdf = surface_pts_sdf[select_surface_ids, :]
        if self.sample_mode:
            near_surface_point_sdf = self.near_surface_points_sdfs[idx]
            assert surface_pts_sdf.shape[1] == near_surface_point_sdf.shape[1]
            select_near_surface_ids = np.random.choice(near_surface_point_sdf.shape[0], self.near_surface_num_samples)
            near_surface_point_sdf = near_surface_point_sdf[select_near_surface_ids, :]
            pts_sdf = np.concatenate((surface_pts_sdf, near_surface_point_sdf), axis=0)
        else:
            pts_sdf = surface_pts_sdf

        sdf_pts = pts_sdf[:, 0:3]
        sdf_pts_sdf = pts_sdf[:, 3:]

        # off_surface_points = np.random.uniform(-1, 1, size=(self.off_num_samples, 3)).astype(np.float32)
        # points_samples = np.concatenate((sdf_pts, off_surface_points), axis=0)
        # off_surface_points_sdf = -1 * np.ones((self.off_num_samples, 1))
        # sdf_samples = np.concatenate((sdf_pts_sdf, off_surface_points_sdf), axis=0)
        points_samples = sdf_pts
        sdf_samples = sdf_pts_sdf

        data_dict['point_samples'] = points_samples.astype(np.float32)
        data_dict['sdf_gt'] = sdf_samples.astype(np.float32)
        data_dict['surface_num'] = self.num_samples
        data_dict['normal_gt'] = surface_normals.astype(np.float32)
        data_dict['idx'] = idx
        data_dict['time'] = np.array( ( idx ) / (self.num_data - 1)).astype(np.float32) # (1, )
        
        # self.__vis__(idx, points_samples)
        return data_dict

    def __vis__(self, frame, surface_pts_normals):

        from plyfile import PlyData, PlyElement
        # 将数据分解为点和法线
        points = surface_pts_normals[:, :3]

        # 创建一个包含点和法线的结构化数组
        vertex = np.zeros(surface_pts_normals.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        # 将数据填充到结构化数组中
        for i in range(surface_pts_normals.shape[0]):
            vertex[i] = (points[i, 0], points[i, 1], points[i, 2])

        # 创建一个PlyElement实例
        vertex_element = PlyElement.describe(vertex, 'vertex')

        # 将PlyElement实例保存为PLY文件
        PlyData([vertex_element], text=True).write(f'data/test_vis/{frame:04d}.ply')


class DeformingThings4D_NGP(torch.utils.data.Dataset):

    def __init__(self, 
                 mode,
                 rep,
                 data_dir,
                 exp_name,
                 num_samples,
                 sample_mode,
                 clip_sdf,
                 data_term,
                 frames,
                 **kwargs):
        '''
        Args:
            mesh_dir: raw mesh dir
            sdf_dir: raw sdf dir
            num_samples: num of samples used to train sdfnet
        '''
        super().__init__()

        self.rep = rep
        self.mode = mode
        self.frames = frames
        if self.mode == 'train':
            split = 'train'
        elif self.mode == 'test':
            split = 'test'
        else: 
            raise ValueError('invalid mode')
        self.exp_name = exp_name
        # import ipdb; ipdb.set_trace()
        if len(self.frames) == 2:
            bgn, end = self.frames
            self.frames = list(range(bgn, end+1))
            self.frames_time = np.linspace(0, 1, len(self.frames))
        elif len(self.frames) == 3:
            bgn, end, step = self.frames
            self.frames = list(range(bgn, end+1, step))
            self.frames_time = np.linspace(0, 1, end+1 - bgn)[self.frames]
        self.num_data = len(self.frames)
        meshes_w_normal = sorted(glob.glob(os.path.join(data_dir, exp_name, 'global_w_normal_obj', '*.obj')))
        self.meshes = []
        self.sdf_fns = []
        # import ipdb; ipdb.set_trace()
        meshes_normalize = normalize_mashed(meshes_w_normal)
        # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
        for mesh in meshes_normalize:
            logger.info(f"[INFO] mesh vertice min: {mesh.vertices.min()}, [INFO] mesh vertice max {mesh.vertices.max()}")
            self.meshes.append(mesh)
            self.sdf_fns.append(pysdf.SDF(mesh.vertices, mesh.faces))
            logger.info(f"[INFO] mesh per frame: {mesh.vertices.shape} {mesh.faces.shape}")

        logger.info(f"dataset mode = {mode}")
        logger.info(f"dataset split = {split}")
        logger.info(f"dataset len = {self.num_data}\n")
        # import ipdb; ipdb.set_trace()
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.data_dir = data_dir
        self.num_samples = num_samples
        self.sample_mode = sample_mode
        self.data_term = data_term

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # import ipdb; ipdb.set_trace()
        
        # # print('idx before: ', idx)
        # if self.__len__() == 1:
        #     time = np.array( ( 0 )).astype(np.float32)
        # else:
        #     # time = np.array( ( idx ) / (self.__len__() - 1)).astype(np.float32)
        #     # time = idx
        time = self.frames_time[idx].astype(np.float32)
        idx = self.frames[idx]
        # import ipdb; ipdb.set_trace()
        data_dict = {}
        mesh = self.meshes[idx]
        sdf_fn = self.sdf_fns[idx]
        sdfs = np.zeros((self.num_samples, 1))
        # surface
        points_surface = mesh.sample(self.num_samples * 7 // 8)
        # perturb surface
        points_surface[self.num_samples // 2:] += 0.01 * np.random.randn(self.num_samples * 3 // 8, 3)
        # random
        points_uniform = np.random.rand(self.num_samples // 8, 3) * 2 - 1
        points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)

        sdfs[self.num_samples // 2:] = -sdf_fn(points[self.num_samples // 2:])[:,None].astype(np.float32)
 
        # clip sdf
        if self.clip_sdf != 'None':
            sdfs = sdfs.clip(-self.clip_sdf, self.clip_sdf)

        data_dict = {
            'point_samples': points,
            'sdf_gt': sdfs,
            'time': time,
            'idx': idx
        }
        # data_dict['point_samples'] = points.astype(np.float32)
        # data_dict['sdf_gt'] = sdfs.astype(np.float32)
        # data_dict['surface_num'] = self.num_samples
        # data_dict['normal_gt'] = surface_normals.astype(np.float32)
        # data_dict['idx'] = idx
        # data_dict['time'] = time
        
        # self.__vis__(idx, points_samples)
        return data_dict

    def __vis__(self, frame, surface_pts_normals):

        from plyfile import PlyData, PlyElement
        # 将数据分解为点和法线
        points = surface_pts_normals[:, :3]

        # 创建一个包含点和法线的结构化数组
        vertex = np.zeros(surface_pts_normals.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        # 将数据填充到结构化数组中
        for i in range(surface_pts_normals.shape[0]):
            vertex[i] = (points[i, 0], points[i, 1], points[i, 2])

        # 创建一个PlyElement实例
        vertex_element = PlyElement.describe(vertex, 'vertex')

        # 将PlyElement实例保存为PLY文件
        PlyData([vertex_element], text=True).write(f'data/test_vis/{frame:04d}.ply')

    
class DeformingThings4D_NGP_partial(torch.utils.data.Dataset):

    def __init__(self, 
                 mode,
                 rep,
                 data_dir,
                 exp_name,
                 num_samples,
                 sample_mode,
                 clip_sdf,
                 data_term,
                 frames,
                 **kwargs):
        '''
        Args:
            mesh_dir: raw mesh dir
            sdf_dir: raw sdf dir
            num_samples: num of samples used to train sdfnet
        '''
        super().__init__()

        self.rep = rep
        self.mode = mode
        self.frames = frames
        if self.mode == 'train':
            split = 'train'
        elif self.mode == 'test':
            split = 'test'
        else: 
            raise ValueError('invalid mode')
        self.exp_name = exp_name
        # import ipdb; ipdb.set_trace()
        if len(self.frames) == 2:
            bgn, end = self.frames
            self.frames = list(range(bgn, end))
        self.num_data = len(self.frames)
        meshes_w_normal = sorted(glob.glob(os.path.join(data_dir, exp_name, 'partial_obj', '*.obj')))
        self.meshes = []
        self.sdf_fns = []

        for mesh_path in meshes_w_normal:
            mesh = trimesh.load(mesh_path, force='mesh')
            logger.info(f"[INFO] mesh vertice min: {mesh.vertices.min()}, [INFO] mesh vertice max {mesh.vertices.max()}")
            self.meshes.append(mesh)
            self.sdf_fns.append(pysdf.SDF(mesh.vertices, mesh.faces))
            logger.info(f"[INFO] mesh per frame: {mesh.vertices.shape} {mesh.faces.shape}")

        logger.info(f"dataset mode = {mode}")
        logger.info(f"dataset split = {split}")
        logger.info(f"dataset len = {self.num_data}\n")
        # import ipdb; ipdb.set_trace()
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.data_dir = data_dir
        self.num_samples = num_samples
        self.sample_mode = sample_mode
        self.data_term = data_term

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # print('idx before: ', idx)
        if self.__len__() == 1:
            time = np.array( ( 0 )).astype(np.float32)
        else:
            time = np.array( ( idx ) / (self.__len__() - 1)).astype(np.float32)
            # time = idx
        idx = self.frames[idx]
        # print('idx after: ', idx, ' time: ', time)
        # import ipdb; ipdb.set_trace()
        data_dict = {}
        mesh = self.meshes[idx]
        sdf_fn = self.sdf_fns[idx]
        sdfs = np.zeros((self.num_samples, 1))
        # surface
        points_surface = mesh.sample(self.num_samples * 7 // 8)
        # perturb surface
        points_surface[self.num_samples // 2:] += 0.01 * np.random.randn(self.num_samples * 3 // 8, 3)
        # random
        points_uniform = np.random.rand(self.num_samples // 8, 3) * 2 - 1
        points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)

        sdfs[self.num_samples // 2:] = -sdf_fn(points[self.num_samples // 2:])[:,None].astype(np.float32)

        # clip sdf
        if self.clip_sdf != 'None':
            sdfs = sdfs.clip(-self.clip_sdf, self.clip_sdf)

        vertex = np.array(mesh.vertices)
        select_ids = np.random.choice(vertex.shape[0], 8192)
        vertex = vertex[select_ids, :]
        vertex_sdf = np.zeros((vertex.shape[0], 1))

        points = np.concatenate([vertex, points], axis=0).astype(np.float32)
        sdfs = np.concatenate([vertex_sdf, sdfs], axis=0).astype(np.float32)

        data_dict = {
            'point_samples': points,
            'sdf_gt': sdfs,
            'time': time,
            'idx': idx
        }
        # data_dict['point_samples'] = points.astype(np.float32)
        # data_dict['sdf_gt'] = sdfs.astype(np.float32)
        # data_dict['surface_num'] = self.num_samples
        # data_dict['normal_gt'] = surface_normals.astype(np.float32)
        # data_dict['idx'] = idx
        # data_dict['time'] = time
        
        # self.__vis__(idx, points_samples)
        return data_dict
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=str, help='sdf or mesh')
    parser.add_argument("--config", type=str, required=True, help='config yaml file path, e.g. ../config/dfaust.yaml')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)
    update_config_from_args(config, args)

    train_dataset = DeformingThings4D(mode='train', rep=config.rep, **config.dataset)
    test_dataset  = DeformingThings4D(mode='test',  rep=config.rep, **config.dataset)

    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    for batch_idx, batch_data in enumerate(train_loader):
    # for batch_idx, batch_data in enumerate(test_loader):
        import ipdb; ipdb.set_trace()
        for i in range(batch_size):
            if args.rep == 'sdf':
                print(i, batch_data['point_samples'].shape)
                print(i, batch_data['sdf_gt'].shape)
            if args.rep == 'mesh':
                raise NotImplementedError
            continue

            import open3d as o3d
            from vis_utils import create_trianlge_mesh, create_pointcloud_from_points

            sdf_mask = (batch_data['sdf_samples'][i, :, 3:] > 0).numpy().astype(np.float32)
            pcd = create_pointcloud_from_points(batch_data['sdf_samples'][i, :, :3], sdf_mask * np.array([1, 0, 0]) + (1 - sdf_mask) * np.array([0, 0, 1]))
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
            mesh = create_trianlge_mesh(batch_data['verts_init'][i], batch_data['faces_init'][i].T)
            o3d.visualization.draw_geometries([mesh, pcd, coord])

        # break
    import ipdb; ipdb.set_trace()





