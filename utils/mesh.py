#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import time
import torch
import trimesh
import plyfile
import numpy as np
import skimage.measure
from skimage import measure

from loguru import logger

def get_sdf_2d_mesh(config, model, latent_vec, N=256, max_batch=int(2 ** 18), time=None):
    """
    Args:
        latent_vec: (d,)
    """
    assert(len(latent_vec.shape) == 1)
    image_range = np.array([-1, 1])
    pixel_size = (image_range[1] - image_range[0]) / N
    hd = pixel_size / 2

    # use pixel middle center instead of corner
    x = torch.linspace(-(1-hd), (1-hd), N)
    y = torch.linspace(-(1-hd), (1-hd), N)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    xy = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2) # (N*N, 2)
    sdf_pred_np = np.zeros(N*N)

    samples = torch.from_numpy(xy).to(latent_vec.device)
    samples.requires_grad = False
    num_samples = samples.shape[0]

    head = 0
    batch_dict = {}
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples)]
        batch_dict['point_samples'] = sample_subset[None, :, :]
        batch_dict['time'] = torch.tensor(time).reshape(-1, 1).float().cuda()
        batch_dict = model(latent_vec[None, :], batch_dict, config) 
        sdf_pred_subset = batch_dict["sdf_pred"][0] # batch_size == 1
        sdf_pred_np[head : min(head + max_batch, num_samples)] = sdf_pred_subset.detach().cpu().numpy()
        head += max_batch

    sdf_grid_pred = sdf_pred_np.reshape(N, N)
    # Find contours at a constant value of thres
    thresh = 0.0
    contours_pred = measure.find_contours(sdf_grid_pred, 0.0)
    # import ipdb; ipdb.set_trace()
    while contours_pred == []:
        thresh += 0.01
        contours_pred = measure.find_contours(sdf_grid_pred, thresh)
    logger.info(f" interpolate sdf surface value = {thresh} ")
    
    return sdf_grid_pred, contours_pred


def get_sdf_3d_mesh(config, model, N=256, max_batch=int(2 ** 18), time=None):
    """
    Args:
        latent_vec: (d,) 删去latent_vec, 暂时用不到
    """
    image_range = np.array([-1, 1])
    pixel_size = (image_range[1] - image_range[0]) / N
    hd = pixel_size / 2

    # use pixel middle center instead of corner
    x = torch.linspace(-(1-hd), (1-hd), N)
    y = torch.linspace(-(1-hd), (1-hd), N)
    z = torch.linspace(-(1-hd), (1-hd), N)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='xy')
    xyz = torch.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)  # (N*N*N, 3)
    sdf_pred_np = np.zeros(N*N*N)

    samples = torch.from_numpy(xyz).to(time.device)
    samples.requires_grad = False
    num_samples = samples.shape[0]

    head = 0
    batch_dict = {}
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples)]
        batch_dict['point_samples'] = sample_subset[None, :, :]
        batch_dict['time'] = torch.tensor(time).reshape(-1, 1).float().cuda()
        batch_dict = model(batch_dict, config) 
        sdf_pred_subset = batch_dict["sdf_pred"][0] # batch_size == 1
        sdf_pred_np[head : min(head + max_batch, num_samples)] = sdf_pred_subset.detach().cpu().numpy()
        head += max_batch

    sdf_grid_pred = sdf_pred_np.reshape(N, N, N)

    # define your level value, the value at which you want to extract the surface
    level = 0.0

    # Apply marching cubes (you may want to choose an appropriate level value)
    verts, faces, normals, values = measure.marching_cubes(sdf_grid_pred, level)

    # verts, faces are numpy arrays that represent the surface as a set of vertices and faces
    
    return sdf_grid_pred, verts, faces, normals, values

 

def mesh_create_mesh_from_single_lat(model, latent_vec, ply_fname, template_faces, data_mean_gpu, data_std_gpu):
    '''
    Args:
        latent_vec: (latent_dim,)
    '''
    model.eval()
    end_points = model(lat_vecs=latent_vec[None, :]) 
    mesh_verts_pred = end_points['mesh_out_pred'] * data_std_gpu + data_mean_gpu
    mesh_verts_pred = mesh_verts_pred.detach().cpu().numpy()[0] # batch_size == 1
    mesh = trimesh.Trimesh(vertices=mesh_verts_pred, faces=template_faces, process=False)
    mesh.export(ply_fname)


def sdf_create_mesh_from_single_lat(
    model, latent_vec, ply_fname, N=256, max_batch=int(2 ** 18), offset=None, scale=None
):
    '''
    Args:
        latent_vec: (latent_dim,). Note: lat_vecs: (B, latent_dim)
    '''
    start = time.time()

    model.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
        data = {}
        data['sdf_samples'] = sample_subset[None, :, :]
        end_points = model(latent_vec[None, :], data) 
        sdf_pred = end_points["sdf_pred"][0] # batch_size == 1
        samples[head : min(head + max_batch, num_samples), 3] = sdf_pred.detach().cpu()

        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    if ply_fname is not None:
        convert_sdf_samples_to_ply(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            ply_fname,
            offset,
            scale,
        )
    else:
        iso_vertices, iso_faces = convert_sdf_samples_to_verts_faces(
                                      sdf_values.data.cpu(),
                                      voxel_origin,
                                      voxel_size,
                                      offset,
                                      scale,
                                  )
        return iso_vertices, iso_faces
                    

def sdf_collect_results_from_single_lat(
    config, model, latent_vec, ply_fname, N=256, max_batch=int(2 ** 18), verts_temp=None, offset=None, scale=None
):
    '''
    Args:
        latent_vec: (latent_dim,). Note: lat_vecs: (B, latent_dim)
    '''
    start = time.time()

    model.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)
    samples_temp = torch.zeros(N ** 3)
    xyz_temp_def = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    data = {'verts_temp': verts_temp[None, :, :].cuda()}
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
        data['sdf_samples'] = sample_subset[None, :, :]
        data['sdf_samples_temp'] = sample_subset[None, :, :]
        end_points = model(latent_vec[None, :], data, config) 
        sdf_pred = end_points["sdf_pred"][0] # batch_size == 1
        #@ sdf_temp_pred = end_points["sdf_temp_pred"][0] # batch_size == 1
        samples[head : min(head + max_batch, num_samples), 3] = sdf_pred.detach().cpu()
        #@ samples_temp[head : min(head + max_batch, num_samples)] = sdf_temp_pred.detach().cpu()
        xyz_temp_def[head : min(head + max_batch, num_samples)] = end_points["xyz_temp_def"][0].detach().cpu()

        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)
    #@ sdf_temp_values = samples_temp.reshape(N, N, N)

    # debug start
    # data['sdf_samples_temp'] = torch.from_numpy(trimesh.load(f"/scratch/cluster/yanght/Projects/ShapeCorres/shapecorres/gencorres/work_dir/Human_hybrid/multistage_2stage_akap/results/train/sdf/100003_sdf_temp.ply", process=False).vertices)[None, :, :].float().cuda()
    # end_points = model(latent_vec[None, :], data, config) 
    # debug end

    end = time.time()
    print("sampling takes: %f" % (end - start))

    if ply_fname is not None:
        assert("currently not used")
    else:
        iso_vertices, iso_faces = convert_sdf_samples_to_verts_faces(
                                      sdf_values.data.cpu(),
                                      voxel_origin,
                                      voxel_size,
                                      offset,
                                      scale,
                                  )
        #@ iso_temp_vertices, iso_temp_faces = convert_sdf_samples_to_verts_faces(
        #@                               sdf_temp_values.data.cpu(),
        #@                               voxel_origin,
        #@                               voxel_size,
        #@                               offset,
        #@                               scale,
        #@                           )

        xyz = samples[:, 0:3].reshape(N, N, N, 3)
        xyz_temp = xyz
        # debug start
        # xyz_temp = data['sdf_samples_temp'].detach().cpu().numpy()
        # xyz_temp_def = end_points['xyz_temp_def'][0].detach().cpu().numpy()
        # sdf_temp_values = end_points['sdf_temp_pred'][0].detach().cpu().numpy()
        # return iso_vertices, iso_faces, iso_temp_vertices, iso_temp_faces, end_points["verts_temp_def"][0].detach().cpu().numpy(), xyz_temp, xyz_temp_def, sdf_temp_values
        # debug end
        #@ return iso_vertices, iso_faces, iso_temp_vertices, iso_temp_faces, end_points["verts_temp_def"][0].detach().cpu().numpy(), xyz_temp, xyz_temp_def.reshape(N, N, N, 3), sdf_temp_values
        return iso_vertices, iso_faces, end_points["verts_temp_def"][0].detach().cpu().numpy(), xyz_temp, xyz_temp_def.reshape(N, N, N, 3)
 

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logger.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logger.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )


def convert_sdf_samples_to_verts_faces(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    return mesh_points, faces
