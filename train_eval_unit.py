import os, sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import trimesh
from loguru import logger
import trimesh

from models.arap import ARAP
from pyutils import get_directory, to_device
from utils.mesh import sdf_create_mesh_from_single_lat, mesh_create_mesh_from_single_lat, sdf_collect_results_from_single_lat
from utils import implicit_utils, mesh

from utils.time_utils import *
from utils.arap_potential import arap_energy_exact, compute_neigh
from utils.diff_operators import jacobian
from utils.scheduler_utils import adjust_learning_rate


# Only set requires_grad is NOT enough to freeze params when using shared optimizer
# Ref: https://discuss.pytorch.org/t/parameters-with-requires-grad-false-are-updated-during-training/90096/9
def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad
        if requires_grad:
            param.grad = torch.zeros_like(param)
        else:
            param.grad = None


def set_params_grad(model, lat_vecs, config):
    model_module = model.module if config.distributed else model 
    set_requires_grad(model_module.decoder, True)
    set_requires_grad(lat_vecs, True)

def train_one_epoch(state_info, config, fabric, train_loader, model, optimizer_train, scheduler_train, scaler, writer):
    model.train()
    device = state_info['device']
    epoch = state_info['epoch']
    schedular_type = config.optimization[config.rep].schedular
    # ASAP
    if config.use_sdf_asap:
        logger.warning("use sdf ASAP loss")

    for b, batch_dict in enumerate(train_loader):
        state_info['b'] = b
        optimizer_train.zero_grad()
        batch_dict['epoch'] = torch.ones(batch_dict['idx'].shape[0], dtype=torch.long) * epoch
        # import ipdb; ipdb.set_trace()
        if fabric is not None:
            (time, batch_dict) = model(batch_dict) # (B, N, 3)
            data_arap_loss = model.get_loss(time, batch_dict, state_info)
            if config.loss.topology_PD_loss and config.use_topo_loss:
                topo_pd_loss = model.get_PD_loss(time, state_info)
                loss = data_arap_loss + topo_pd_loss
            else:
                loss = data_arap_loss

            # 使用 GradScaler 缩放损失
            fabric.backward(scaler.scale(loss))

        # ----------------------------- old ----------------------- #
        else:
            batch_dict = to_device(batch_dict, device)

            with autocast():
                (time, batch_dict) = model(batch_dict) # (B, N, 3)
                data_arap_loss = model.get_loss(time, batch_dict, state_info)
                if config.loss.topology_PD_loss and config.use_topo_loss:
                    topo_pd_loss = model.get_PD_loss(time, state_info)
                    loss = data_arap_loss + topo_pd_loss
                else:
                    loss = data_arap_loss
                # import ipdb; ipdb.set_trace()
                if config.loss.EDR:
                    rand_coords = torch.rand_like(batch_dict['point_samples']) * 2 - 1
                    rand_coords_offset = rand_coords + torch.randn_like(rand_coords) * 1e-2
                    batch_dict['point_samples'] = rand_coords
                    (_, d_rand_coords) = model(batch_dict)
                    d_rand_coords = d_rand_coords["sdf_pred"].reshape(-1, 1)
                    batch_dict['point_samples'] = rand_coords_offset
                    _, d_rand_coords_offset = model(batch_dict)
                    d_rand_coords_offset = d_rand_coords_offset["sdf_pred"].reshape(-1, 1)
                    EDR_loss = nn.functional.mse_loss(d_rand_coords, d_rand_coords_offset)
                    state_info['EDR_loss'] = EDR_loss.item()
                    state_info['EDR_weight'] = 1e3
                    EDR_loss = EDR_loss * 1e3
                    loss += EDR_loss
                if config.loss.Time_EDR:
                    rand_coords = torch.rand_like(batch_dict['point_samples']) * 2 - 1
                    batch_dict['time'] = torch.rand_like(batch_dict['time'])
                    batch_dict['point_samples'] = rand_coords
                    (_, d_rand_coords) = model(batch_dict)
                    d_rand_coords = d_rand_coords["sdf_pred"].reshape(-1, 1)
                    # 生成随机噪声
                    noise = torch.rand_like(batch_dict['time']) * 0.01 - 0.005  # 这会生成一个范围在 [-0.005, 0.005] 的随机噪声
                    batch_dict['time'] += noise  # 将噪声添加到 x
                    _, d_rand_coords_offset = model(batch_dict)
                    d_rand_coords_offset = d_rand_coords_offset["sdf_pred"].reshape(-1, 1)
                    Time_EDR_loss = nn.functional.mse_loss(d_rand_coords, d_rand_coords_offset)
                    state_info['Time_EDR_loss'] = Time_EDR_loss.item()
                    state_info['Time_EDR_weight'] = 1e3
                    Time_EDR_loss = Time_EDR_loss * 1e3
                    loss += Time_EDR_loss
            # 使用 GradScaler 缩放损失
            scaler.scale(loss).backward()

        # 使用 GradScaler 调用 step
        scaler.step(optimizer_train)
        scaler.update()
        lr_group = adjust_learning_rate(schedular_type, scheduler_train, optimizer_train, epoch)
        state_info.update({'lr': lr_group})

        # loss.backward()
        # optimizer_train.step()

        if b % config.log.print_batch_interval == 0:
            global_step = (state_info['epoch'] * state_info['len_train_loader'] + b ) * config.optimization[config.rep].batch_size
            writer.log_state_info(state_info)
            writer.log_summary(state_info, global_step, mode='train')

    # writer.log_state_info(state_info)
    # writer.log_summary_epoch(state_info, mode='train')

    return state_info

def train_debug_DeepSDF(state_info, config, lat_vec, train_loader, model, optimizer_train, scheduler_train, scaler, writer):
    model.train()
    device = state_info['device']
    epoch = state_info['epoch']
    schedular_type = config.optimization[config.rep].schedular
    # ASAP
    if config.use_sdf_asap:
        logger.warning("use sdf ASAP loss")

    for b, batch_dict in enumerate(train_loader):
        state_info['b'] = b
        optimizer_train.zero_grad()
        # import ipdb; ipdb.set_trace()
        batch_dict['epoch'] = torch.ones(batch_dict['idx'].shape[0], dtype=torch.long) * epoch
        batch_dict['lat_vec'] = lat_vec(batch_dict['idx'].to(device))[:,:] # (B, latent_dim)
        batch_dict = to_device(batch_dict, device)

        with autocast():
            (time, batch_dict) = model(batch_dict) # (B, N, 3)
            loss = model.get_loss(time, batch_dict, state_info)
            l2_size_loss = torch.sum(torch.norm(batch_dict['lat_vec'], dim=1))
            reg_loss = (
                (1e-4) * min(1, epoch / 100) * l2_size_loss
            ) / batch_dict['lat_vec'].shape[0]
            state_info['l2_reg_loss'] = reg_loss.item()
            loss += reg_loss
            # if config.loss.topology_PD_loss and config.use_topo_loss:
            #     topo_pd_loss = model.get_PD_loss(time, state_info)
            #     loss += topo_pd_loss

        # 使用 GradScaler 缩放损失
        scaler.scale(loss).backward()

        # 使用 GradScaler 调用 step
        scaler.step(optimizer_train)
        scaler.update()

        # loss.backward()
        # optimizer_train.step()

        if b % config.log.print_batch_interval == 0:
            global_step = (state_info['epoch'] * state_info['len_train_loader'] + b ) * config.optimization[config.rep].batch_size
            writer.log_state_info(state_info)
            writer.log_summary(state_info, global_step, mode='train')

    return state_info


def get_jacobian_rand(cur_shape, z, data_mean_gpu, data_std_gpu, model, device, epsilon=[1e-3],nz_max=10):
    nb, nz = z.size()
    _, n_vert, nc = cur_shape.size()
    if nz >= nz_max:
      rand_idx = np.random.permutation(nz)[:nz_max]
      nz = nz_max
    else:
      rand_idx = np.arange(nz)
    
    jacobian = torch.zeros((nb, n_vert*nc, nz)).to(device)
    for i, idx in enumerate(rand_idx):
        dz = torch.zeros(z.size()).to(device)
        dz[:, idx] = epsilon
        z_new = z + dz
        out_new = model(z_new)
        out_new = out_new['mesh_out_pred']
        shape_new = out_new * data_std_gpu + data_mean_gpu
        dout = (shape_new - cur_shape).view(nb, -1)
        jacobian[:, :, i] = dout/epsilon
    return jacobian



def test_opt_one_epoch(state_info, config, test_loader, model, test_lat_vecs, optimizer_test, writer):
    model.eval()

    epoch = state_info['epoch']
    device = state_info['device']

    if config.rep == 'mesh':
        data_mean_gpu = test_loader.dataset.mean.to(device) # same as train_loader
        data_std_gpu = test_loader.dataset.std.to(device)

    # ASAP
    if config.use_sdf_asap:
        logger.warning("use sdf ASAP loss")

    for b, batch_dict in enumerate(test_loader):
        state_info['b'] = b
        optimizer_test.zero_grad()
        batch_dict = to_device(batch_dict, device)

        batch_vecs = test_lat_vecs(batch_dict['idx']) # (B, latent_dim)

        batch_dict = model(batch_vecs, batch_dict, config, state_info) # (B, N, 3)
        loss = batch_dict["loss"]

        loss.backward()
        optimizer_test.step()

        if b % config.log.print_batch_interval == 0:
            global_step = (state_info['epoch'] * state_info['len_test_loader'] + b ) * config.optimization[config.rep].batch_size
            writer.log_state_info(state_info)
            writer.log_summary(state_info, global_step, mode='test')

    return state_info


def recon_from_lat_vecs(config, results_dir, model, recon_lat_vecs, mesh_sdf_dataset):
    '''
    Args:
        recon_lat_vecs: Embedding(D, latent_dim)
    '''
    model.eval()
    num_recon = recon_lat_vecs.weight.shape[0]
    template_faces = mesh_sdf_dataset.template_faces
    assert(num_recon == len(mesh_sdf_dataset.fid_list))

    def save_obj(fname, vertices, faces):
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.export(fname)

    if config.rep == 'sdf':
        logger.info(" reconstruct mesh from sdf predicted by sdfnet ")
        recon_idx_list = range(num_recon)
        for i, data_dict in enumerate(mesh_sdf_dataset):
            if i not in recon_idx_list:
                continue
            fid = mesh_sdf_dataset.fid_list[i]
            print(f"i={i}, fid={fid}")
            sdf_grid_pred, contours_pred = mesh.get_sdf_2d_mesh(config, model, recon_lat_vecs.weight[i], N=256, max_batch=int(2**18)) 

            with open(f"{results_dir}/{fid}_sdf.pkl", "wb") as f:
                pickle.dump({'sdf_pred': sdf_grid_pred, 'contours_pred': contours_pred,
                             'point_samples': data_dict['point_samples'], 'sdf_samples': data_dict['sdf_samples'],
                             'vertices': data_dict['vertices'], 'faces': template_faces, }, f )

    else:
        assert(config.rep in ['mesh', 'all'])
        raise NotImplementedError


def bak_interp_from_lat_vecs(config, results_dir, model, recon_lat_vecs, mesh_sdf_dataset):
    '''
    Args:
        recon_lat_vecs: Embedding(D, latent_dim)
    '''
    model.eval()
    num_recon = recon_lat_vecs.weight.shape[0]
    template_faces = mesh_sdf_dataset.template_faces
    num_interp = 10
    if config.split == 'train':
        src_fid = '0_0_0_0'
        tgt_fid = '8_8_0_0'
    elif config.split == 'test':
        src_fid = '1_1_0_0'
        tgt_fid = '9_9_0_0'
    else:
        raise NotImplementedError

    logger.info(" interpolate sdf predicted by sdfnet ")
    for i, fid in enumerate(mesh_sdf_dataset.fid_list):
        if fid == src_fid:
            src_idx = i
        if fid == tgt_fid:
            tgt_idx = i

    logger.info(f"interpolate {src_fid} ({src_idx}-th) and {tgt_fid} ({tgt_idx}-th)")
    for i_interp in range(0, num_interp + 1): 
        ri = i_interp / num_interp

        interp_lat_vecs = recon_lat_vecs.weight[src_idx] * (1 - ri) + recon_lat_vecs.weight[tgt_idx] * ri

        dump_dir = get_directory( f"{results_dir}/{src_idx}_{tgt_idx}" )
        sdf_grid_pred, contours_pred = mesh.get_sdf_2d_mesh(config, model, interp_lat_vecs, N=256, max_batch=int(2**18)) 

        with open(f"{dump_dir}/{src_idx}_{tgt_idx}_{i_interp:02d}.pkl", "wb") as f:
            pickle.dump({'sdf_pred': sdf_grid_pred, 'contours_pred': contours_pred,
                         'src_point_samples': mesh_sdf_dataset[src_idx]['point_samples'], 'src_sdf_samples': mesh_sdf_dataset[src_idx]['sdf_samples'],
                         'src_vertices': mesh_sdf_dataset[src_idx]['vertices'], 'tgt_point_samples': mesh_sdf_dataset[tgt_idx]['point_samples'], 
                         'tgt_sdf_samples': mesh_sdf_dataset[tgt_idx]['sdf_samples'], 'tgt_vertices': mesh_sdf_dataset[tgt_idx]['vertices'], 
                         'faces': template_faces, }, f )

def interp_from_lat_vecs(config, results_dir, model, lat_vec, mesh_sdf_dataset):
    '''
    Args:
        recon_lat_vecs: Embedding(D, latent_dim)
    '''
    # import ipdb; ipdb.set_trace()
    model.eval()
    num_interp = config.dataset.frames[1] - config.dataset.frames[0]
    num_interp = 10*num_interp
    logger.info(f" interpolate sdf predicted by sdfnet, len = {num_interp} ")
    resolution=128
    with autocast():
        if num_interp == 0:
            ri = 0.0

            dump_dir = get_directory( f"{results_dir}/" )
            # sdf_grid_pred, contours_pred = mesh.get_sdf_2d_mesh(config, model, interp_lat_vecs, N=256, max_batch=int(2**18), time=ri) 
            x_range, y_range, z_range = config.loss.get('x_range', [-1, 1]), config.loss.get('y_range', [-1, 1]), config.loss.get('z_range', [-1, 1])
            verts, faces = implicit_utils.sdf_decode_mesh_from_single_lat(config, model, ri, resolution=resolution, max_batch=int(2 ** 18), offset=None, scale=None, x_range=x_range, y_range=y_range, z_range=z_range)
            # mesh_sim = trimesh.Trimesh(vertices=verts, faces=faces).simplify_quadratic_decimation(2000)
            mesh_sim = trimesh.Trimesh(vertices=verts, faces=faces)
            verts = mesh_sim.vertices
            faces = mesh_sim.faces

            save_obj(f"{dump_dir}/test.obj", verts, faces)
        else:
            for i_interp in range(0, num_interp + 1): 
                ri = i_interp / num_interp
                # ri = i_interp
                # ri = lat_vec[ri]
            # for i_interp in [8,9,10,11,12,13,14,15,16]:
            #     ri = i_interp / 8

                dump_dir = get_directory( f"{results_dir}/" )
                # sdf_grid_pred, contours_pred = mesh.get_sdf_2d_mesh(config, model, interp_lat_vecs, N=256, max_batch=int(2**18), time=ri) 
                x_range, y_range, z_range = config.loss.get('x_range', [-1, 1]), config.loss.get('y_range', [-1, 1]), config.loss.get('z_range', [-1, 1])
                verts, faces = implicit_utils.sdf_decode_mesh_from_single_lat(config, model, ri, resolution=resolution, max_batch=int(2 ** 16), offset=None, scale=None, x_range=x_range, y_range=y_range, z_range=z_range)
                # mesh_sim = trimesh.Trimesh(vertices=verts, faces=faces).simplify_quadratic_decimation(2000)
                mesh_sim = trimesh.Trimesh(vertices=verts, faces=faces)
                verts = mesh_sim.vertices
                faces = mesh_sim.faces

                save_obj(f"{dump_dir}/{i_interp:02d}.obj", verts, faces)


def save_obj(fname, vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(fname)

def interp_from_lat_vecs_gt(config, results_dir, model, recon_lat_vecs, mesh_sdf_dataset):
    '''
    Args:
        recon_lat_vecs: Embedding(D, latent_dim)
    '''
    model.eval()
    num_recon = recon_lat_vecs.weight.shape[0]
    template_faces = mesh_sdf_dataset.template_faces
    num_interp = 10
    # dense dataset
    if config.split == 'train':
        fid_list = [f'{i}_{i}_0_0' for i in [0, 2, 4, 6, 8, 10, 12, 14, 16]]
        fid_list += [f'0_0_{i}_{i}' for i in [0, 2, 4, 6, 8]]
    elif config.split == 'test':
        fid_list = [f'{i+1}_{i+1}_0_0' for i in [0, 2, 4, 6, 8, 10, 12, 14, 16]]
        fid_list += [f'0_0_{i+1}_{i+1}' for i in [0, 2, 4, 6, 8]]
    else:
        raise NotImplementedError

    logger.info(" interpolate sdf predicted by sdfnet ")
    idx_list = [0 for _ in fid_list]
    for i, query_fid in enumerate(mesh_sdf_dataset.fid_list):
        for j, fid in enumerate(fid_list):
            if query_fid == fid:
                idx_list[j] = i

    for idx in idx_list:

        interp_lat_vecs = recon_lat_vecs.weight[idx]

        dump_dir = get_directory( f"{results_dir}/gt" )
        sdf_grid_pred, contours_pred = mesh.get_sdf_2d_mesh(config, model, interp_lat_vecs, N=256, max_batch=int(2**18)) 

        with open(f"{dump_dir}/{idx}.pkl", "wb") as f:
            pickle.dump({'sdf_pred': sdf_grid_pred, 'contours_pred': contours_pred, 'faces': template_faces, }, f )


def analysis_one_epoch(state_info, config, results_dir, model, lat_vecs, mesh_sdf_loader):
    '''
    Args:
        lat_vecs: Embedding(D, latent_dim)
    '''
    model.eval()

    epoch = state_info['epoch']
    device = state_info['device']

    num_recon = lat_vecs.weight.shape[0]
    assert(num_recon == len(mesh_sdf_loader.dataset.fid_list))

    if config.rep == 'sdf':
        traces_list = []
        for b, batch_dict in enumerate(mesh_sdf_loader):
            print(b)
            batch_dict = to_device(batch_dict, device)

            batch_vecs = lat_vecs(batch_dict['idx']) # (B, latent_dim)

            batch_dict = model(batch_vecs, batch_dict, config) # (B, N, 3)
            traces = batch_dict['traces']

            traces_list.append(traces.detach().cpu().numpy())

        from IPython import embed; embed()

    else:
        assert(config.rep in ['mesh', 'all'])
        raise NotImplementedError


