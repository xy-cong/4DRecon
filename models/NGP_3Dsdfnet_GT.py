import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


import numpy as np
import triangle as tr
from skimage import measure
import trimesh

from models.sdfnet_base import SdfDecoder, SdfDecoder_Encoder, SdfDecoder_Modify, SdfDecoder_freenerf, SdfDecoder_Weightnorm
from models.MultiMLP_base import SdfDecoder_MultiMLP
from models.NGP_warpnet import SDF_Warping
from models.NGP_sdfnet_base import SDFNetwork
from utils.diff_operators import gradient
from models.asap import compute_asap3d_sparse
from models.implicit_reg import implicit_reg
from utils import implicit_utils

from einops import rearrange, repeat
import torch_sparse as ts

# import topologylayer.nn
from topologylayer.nn import BatchedLevelSetLayer3D
from pytorch3d.loss import chamfer_distance


# from models.udf2mesh import find_contours
from models.marching_square import marching_square, contours_process

class DiffSdf2Verts(torch.autograd.Function):
    """ differentiable sdf to verts
    """
    @staticmethod
    def forward(ctx, sdf, x, fx):
        """
        sdf: (n, 1)
        x:   (n, query_dim)
        fx:  (n, query_dim), gradient, d(sdf)/d(x)
        """
        assert(sdf.shape[0] == x.shape[0] == fx.shape[0])
        ctx.save_for_backward(fx)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        fx, = ctx.saved_tensors
        grad_sdf = - torch.sum(grad_output * fx, dim=-1, keepdims=True)

        return grad_sdf, None, None

diff_sdf2verts = DiffSdf2Verts.apply


class SdfNet(nn.Module):
    def __init__(self,
                 config
                ):
        super().__init__()
        self.config = config
        self.freenerf = False
        self.MultiMLP = False

        if config.model.SDF_type == 'NGP_sdf':
            self.decoder = SDFNetwork(**config.model.NGP_sdf)
        elif config.model.SDF_type == 'DeepSDF':
            self.decoder = SdfDecoder_Encoder(**config.model.sdf)
        elif config.model.SDF_type == 'ModifySDF':
            self.decoder = SdfDecoder_Modify(**config.model.Modify_sdf)
        elif config.model.SDF_type == 'WeightSDF':
            self.decoder = SdfDecoder_Weightnorm(**config.model.Modify_sdf)
        elif config.model.SDF_type == 'WarpSDF':
            self.decoder = SDF_Warping(config.model)
        elif config.model.SDF_type == 'freenerf':
            self.decoder = SdfDecoder_freenerf(**config.model.freenerf)
            self.freenerf = True
        elif config.model.SDF_type == 'MultiMLP':
            self.decoder = SdfDecoder_MultiMLP(**config.model.MultiMLP)
            self.MultiMLP = True
        else:
            raise NotImplementedError
        if config.loss.topology_PD_loss:
            self.meshgrid_size = config.dataset.meshgrid_size
            self.topology_PD_loss = BatchedLevelSetLayer3D(size=(self.meshgrid_size,self.meshgrid_size,self.meshgrid_size), sublevel=False)
            GT_pd_dir = os.path.join(config.data_dir, config.exp_name, 'GT_pd')
            self.GT_pd_h0 = torch.from_numpy(np.load(os.path.join(GT_pd_dir, 'GT_pd_h0.npy')))
            self.GT_pd_h1 = torch.from_numpy(np.load(os.path.join(GT_pd_dir, 'GT_pd_h0.npy')))
            self.GT_pd_h2 = torch.from_numpy(np.load(os.path.join(GT_pd_dir, 'GT_pd_h0.npy')))
    
    def forward(self, batch_dict):
        '''
        Args:
            batch_vecs: (B, latent_dim)
            xyz: (B, num_query, query_dim={2, 3})
        '''
        # import ipdb; ipdb.set_trace()
        time = batch_dict['time'].reshape(-1, 1) # (B, 1)

        query = batch_dict['point_samples'].clone().detach().requires_grad_(True) # (B, S, query_dim)

        B, S = query.shape[0], query.shape[1]

        if self.freenerf:
            sdf_pred = self.decoder(repeat(time, 'B d -> (B S) d', S=S),
                                    rearrange(query, 'B S d -> (B S) d'),
                                    batch_dict['epoch'])
        elif self.MultiMLP:
            sdf_pred = self.decoder(time, query)
        else:
            # if batch_dict['time'] == 0:
            #     is_canonical = True
            # else:
            #     is_canonical = False
            sdf_pred = self.decoder(repeat(time, 'B d -> (B S) d', S=S),
                                    rearrange(query, 'B S d -> (B S) d'),
                                    )
        sdf_pred = sdf_pred.reshape(B, S)

        batch_dict['query'] = query
        batch_dict["sdf_pred"] = sdf_pred # (B, S)

        if self.config.dataset.data_term:
            gradient = torch.autograd.grad(sdf_pred, query, torch.ones_like(sdf_pred), create_graph=True)[0]
            batch_dict['gradient'] = gradient

        return (time, batch_dict)

    def get_sdf_from_samples(self, batch_dict):
        '''
        Args:
            batch_vecs: (B, latent_dim)
            xyz: (B, num_query, query_dim={2, 3})
            没有任何多余的计算, 只是给定samples过MLP得到sdf
        '''
        # import ipdb; ipdb.set_trace()
        time = batch_dict['time'].reshape(-1, 1) # (B, 1)
        # import ipdb; ipdb.set_trace()
        batch_vecs = time
        query = batch_dict['point_samples'].clone().detach().requires_grad_(True) # (B, S, query_dim)

        B, S = query.shape[0], query.shape[1]
        if self.freenerf:
            sdf_pred = self.decoder(repeat(time, 'B d -> (B S) d', S=S),
                                    rearrange(query, 'B S d -> (B S) d'),
                                    torch.tensor(100000).cuda())
        else:
            # if batch_dict['time'] == 0:
            #     is_canonical = True
            # else:
            #     is_canonical = False
            sdf_pred = self.decoder(repeat(batch_vecs, 'B d -> (B S) d', S=S),
                                    rearrange(query, 'B S d -> (B S) d'),
                                    )
        sdf_pred = sdf_pred.reshape(B, S)

        batch_dict['query'] = query
        batch_dict["sdf_pred"] = sdf_pred # (B, S)

        return batch_dict

    def get_ndpd(self, sdfs: torch.Tensor, min_persistence=0.0):
        # import ipdb; ipdb.set_trace()
        # sdf.shape: (B, meshgrid_size, meshgrid_size, meshgrid_size)
        dgms, _ = self.topology_PD_loss(sdfs) 
        res = []
        for dgm in dgms:
            inds = torch.all(torch.all(torch.isfinite(dgm), dim=2), dim=0)
            fh = dgm[:, inds, ...]
            diag = fh[:, :, 0] - fh[:, :, 1]
            res.append(fh[:, torch.any(diag > min_persistence, axis=0), ...])
        return res
    
    def get_loss(self, time, batch_dict, state_info):
        # import ipdb; ipdb.set_trace()
        B = time.shape[0]
        device = time.device
        loss = torch.zeros(1).to(device) 
        assert(self.config.rep in ['sdf'])        

        if self.config.use_data_term:
            gt_sdf = batch_dict['sdf_gt']
            pred_sdf = batch_dict["sdf_pred"][..., None]

            # import ipdb; ipdb.set_trace()
            diff = (pred_sdf - gt_sdf).abs()
            scale = 1 / (gt_sdf.abs() + 1e-2)
            sdf_constraint = torch.where(gt_sdf != -100, diff * scale, torch.zeros_like(pred_sdf))
            state_info['points_loss'] = sdf_constraint.mean().item()
            state_info['points_weight'] = self.config.loss.data_term_surface_points_weight
            surface_loss = sdf_constraint.mean() * self.config.loss.data_term_surface_points_weight
            loss += surface_loss
            # batch_dict['points_loss'] = surface_loss

            if self.config.use_gradient:
                # import ipdb; ipdb.set_trace()
                gradient_constraint = torch.abs(torch.norm(batch_dict['gradient'], dim=-1)-1.).mean()
                state_info['gradient_loss'] = gradient_constraint.item()
                state_info['gradient_weight'] = self.config.loss.data_term_gradient_weight
                gradient_loss = gradient_constraint * self.config.loss.data_term_gradient_weight
                loss += gradient_loss
                # batch_dict['gradient_loss'] = gradient_loss
            
            if self.config.use_inter:

                inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
                state_info['inter_constraint_loss'] = inter_constraint.mean().item()
                state_info['inter_constraint_weight'] = self.config.loss.data_term_inter_constraint_weight
                inter_constraint_loss = inter_constraint.mean() * self.config.loss.data_term_inter_constraint_weight
                loss += inter_constraint_loss
                # batch_dict['inter_constraint_loss'] = inter_constraint_loss
                

            # import ipdb; ipdb.set_trace()
            if self.config.use_surface_normal:
                gt_normals = batch_dict['normal_gt']
                surface_num = batch_dict['surface_num'][0]
                surface_normal_constraint = 1 - F.cosine_similarity(batch_dict['gradient'][:, :surface_num, :], gt_normals, dim=-1)[..., None]
                # surface_normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(batch_dict['gradient'], gt_normals, dim=-1)[..., None],
                #                     torch.zeros_like((batch_dict['gradient'][..., :1])))
                state_info['surface_normal_loss'] = surface_normal_constraint.mean().item()
                state_info['surface_normal_weight'] = self.config.loss.data_term_surface_normal_weight
                surface_normal_loss = surface_normal_constraint.mean() * self.config.loss.data_term_surface_normal_weight
                loss += surface_normal_loss
                # batch_dict['surface_normal_loss'] = surface_normal_loss
                

        # sdf asap loss
        if self.config.use_sdf_asap:
            # sample latents
            sample_latent_space = self.config.loss.get('sample_latent_space', None)
            assert(sample_latent_space is not None)
            if sample_latent_space:

                sample_latent_space_type = self.config.loss.get('sample_latent_space_type', 'line')
                if sample_latent_space_type == 'line':
                    rand_idx = np.random.choice(B, size=(B,))
                    rand_ratio = torch.rand((B, 1), device=device)
                    time = time * rand_ratio + time[rand_idx] * (1 - rand_ratio) # (B, d)
                    batch_dict['rand_idx'] = rand_idx
                    batch_dict['rand_ratio'] = rand_ratio
                elif sample_latent_space_type == 'identical':
                    pass
                elif sample_latent_space_type == 'permutation':
                    # import ipdb; ipdb.set_trace()
                    time = time[torch.randperm(B)]
                elif sample_latent_space_type == 'random':
                    # import ipdb; ipdb.set_trace()
                    time = torch.rand_like(time)
                elif sample_latent_space_type == 'line-identical':
                    # import ipdb; ipdb.set_trace()
                    rand_idx = np.random.choice(B, size=(B,))
                    rand_ratio = torch.rand((B, 1), device=device)
                    time_line = time * rand_ratio + time[rand_idx] * (1 - rand_ratio) # (B, d)
                    # batch_dict['rand_idx'] = rand_idx
                    # batch_dict['rand_ratio'] = rand_ratio
                    time = torch.cat([time, time_line], dim=0)
                else:
                    raise NotImplementedError

            use_approx = self.config.loss.get('use_approx', False)
            if use_approx:
                sdf_asap_loss = self.get_sdf_asap_loss_approx(time, self.config.loss, batch_dict=batch_dict)
            else:
                sdf_asap_loss = self.get_sdf_asap_loss(time, self.config.loss, batch_dict=batch_dict)
            state_info['sdf_asap_loss'] = sdf_asap_loss.mean().item()
            state_info['asap_weight'] = self.config.sdf_asap_weight_epoch
            sdf_asap_loss = sdf_asap_loss.mean() * self.config.sdf_asap_weight_epoch
            loss += sdf_asap_loss
            # batch_dict['sdf_asap_loss'] = sdf_asap_loss

        state_info['loss'] = loss.item()

        return loss

    def get_PD_loss(self, time, state_info):
        # import ipdb; ipdb.set_trace()
        batch_size = time.shape[0]
        sdf_topo_loss = torch.zeros(1).to(time.device) 

        if self.config.loss.topology_PD_loss_type == 'random':
            time = (np.random.rand()*torch.ones_like(time)).to(time.device) 
        elif self.config.loss.topology_PD_loss_type == 'identical':
            pass
        else:
            raise NotImplementedError
        
        x_range, y_range, z_range = self.config.loss.get('x_range', [-1, 1]), self.config.loss.get('y_range', [-1, 1]), self.config.loss.get('z_range', [-1, 1])
        sdfs = implicit_utils.sdf_decode_from_meshgrid(self.config, self, time, resolution=self.meshgrid_size, max_batch=int(2 ** 18), x_range=x_range, y_range=y_range, z_range=z_range)
        # sdf.shape = (B, meshgrid_size, meshgrid_size, meshgrid_size)
        dgms = self.get_ndpd(-sdfs)
        # ----------------------------------------------- old group_num 相邻, 隔一个, 隔两个 ------------------------------------------------------------ #
        # Group_num = 3
        # for idx in range(Group_num):
        #     chamfer_distance_1 = chamfer_distance(dgms[0][idx+1:], dgms[0][:-idx-1], batch_reduction='mean')[0]
        #     chamfer_distance_2 = chamfer_distance(dgms[1][idx+1:], dgms[1][:-idx-1], batch_reduction='mean')[0]
        #     chamfer_distance_3 = chamfer_distance(dgms[2][idx+1:], dgms[2][:-idx-1], batch_reduction='mean')[0]  
        #     sdf_topo_loss += (chamfer_distance_1 + chamfer_distance_2 + chamfer_distance_3)
        #     # import ipdb; ipdb.set_trace()
        # ----------------------------------------------- old group_num 相邻, 隔一个, 隔两个 ------------------------------------------------------------ #


        # ----------------------------------------------- new ------------------------------------------------------------ #
        GT_pd_h0 = self.GT_pd_h0.to(time.device).unsqueeze(0).expand(batch_size, -1, -1, -1)
        GT_pd_h1 = self.GT_pd_h1.to(time.device).unsqueeze(0).expand(batch_size, -1, -1, -1)
        GT_pd_h2 = self.GT_pd_h2.to(time.device).unsqueeze(0).expand(batch_size, -1, -1, -1)
        GT_Frames = GT_pd_h0.shape[1]
        h0 = dgms[0].unsqueeze(1).expand(-1, GT_Frames, -1, -1)
        h1 = dgms[1].unsqueeze(1).expand(-1, GT_Frames, -1, -1)
        h2 = dgms[2].unsqueeze(1).expand(-1, GT_Frames, -1, -1)
        try:
            if h0.shape[2] and GT_pd_h0.shape[2]:
                chamfer_distance_1 = chamfer_distance(
                                            h0.contiguous().view(-1, h0.shape[2], 2), 
                                            GT_pd_h0.contiguous().view(-1, GT_pd_h0.shape[2], 2), 
                                            batch_reduction='mean'
                                        )[0]
                sdf_topo_loss += chamfer_distance_1
            if h1.shape[2] and GT_pd_h1.shape[2]:
                chamfer_distance_2 = chamfer_distance(
                                            h1.contiguous().view(-1, h1.shape[2], 2), 
                                            GT_pd_h1.contiguous().view(-1, GT_pd_h1.shape[2], 2), 
                                            batch_reduction='mean'
                                        )[0]
                sdf_topo_loss += chamfer_distance_2
            if h2.shape[2] and GT_pd_h2.shape[2]:
                chamfer_distance_3 = chamfer_distance(
                                            h2.contiguous().view(-1, h2.shape[2], 2), 
                                            GT_pd_h2.contiguous().view(-1, GT_pd_h2.shape[2], 2), 
                                            batch_reduction='mean'
                                        )[0]
                sdf_topo_loss += chamfer_distance_3
        except:
            import ipdb; ipdb.set_trace()
            # 有时候会遇到h2没有点 这是为什么?? 
            
        state_info['sdf_topo_loss'] = sdf_topo_loss.item()
        state_info['sdf_topo_weight'] = self.config.loss.Topology_loss_coefficient
        sdf_topo_loss = sdf_topo_loss * self.config.loss.Topology_loss_coefficient
        
        return sdf_topo_loss


    def analysis(self, batch_vecs, batch_dict, config, state_info):
        use_approx = config.loss.get('use_approx', False)
        if use_approx:
            traces = self.get_sdf_asap_loss_approx(batch_vecs, config.loss, batch_dict=batch_dict)
        else:
            traces = self.get_sdf_asap_loss(batch_vecs, config.loss, batch_dict=batch_dict)
        batch_dict['traces'] = traces


    def sdf_to_mesh2d(self, batch_vecs, N=128):
        """ convert sdf to 2d mesh using grid sampling, the range is [-1, 1]
        Args:
            batch_vecs: (B, d)
            N: resolution of sampled sdf
        """
        B, S = batch_vecs.shape[0], N * N
        image_range = np.array([-1, 1])
        pixel_size = (image_range[1] - image_range[0]) / N

        # use corner instead of middle
        x = torch.linspace(-1, 1, N)
        y = torch.linspace(-1, 1, N)
        # grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        grid_x, grid_y = torch.meshgrid(x, y)
        samples = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).to(batch_vecs.device) # (S, 2)

        sdf_pred = self.decoder(repeat(batch_vecs, 'B d -> (B S) d', S=S),
                                repeat(samples,    'S d -> (B S) d', B=B))
        sdf_pred = sdf_pred.reshape(B, N, N)

        contours_pred_list = []
        # x = [i for i in range(sdf_pred.shape[-1])]
        # y = [i for i in range(sdf_pred.shape[-1])]

        # import ipdb; ipdb.set_trace()
        for b in range(B):
            # contours_pred_b = marching_square(x, y, sdf_pred[b].detach().cpu().numpy(), 0.005)
            # contours_pred_b = contours_process(contours_pred_b)
            # contours_pred_b = find_contours(self.decoder, batch_vecs[0], 128, False, True, False)
            thresh = 0.0
            contours_pred_b = measure.find_contours(sdf_pred[b].detach().cpu().numpy(), thresh)
            while contours_pred_b == []:
                import ipdb; ipdb.set_trace()
                thresh += 0.01
                contours_pred_b = measure.find_contours(sdf_pred[b].detach().cpu().numpy(), thresh)
            contours_pred_list.append(contours_pred_b)
        return contours_pred_list

 
    def get_jac_x_disc(self, sdf_cur, x, z, eps=1e-3):
        """ compute jac of sdf_cur w.r.t. x
        Args:
            sdf_cur: (n=n1+...+nB, 1)
            x: (n=n1+...+nB, nx=2)
            z: (n=n1+...+nB, nz=d)
        Returns:
            jac_x: (n, nx)
        """
        device = x.device
        assert(sdf_cur.shape[0] == x.shape[0] == z.shape[0])
        n, nz, nx = z.shape[0], z.shape[1], x.shape[1]

        # compute jac_x
        dx = torch.kron(torch.eye(nx), torch.ones(n, 1)).reshape(nx, n, nx).to(device) * eps # (nx, n, nx)
        x_new = x[None, :, :].repeat(nx, 1, 1) + dx # (nx, n, nx)
        z_new = z[None, :, :].repeat(nx, 1, 1) # (nx, n, nz)

        sdf_new = self.decoder(z_new.reshape(nx * n, nz), x_new.reshape(nx * n, nx)) # (nx * n, 1)
        jac_x = (sdf_new - sdf_cur[None, :, :].repeat(nx, 1, 1).reshape(nx * n, 1)) / eps # (nx * n, 1)
        jac_x = jac_x.reshape(nx, n).T # (n, nx)
        return jac_x


    def get_jac_z_disc(self, sdf_cur, x, z, eps=1e-3):
        """ compute jac of sdf_cur w.r.t. z
        Args:
            sdf_cur: (n=n1+...+nB, 1)
            x: (n=n1+...+nB, nx=2)
            z: (n=n1+...+nB, nz=d)
        Returns:
            jac_z: (n, nz)
        """
        device = x.device
        assert(sdf_cur.shape[0] == x.shape[0] == z.shape[0])
        n, nz, nx = z.shape[0], z.shape[1], x.shape[1]

        # compute jac_z
        dz = torch.kron(torch.eye(nz), torch.ones(n, 1)).reshape(nz, n, nz).to(device) * eps # (nz, n, nz)
        x_new = x[None, :, :].repeat(nz, 1, 1) # (nz, n, nx)
        z_new = z[None, :, :].repeat(nz, 1, 1) + dz # (nz, n, nz)

        sdf_new = self.decoder(z_new.reshape(nz * n, nz), x_new.reshape(nz * n, nx)) # (nz * n, 1)
        jac_z = (sdf_new - sdf_cur[None, :, :].repeat(nz, 1, 1).reshape(nz * n, 1)) / eps # (nz * n, 1)
        jac_z = jac_z.reshape(nz, n).T # (n, nz)
        return jac_z


    def get_jacobian_disc(self, sdf_cur, x, z, eps_x=1e-3, eps_z=1e-3):
        """
        Args:
            sdf_cur: (n=n1+...+nB, 1)
            x: (n=n1+...+nB, nx=2)
            z: (n=n1+...+nB, nz=d)
        Returns:
            jac_x: (n, nx)
            jac_z: (n, nz)
        """
        jac_x = self.get_jac_x_disc(sdf_cur, x, z, eps=eps_x)
        jac_z = self.get_jac_z_disc(sdf_cur, x, z, eps=eps_z)
        return jac_x, jac_z


    def create_mesh2d(self, batch_vecs, cfg, batch_dict=None):
        """
        Args:
            batch_vecs: (B, d)
            N: resolution of sampled sdf
        Returns:
            batch_idx: (n1+...+nB, )
            contour_pts: (n1+...+nB, 2)
            contour_ids: list of (n_edges, 2)
        """
        device = batch_vecs.device
        N = cfg.sdf_grid_size
        use_eval_mode_sdf_to_mesh2d = cfg.get('use_eval_mode_sdf_to_mesh2d', False)

        # XXXXXX extract mesh from sdf XXXXXX
        if use_eval_mode_sdf_to_mesh2d:
            self.decoder.eval()

        contours_pred_list = self.sdf_to_mesh2d(batch_vecs, N=N)

        if use_eval_mode_sdf_to_mesh2d:
            self.decoder.train()

        # XXXXXX extract contour points XXXXXX
        B = len(contours_pred_list)
        batch_idx = []
        contour_pts = []
        contour_ids = []
        for b in range(B):
            idx = np.argmax([len(v) for v in contours_pred_list[b]])
            # IMPORTANT: find_contours only works on inner edges, so only (N-1) intervals NOT N.
            contour_pts_b = contours_pred_list[b][idx] / (N - 1) * 2 - 1 # (n_b, 2)

            # create contour_pts_b, NOTE: remove contour points that are too close (within threshold 0.0001)
            aug_contour = np.concatenate((contour_pts_b, contour_pts_b[0:1])) # closed curve, [0] == [-1]
            uni_mask = np.linalg.norm(aug_contour[1:] - aug_contour[:-1], axis=-1) >= 0.0001 # dist between consecutive points >= thres
            contour_pts_b = aug_contour[1:][uni_mask]
            contour_pts_b = torch.from_numpy(contour_pts_b.astype(np.float32)).to(device) # (n_b, 2)
            n_b = contour_pts_b.shape[0]

            # create contour_ids_b, NOTE: {num_ring}-ring neighbor
            num_ring = cfg.get('num_ring', 2)
            if num_ring != -1:
                es_n_b = np.arange(n_b)
                contour_ids_b = []
                for nr in range(1, num_ring + 1):
                    contour_ids_b.append( np.stack((es_n_b, (es_n_b + nr) % n_b)) )
                    contour_ids_b.append( np.stack((es_n_b, (es_n_b - nr) % n_b)) )
                contour_ids_b = np.concatenate(contour_ids_b, axis=-1).T # (num_ring * n_b, 2)
            else:
                # do triangulate
                es_n_b = np.arange(n_b)
                et_n_b = (es_n_b + 1) % n_b
                segments = np.stack((es_n_b, et_n_b)).T

                # https://rufat.be/triangle/API.html
                tri_dict = tr.triangulate(tri={
                    'vertices': contour_pts_b.detach().cpu().numpy(),
                    'segments': segments,
                }, opts='-p -S0') # opts='-p -D -S0'
                verts = tri_dict['vertices'].astype(np.float32)
                faces = tri_dict['triangles']

                contour_pts_b = torch.from_numpy(verts).to(device)
                contour_ids_b = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
                # It is likely tr.triangulate will remove some points
                es_n_b_new = np.arange(contour_pts_b.shape[0])
                et_n_b_new = (es_n_b_new + 1) % contour_pts_b.shape[0]
                contour_ids_b = np.concatenate((contour_ids_b, np.stack((es_n_b_new, et_n_b_new)).T), axis=0) # NOTE: there are some duplicated rows but later adj can deal with it

            # create batch_idx_b
            batch_idx_b = torch.ones_like(contour_pts_b[:, 0]) * b # (n_b,)

            contour_pts.append(contour_pts_b)
            contour_ids.append(contour_ids_b)
            batch_idx.append(batch_idx_b)

        batch_idx = torch.cat(batch_idx) # (n1+...+nB)
        contour_pts = torch.cat(contour_pts) # (n1+...+nB, 2)

        return batch_idx, contour_pts, contour_ids

    def get_sdf_asap_loss(self, batch_vecs, cfg, batch_dict=None):
        """
        Args:
            batch_vecs: (B, d)
            N: resolution of sampled sdf
        """
        # import ipdb; ipdb.set_trace()
        device = batch_vecs.device
        B = batch_vecs.shape[0]

        with torch.no_grad():
            batch_verts_idx, batch_faces_idx, batch_verts, batch_faces = self.extract_iso_surface(batch_vecs, cfg, batch_dict=None)
        # import ipdb; ipdb.set_trace()
        batch_vecs_expand = []
        for b in range(B):
            n_b = torch.where(batch_verts_idx == b)[0].shape[0]
            batch_vecs_expand.append(batch_vecs[b:(b+1)].repeat(n_b, 1))
        batch_vecs_expand = torch.cat(batch_vecs_expand) # (n1+...+nB, d)

        # XXXXXX compute gradient XXXXXX
        batch_verts = batch_verts.clone().detach().requires_grad_(True) # (n1+...+nB, 3)
        batch_vecs_expand = batch_vecs_expand.clone().detach().requires_grad_(True) # (n1+...+nB, d)
        
        iso_sdf_pred = self.decoder(batch_vecs_expand, batch_verts) # (n1+...+NB, 1)

        fx = gradient(iso_sdf_pred, batch_verts) # (n1+...+nB, 3)
        fz = gradient(iso_sdf_pred, batch_vecs_expand) # (n1+...+nB, d)

        # XXXXXX compute regularization loss XXXXXX
        trace_list = []

        for b in range(B):
            batch_verts_mask = (batch_verts_idx == b)
            batch_faces_mask = (batch_faces_idx == b)
            verts_b = batch_verts[batch_verts_mask]
            faces_b = batch_faces[batch_faces_mask]
            n_b = verts_b.shape[0]

            # compute C
            fx_b = fx[batch_verts_mask] # (n_b, 3)
            C0, C1, C_vals = [], [], []
            lin_b = torch.arange(n_b, device=device)
            C0 = torch.stack((lin_b, lin_b, lin_b), dim=0).T.reshape(-1)
            C1 = torch.stack((lin_b * 3, lin_b * 3 + 1, lin_b * 3 + 2), dim=0).T.reshape(-1)
            C_vals = fx_b.reshape(-1)
            C_indices, C_vals = ts.coalesce([C0, C1], C_vals, n_b, n_b * 3)
            C = torch.sparse_coo_tensor(C_indices, C_vals, (n_b, 3*n_b))

            # compute F
            F = fz[batch_verts_mask] # (n_b, d)

            hessian_b = compute_asap3d_sparse(verts_b, faces_b, weight_asap=cfg.weight_asap) # (3*n_b, 3*n_b), sparse
            hessian_b = hessian_b.float()

            implicit_reg_type = cfg.get('implicit_reg_type', None)
            if implicit_reg_type == 'dense_inverse':
                # import ipdb; ipdb.set_trace()
                hessian_b = hessian_b.to_dense()
                hessian_b = hessian_b + cfg.mu_asap * torch.eye(n_b * 3, device=device) if cfg.get('add_mu_diag_to_hessian', True) else hessian_b
                hessian_b_pinv = torch.linalg.inv(hessian_b)
                hessian_b_pinv = (hessian_b_pinv + hessian_b_pinv.T) / 2.0 # hessian_b_pinv is symmetric

                CH = ts.spmm(C_indices, C_vals, n_b, n_b * 3, hessian_b_pinv) # (n_b, 3*n_b)
                CHCT = ts.spmm(C_indices, C_vals, n_b, n_b * 3, CH.T) # (n_b, n_b)
                CHCT = (CHCT + CHCT.T) / 2
                CHCT = CHCT + cfg.mu_asap * torch.eye(n_b, device=device) # some row of C might be 0

                CHCT_inv = torch.linalg.inv(CHCT)
                CHCT_inv = (CHCT_inv + CHCT_inv.T) / 2

                R = F.T @ CHCT_inv @ F
            else:
                raise NotImplementedError
            with autocast(enabled=False):
                e = torch.linalg.eigvalsh(R.float()).clamp(0)
            e = e ** 0.5
            trace = e.sum()
            trace_list.append(trace)

        traces = torch.stack(trace_list)
        return traces

    def extract_iso_surface(self, batch_vecs, cfg, batch_dict=None):
        """
        Args:
            batch_vecs: (B, d)
            N: resolution of sampled sdf
        Returns:
            batch_verts_idx: (n1+...+nB, )
            batch_faces_idx: (m1+...+mB, )
            batch_verts: (n1+...+nB, 3)
            batch_faces: (m1+...+mB, 3)
        """
        device = batch_vecs.device
        B = batch_vecs.shape[0]

        batch_verts_idx = []
        batch_faces_idx = []
        batch_verts = []
        batch_faces = []
        for b in range(B):
            # import ipdb; ipdb.set_trace()
            x_range, y_range, z_range = cfg.get('x_range', [-1, 1]), cfg.get('y_range', [-1, 1]), cfg.get('z_range', [-1, 1])
            verts, faces = implicit_utils.sdf_decode_mesh_from_single_lat(self.config, self, batch_vecs[b], resolution=cfg.sdf_grid_size, voxel_size=None,
                                                                          max_batch=int(2 ** 16), offset=None, scale=None, points_for_bound=None, verbose=False,
                                                                          x_range=x_range, y_range=y_range, z_range=z_range)
            # denoise mesh, remove small connected components
            # import ipdb; ipdb.set_trace()
            split_mesh_list = trimesh.graph.split(trimesh.Trimesh(vertices=verts, faces=faces), only_watertight=False, engine='scipy')
            largest_mesh_idx = np.argmax([split_mesh.vertices.shape[0] for split_mesh in split_mesh_list])
            verts = np.asarray(split_mesh_list[largest_mesh_idx].vertices)
            faces = np.asarray(split_mesh_list[largest_mesh_idx].faces)

            if cfg.get('simplify_mesh', True):
                mesh_sim = trimesh.Trimesh(vertices=verts, faces=faces).simplify_quadratic_decimation(2000)
                verts = mesh_sim.vertices
                faces = mesh_sim.faces
                # import ipdb; ipdb.set_trace()
                # mesh_sim.export('/home/xiaoyan/3D/4DRecon/4DRep_DeepSDF3D/test/mesh_sim.obj')

            verts = torch.from_numpy(verts).float().to(device)
            faces = torch.from_numpy(faces).long().to(device)

            batch_verts_idx.append(torch.ones_like(verts[:, 0]) * b) # (n_b,)
            batch_faces_idx.append(torch.ones_like(faces[:, 0]) * b) # (m_b,)
            batch_verts.append(verts) # (n_b, 3)
            batch_faces.append(faces) # (m_b, 3)

        batch_verts_idx = torch.cat(batch_verts_idx) # (n1+...+nB)
        batch_faces_idx = torch.cat(batch_faces_idx) # (m1+...+mB)
        batch_verts = torch.cat(batch_verts) # (n1+...+nB, 3)
        batch_faces = torch.cat(batch_faces) # (m1+...+mB, 3)

        return batch_verts_idx, batch_faces_idx, batch_verts, batch_faces

    def get_sdf_asap_loss_2D(self, batch_vecs, cfg, batch_dict=None):
        """
        Args:
            batch_vecs: (B, d)
            N: resolution of sampled sdf
        """
        device = batch_vecs.device
        B = batch_vecs.shape[0]

        batch_idx, contour_pts, contour_ids = self.create_mesh2d(batch_vecs, cfg, batch_dict=batch_dict)

        batch_vecs_expand = []
        for b in range(B):
            n_b = torch.where(batch_idx == b)[0].shape[0]
            batch_vecs_expand.append(batch_vecs[b:(b+1)].repeat(n_b, 1))
        batch_vecs_expand = torch.cat(batch_vecs_expand) # (n1+...+nB, d)

        contour_pts = contour_pts.clone().detach().requires_grad_(True) # (n1+...+nB, 2)
        batch_vecs_expand = batch_vecs_expand.clone().detach().requires_grad_(True) # (n1+...+nB, d)

        # XXXXXX compute gradient XXXXXX
        # NOTE: disable dropout with eval mode
        use_eval_mode_get_jacobian = cfg.get('use_eval_mode_get_jacobian', False)
        if use_eval_mode_get_jacobian:
            self.decoder.eval()

        sdf_pred = self.decoder(batch_vecs_expand, contour_pts) # (n1+...+NB, 1)
        use_discrete_jac = cfg.get('use_discrete_jac', False)
        eps_x = cfg.get('eps_x', 0.001)
        eps_z = cfg.get('eps_z', 0.001)
        if use_discrete_jac:
            fx, fz = self.get_jacobian_disc(sdf_pred, contour_pts, batch_vecs_expand, eps_x=eps_x, eps_z=eps_z)
        else:
            fx = gradient(sdf_pred, contour_pts) # (n1+...+nB, 2)
            fz = gradient(sdf_pred, batch_vecs_expand) # (n1+...+nB, d)

        detach_fx = cfg.get('detach_fx', False)
        if detach_fx:
            fx = fx.detach()

        if use_eval_mode_get_jacobian:
            self.decoder.train()

        # XXXXXX compute regularization loss XXXXXX
        trace_list = []
        for b in range(B):
            batch_mask = (batch_idx == b)
            contour_pts_b = contour_pts[batch_mask]
            n_b = contour_pts_b.shape[0]
            contour_ids_b = contour_ids[b]

            # compute C
            fx_b = fx[batch_mask] # (n_b, 2)
            C0, C1, C_vals = [], [], []
            lin_b = torch.arange(n_b).to(device)
            C0 = torch.stack((lin_b, lin_b), dim=0).T.reshape(-1)
            C1 = torch.stack((lin_b * 2, lin_b * 2 + 1), dim=0).T.reshape(-1)
            C_vals = fx_b.reshape(-1).double()
            C_indices, C_vals = ts.coalesce([C0, C1], C_vals, n_b, n_b * 2)

            # compute F
            F = fz[batch_mask] # (n_b, d)

            # compute hessian
            use_diffL = cfg.get('use_diffL', False)
            if use_diffL:
                contour_pts_b = contour_pts_b - sdf_pred[batch_mask] * fx_b
                hessian_b = func_asap2d_sparse_diff(contour_ids_b, contour_pts_b, weight_asap=cfg.weight_asap)
            else:
                hessian_b = func_asap2d_sparse(contour_ids_b, contour_pts_b, weight_asap=cfg.weight_asap)
            hessian_b = hessian_b.to_dense()

            # compute hessian_b_pinv
            hessian_b = hessian_b + cfg.mu_asap * torch.eye(n_b * 2).to(device)
            hessian_b_pinv = torch.linalg.inv(hessian_b)
            hessian_b_pinv = (hessian_b_pinv + hessian_b_pinv.T) / 2.0

            # hessian_b_pinv is symmetric
            CH = ts.spmm(C_indices, C_vals, n_b, n_b * 2, hessian_b_pinv) # (n_b, 2*n_b)
            CHCT = ts.spmm(C_indices, C_vals, n_b, n_b * 2, CH.T) # (n_b, n_b)
            CHCT = (CHCT + CHCT.T) / 2
            CHCT = CHCT + cfg.mu_asap * torch.eye(n_b).to(device) # some row of C might be 0

            CHCT_inv = torch.linalg.inv(CHCT)
            CHCT_inv = (CHCT_inv + CHCT_inv.T) / 2
            CHCT_inv = CHCT_inv.float()

            R = F.T @ CHCT_inv @ F
            e = torch.linalg.eigvalsh(R).clamp(0)

            e = e ** 0.5
            trace = e.sum()
            trace_list.append(trace)

            # DEBUG START
            # idx = batch_dict['idx'][b].item()
            # dump_dict = {
            #     'idx': idx, 'eps_x': eps_x, 'eps_z': eps_z, 'vertices_b': batch_dict['vertices'][b].detach().cpu().numpy(),
            #     'contour_pts_b': contour_pts_b.detach().cpu().numpy(), 'contour_ids_b': contour_ids_b,
            #     'point_samples_b': batch_dict['point_samples'][b].detach().cpu().numpy(),
            #     'sdf_samples_b': batch_dict['sdf_samples'][b].detach().cpu().numpy(),
            #     'fx_b': fx_b.detach().cpu().numpy(), 'fz_b': F.detach().cpu().numpy(),
            # }
            # import pickle
            # with open(f'/scratch/cluster/yanght/Projects/GenCorres/ARAP2D/gencorres2d/work_dir/Human2D/dense/hand_only/arapMring/debug/dump/{idx}.pkl', 'wb') as f:
            #     pickle.dump(dump_dict, f)
            # from IPython import embed; embed()
            # DEBUG END


        traces = torch.stack(trace_list)
        return traces


    def get_sdf_asap_loss_approx(self, batch_vecs, cfg, batch_dict=None):
        """
        Args:
            batch_vecs: (B, d)
            N: resolution of sampled sdf
        """
        device = batch_vecs.device
        B = batch_vecs.shape[0]

        batch_idx, contour_pts, contour_ids = self.create_mesh2d(batch_vecs, cfg, batch_dict=batch_dict)

        batch_vecs_expand = []
        for b in range(B):
            n_b = torch.where(batch_idx == b)[0].shape[0]
            batch_vecs_expand.append(batch_vecs[b:(b+1)].repeat(n_b, 1))
        batch_vecs_expand = torch.cat(batch_vecs_expand) # (n1+...+nB, d)

        contour_pts = contour_pts.clone().detach().requires_grad_(True) # (n1+...+nB, 2)
        batch_vecs_expand = batch_vecs_expand.clone().detach().requires_grad_(True) # (n1+...+nB, d)

        # XXXXXX compute gradient XXXXXX
        # NOTE: disable dropout with eval mode
        use_eval_mode_get_jacobian = cfg.get('use_eval_mode_get_jacobian', False)
        if use_eval_mode_get_jacobian:
            self.decoder.eval()

        sdf_pred = self.decoder(batch_vecs_expand, contour_pts) # (n1+...+NB, 1)
        use_discrete_jac = cfg.get('use_discrete_jac', False)
        eps_x = cfg.get('eps_x', 0.001)
        eps_z = cfg.get('eps_z', 0.001)
        if use_discrete_jac:
            fx, fz = self.get_jacobian_disc(sdf_pred, contour_pts, batch_vecs_expand, eps_x=eps_x, eps_z=eps_z)
        else:
            fx = gradient(sdf_pred, contour_pts) # (n1+...+nB, 2)
            fz = gradient(sdf_pred, batch_vecs_expand) # (n1+...+nB, d)

        detach_fx = cfg.get('detach_fx', False)
        if detach_fx:
            fx = fx.detach()

        if use_eval_mode_get_jacobian:
            self.decoder.train()

        # XXXXXX compute regularization loss XXXXXX
        trace_list = []
        for b in range(B):
            batch_mask = (batch_idx == b)
            contour_pts_b = contour_pts[batch_mask]
            n_b = contour_pts_b.shape[0]
            contour_ids_b = contour_ids[b]

            # compute C
            fx_b = fx[batch_mask] # (n_b, 2)
            C0, C1, C_vals = [], [], []
            lin_b = torch.arange(n_b).to(device)
            C0 = torch.stack((lin_b, lin_b), dim=0).T.reshape(-1)
            C1 = torch.stack((lin_b * 2, lin_b * 2 + 1), dim=0).T.reshape(-1)
            C_vals = fx_b.reshape(-1).double()
            C_indices, C_vals = ts.coalesce([C0, C1], C_vals, n_b, n_b * 2)
            C = torch.sparse_coo_tensor(C_indices, C_vals, (n_b, 2*n_b))

            # compute F
            F = fz[batch_mask].double() # (n_b, d)

            use_diffL = cfg.get('use_diffL', False)
            if use_diffL:
                contour_pts_b = diff_sdf2verts(sdf_pred[batch_mask], contour_pts_b, fx_b)

            hessian_b = func_asap2d_sparse_diff(contour_ids_b, contour_pts_b, weight_asap=cfg.weight_asap) # (2*n_b, 2*n_b)
            hessian_b = hessian_b.double()

            add_mu_diag_to_hessian = cfg.get('add_mu_diag_to_hessian', True)
            if add_mu_diag_to_hessian:
                rowId = torch.arange(2*n_b, device=device)
                sparse_eye = torch.sparse_coo_tensor(torch.stack((rowId, rowId)), torch.ones_like(rowId) * cfg.mu_asap, (2 * n_b, 2 * n_b))
                hessian_b = hessian_b + sparse_eye

            num_basis = cfg.get('num_basis', 100)

            R = implicit_reg(F, C, hessian_b, num_basis, 1e-10)

            e = torch.linalg.eigvalsh(R).clamp(0)
            e = e ** 0.5
            trace = e.sum()
            trace_list.append(trace)

            # DEBUG START
            # idx = batch_dict['idx'][b].item()
            # dump_dict = {
            #     'idx': idx, 'eps_x': eps_x, 'eps_z': eps_z, 'vertices_b': batch_dict['vertices'][b].detach().cpu().numpy(),
            #     'contour_pts_b': contour_pts_b.detach().cpu().numpy(), 'contour_ids_b': contour_ids_b,
            #     'point_samples_b': batch_dict['point_samples'][b].detach().cpu().numpy(),
            #     'sdf_samples_b': batch_dict['sdf_samples'][b].detach().cpu().numpy(),
            #     'fx_b': fx_b.detach().cpu().numpy(), 'fz_b': F.detach().cpu().numpy(),
            # }
            # import pickle
            # with open(f'/scratch/cluster/yanght/Projects/GenCorres/ARAP2D/gencorres2d/work_dir/Human2D/dense/hand_only/arap2ring/arapSvd/debug/dump/{idx}.pkl', 'wb') as f:
            #     pickle.dump(dump_dict, f)
            # from IPython import embed; embed()
            # DEBUG END

        traces = torch.stack(trace_list)
        return traces


