import torch
import os
import numpy as np
from glob import glob
from scipy.spatial import cKDTree as KDTree
from matplotlib import cm



def makedirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def to_sparse(spmat):
    return torch.sparse.FloatTensor(
        torch.LongTensor([spmat.tocoo().row,
                          spmat.tocoo().col]),
        torch.FloatTensor(spmat.tocoo().data), torch.Size(spmat.tocoo().shape))


def to_edge_index(mat):
    return torch.LongTensor(np.vstack(mat.nonzero()))

def get_colors_from_diff_pc(diff_pc, min_error, max_error):
    colors = np.zeros((diff_pc.shape[0],3))
    mix = (diff_pc-min_error)/(max_error-min_error)
    mix = np.clip(mix, 0,1) #point_num
    cmap=cm.get_cmap('coolwarm')
    colors = cmap(mix)[:,0:3]
    return colors


def save_pc_with_color_into_ply(template_ply, pc, color, fn):
    plydata=template_ply
    #pc = pc.copy()*pc_std + pc_mean
    plydata['vertex']['x']=pc[:,0]
    plydata['vertex']['y']=pc[:,1]
    plydata['vertex']['z']=pc[:,2]

    plydata['vertex']['red']=color[:,0]
    plydata['vertex']['green']=color[:,1]
    plydata['vertex']['blue']=color[:,2]

    plydata.write(fn)
    plydata['vertex']['red']=plydata['vertex']['red']*0+0.7*255
    plydata['vertex']['green']=plydata['vertex']['red']*0+0.7*255
    plydata['vertex']['blue']=plydata['vertex']['red']*0+0.7*255

def compute_trimesh_chamfer(gt_points, gen_points):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.
    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)
    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)
    """ 
    # only need numpy array of points
    # gt_points_np = gt_points.vertices
    gt_points_np = gt_points.detach().cpu().numpy()
    gen_points_sampled = gen_points.detach().cpu().numpy()

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer


def get_freq_reg_mask(pos_enc_length, current_iter, total_reg_iter, max_visible=None, type='submission'):
    '''
    Returns a frequency mask for position encoding in NeRF.

    Args:
    pos_enc_length (int): Length of the position encoding.
    current_iter (int): Current iteration step.
    total_reg_iter (int): Total number of regularization iterations.
    max_visible (float, optional): Maximum visible range of the mask. Default is None. 
        For the demonstration study in the paper.

    Correspond to FreeNeRF paper:
        L: pos_enc_length
        t: current_iter
        T: total_iter

    Returns:
    jnp.array: Computed frequency or visibility mask.
    '''
    if max_visible is None:
    # default FreeNeRF
        if current_iter < total_reg_iter:
            freq_mask = np.zeros(pos_enc_length)  # all invisible
            ptr = pos_enc_length / 3 * current_iter / total_reg_iter + 1 
            ptr = ptr if ptr < pos_enc_length / 3 else pos_enc_length / 3
            int_ptr = int(ptr)
            freq_mask[: int_ptr * 3] = 1.0  # assign the integer part
            freq_mask[int_ptr * 3 : int_ptr * 3 + 3] = (ptr - int_ptr)  # assign the fractional part
            return np.clip(np.array(freq_mask), 1e-8, 1-1e-8)  # for numerical stability
        else:
            return np.ones(pos_enc_length)
    else:
        # For the ablation study that controls the maximum visible range of frequency spectrum
        freq_mask = np.zeros(pos_enc_length)
        freq_mask[: int(pos_enc_length * max_visible)] = 1.0
        return np.array(freq_mask)