#!/usr/bin/env python
# coding=utf-8
import torch
from torch.autograd import gradcheck

import numpy as np
import scipy.sparse.linalg


def mul_mask_C(P, Q):
    ''' multiply P by Q and extract with C mask
    Args:
        P: (n, m), dense
        Q: (m, 2n), dense
    Returns:
        Ms: (n, 2n), sparse
    '''
    device = P.device
    n, m = P.shape
    assert(Q.shape[0] == m and Q.shape[1] == 2 * n)

    Ps_row = torch.arange(n, device=device)[:, None].repeat(1, m).reshape(-1) # (n*m,)
    Ps_col = torch.arange(n * m, device=device) # (n*m,)
    Ps_indices = torch.stack((Ps_row, Ps_col)) # (2, n*m)
    Ps = torch.sparse_coo_tensor(Ps_indices, P.reshape(-1), (n, m * n)) # (n, m*n)

    Qs_row = torch.arange(n * m, device=device).reshape(n, 1, m).repeat(1, 2, 1).reshape(-1) # (2*n*m,)
    Qs_col = torch.arange(2 * n, device=device).reshape(n, 2, 1).repeat(1, 1, m).reshape(-1) # (2*n*m,)
    Qs_indices = torch.stack((Qs_row, Qs_col)) # (2, 2*n*m)
    Qs = torch.sparse_coo_tensor(Qs_indices, Q.T.reshape(-1), (m * n, 2 * n)) # (m*n, 2*n)

    Ms = torch.sparse.mm(Ps, Qs) # (n, 2*n)

    return Ms


def mul_mask(P, Q, M):
    ''' multiply P by Q and extract with mask M
    Args:
        P: (n, d), dense
        Q: (d, m), dense
        M: (n, m), sparse
    Returns:
        Ms: (n, m), sparse
    '''
    device = P.device
    n, m, d = P.shape[0], Q.shape[1], P.shape[1]
    assert(P.shape[1] == Q.shape[0])
    assert(n == M.shape[0] and m == M.shape[1])
    if not M.is_coalesced():
        M = M.coalesce()
    rowId, colId = M.indices()[0], M.indices()[1]
    vals = (P[rowId] * Q[:, colId].T).sum(1)
    Ms = torch.sparse_coo_tensor(M.indices(), vals, M.shape)
    return Ms


class ImplicitReg(torch.autograd.Function):
    """ implicit regularization
    """
    @staticmethod
    def forward(ctx, F, C, L, num_basis=-1, nnz_thres=1e-5):
        """
        Args:
            F: (n, d), dense matrix
            C: (n, 2*n), sparse matrix
            L: (2*n, 2*n), sparse matrix
            num_basis: top-num_basis eigen bases
            nnz_thres: thres of regarding eigenvalues as non-zero
        Returns:
            R: (d, d), dense matrix
        """
        assert(len(L.shape) == 2)
        assert(L.shape[0] == L.shape[1])
        num_basis = L.shape[0] if num_basis == -1 else num_basis
        # non-Tensor arguments are directly stored in ctx
        # ctx.num_basis = num_basis
        # ctx.nnz_thres = nnz_thres

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        # top-q singular vectors
        # NOTE: svd_lowrank on sparse matrix
        # U, Sigma, _ = torch.svd_lowrank(L, q=num_basis) # (2n, m), (m,)
        # NOTE: eigh on dense matrix
        Sigma, U = torch.linalg.eigh(L.to_dense())
        U = U[:, -num_basis:]
        Sigma = Sigma[-num_basis:]

        # remove near-zero column and eigenvalues
        nnz_mask = Sigma >= nnz_thres
        U = U[:, nnz_mask]
        Sigma = Sigma[nnz_mask]

        # C*U*Sigma^-0.5
        CU = torch.sparse.mm(C, U) # (n, m)
        CUS = CU * Sigma.pow(-0.5) # (n, m)
 
        if max(CUS.shape) / min(CUS.shape) > 2.2:
            driver = 'gesvda' # svd: thin/wide matrix use gesvda instead of default
        else:
            driver = None
        V, Omega, Wh = torch.linalg.svd(CUS, full_matrices=False, driver=driver) # (n, m), (m,), (m, m), when n > m 

        # remove near-zero column and singular values
        omega_nnz_mask = Omega >= nnz_thres
        V = V[:, omega_nnz_mask]
        Omega = Omega[omega_nnz_mask]

        # DEBUG
        assert(torch.allclose( (V @ torch.diag(Omega.pow(-2))), (V / (Omega * Omega)) ))

        VOinv2VTF = (V / (Omega * Omega)) @ (V.T @ F) # (n, d)
        R = F.T @ VOinv2VTF

        ctx.save_for_backward(F, C, U, Sigma, CU, VOinv2VTF, L)

        return R

    @staticmethod
    def backward(ctx, grad_output):
        # Ref: https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
        F, C, U, Sigma, CU, VOinv2VTF, L = ctx.saved_tensors
        if not C.is_coalesced():
            C = C.coalesce()

        # grad_F
        grad_F = 2 * VOinv2VTF @ grad_output # (n, d)

        # grad_C
        gC = VOinv2VTF @ grad_output # (n, d)
        gC = gC @ (VOinv2VTF.T @ CU) # (n, d) @ (d, m) -> (n, m)

        SinvUT = torch.diag(1 / Sigma) @ U.T # (m, 2n)

        # grad_C = -2 * gC @ SinvUT 
        # grad_C_indices = C.indices()
        # grad_C_vals = grad_C[grad_C_indices[0], grad_C_indices[1]]
        # grad_C_sparse = torch.sparse_coo_tensor(grad_C_indices, grad_C_vals, C.shape)

        grad_C_sparse = -2 * mul_mask_C(gC, SinvUT)

        # grad_L
        Lr = (VOinv2VTF.T @ CU) @ (SinvUT) # (d, n) @ (n, m) @ (m, 2n) = (d, 2n)
        RLr = grad_output @ Lr # (d, 2n)
        grad_L_sparse = mul_mask(Lr.T, RLr, L)

        return grad_F, grad_C_sparse, grad_L_sparse, None, None


implicit_reg = ImplicitReg.apply


if __name__ == '__main__':
    torch.set_printoptions(precision=8, linewidth=240, sci_mode=False)
    # dtype = torch.float
    dtype = torch.double
    # device = torch.device("cpu")
    device = torch.device("cuda:0")  # Uncomment this to run on GPU

    # n = 3
    # torch.manual_seed(0)
    # A =  torch.rand((n, n), device=device, dtype=dtype)
    # A = A @ A.T + torch.eye(n, device=device, dtype=dtype)

    #######################################################################
    n, d = 10, 8
    import sys
    sys.path.append('../utils')
    from test_utils import random_sparse_pd_matrix
    L = random_sparse_pd_matrix(2*n, density=0.2, device=device, dtype=dtype)
    L.requires_grad_(True)

    Cd = torch.rand((n, 2), device=device, dtype=dtype)
    C0, C1, C_vals = [], [], []
    lin = torch.arange(n).to(device)
    C0 = torch.stack((lin, lin), dim=0).T.reshape(-1)
    C1 = torch.stack((lin * 2, lin * 2 + 1), dim=0).T.reshape(-1)
    C_vals = Cd.reshape(-1)
    C = torch.sparse_coo_tensor(torch.stack((C0, C1)), C_vals, (n, 2*n))
    C.requires_grad_(True)

    F = torch.rand((n, d), device=device, dtype=dtype)
    F.requires_grad_(True)
    #######################################################################

    #######################################################################
    # dump_dict = torch.load('./FCL.pt')
    # F = dump_dict['F']
    # n = F.shape[0]

    # C_indices = dump_dict['C_indices']
    # C_values = dump_dict['C_values']
    # C = torch.sparse_coo_tensor(C_indices, C_values, (n, 2*n))
    # C = C.detach().requires_grad_(True)

    # L_indices = dump_dict['L_indices']
    # L_values = dump_dict['L_values']
    # L = torch.sparse_coo_tensor(L_indices, L_values, (2*n, 2*n))
    # L = L.detach().requires_grad_(True)
    #######################################################################

    R = implicit_reg(F, C, L, 2*n, 1e-10)
    # R = implicit_reg(F, C, L, 6, 1e-10)
    R.sum().backward()
    print('-' * 30 + ' approx ' + '-' * 30)
    print(F.grad)
    print(C.grad.coalesce().to_dense())
    F.grad.zero_(); C.grad.zero_()

    Rgt = F.T @ torch.linalg.pinv(C.to_dense() @ torch.linalg.pinv(L.to_dense()) @ C.to_dense().T) @ F
    Rgt.sum().backward()
    print('-' * 30 + ' GT ' + '-' * 30)
    print(F.grad)
    print(C.grad.coalesce().to_dense())
    F.grad.zero_(); C.grad.zero_()

    # test = gradcheck(implicit_reg, (F, C, L, 2*n, 1e-10), eps=1e-6, atol=1e-4, check_sparse_nnz=True)
    # print(test)

    from IPython import embed; embed()










