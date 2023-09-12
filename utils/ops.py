#!/usr/bin/env python
# coding=utf-8
import torch
from torch.autograd import gradcheck


class SparseInvhApprox(torch.autograd.Function):
    """ inverse a sparse symmetric matrix using low-rank approximation
    """
    @staticmethod
    def forward(ctx, A, num_basis=-1, nnz_thres=1e-6):
        """
        A: (n, n), sparse symmetric matrix
        num_basis: top-num_basis eigen bases
        nnz_thres: thres of regarding eigenvalues as non-zero
        """
        assert(len(A.shape) == 2)
        assert(A.shape[0] == A.shape[1])
        num_basis = A.shape[0] if num_basis == -1 else num_basis
        # non-Tensor arguments are directly stored in ctx
        # ctx.num_basis = num_basis
        # ctx.nnz_thres = nnz_thres

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        U, Sigma, V = torch.svd_lowrank(A, q=num_basis)

        # remove near-zero column and eigenvalues
        nnz_mask = Sigma >= nnz_thres
        U = U[:, nnz_mask]
        Sigma = Sigma[nnz_mask]

        Ainv = U @ torch.diag(1 / Sigma) @ U.T

        ctx.save_for_backward(U, Sigma, A, Ainv)

        return Ainv

    @staticmethod
    def backward(ctx, grad_output):
        # Ref: https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
        U, Sigma, A, Ainv = ctx.saved_tensors

        if not A.is_coalesced():
            A = A.coalesce()

        grad_input = - Ainv @ grad_output @ Ainv

        grad_indices = A.indices()
        grad_vals = grad_input[grad_indices[0], grad_indices[1]]
        grad_input_sparse = torch.sparse_coo_tensor(grad_indices, grad_vals, A.shape)

        return grad_input_sparse, None, None


class DenseInvhApprox(torch.autograd.Function):
    """ inverse a dense symmetric matrix using low-rank approximation
    """
    @staticmethod
    def forward(ctx, A, num_basis=-1, nnz_thres=1e-6):
        """
        A: (n, n), dense symmetric matrix
        num_basis: top-num_basis eigen bases
        nnz_thres: thres of regarding eigenvalues as non-zero
        """
        assert(len(A.shape) == 2)
        assert(A.shape[0] == A.shape[1])
        num_basis = A.shape[0] if num_basis == -1 else num_basis
        # non-Tensor arguments are directly stored in ctx
        # ctx.num_basis = num_basis
        # ctx.nnz_thres = nnz_thres

        Sigma, U = torch.linalg.eigh(A)

        # select top-num_basis eigen bases
        U = U[:, -num_basis:]
        Sigma = Sigma[-num_basis:]

        # remove near-zero column and eigenvalues
        nnz_mask = Sigma >= nnz_thres
        U = U[:, nnz_mask]
        Sigma = Sigma[nnz_mask]

        Ainv = U @ torch.diag(1 / Sigma) @ U.T

        ctx.save_for_backward(U, Sigma)

        return Ainv

    @staticmethod
    def backward(ctx, grad_output):
        # Ref: https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
        U, Sigma = ctx.saved_tensors

        M = U.T @ grad_output @ U
        US = U * (1 / Sigma)
        grad_input = - US @ M @ US.T

        return grad_input, None, None


dense_invh_approx = DenseInvhApprox.apply
sparse_invh_approx = SparseInvhApprox.apply


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
    n = 10
    from test_utils import random_sparse_pd_matrix
    A = random_sparse_pd_matrix(n, density=0.2)
    A.requires_grad_(True)
    Ad = A.to_dense().detach()
    Ad.requires_grad_(True)

    # test = gradcheck(dense_invh_approx, (A,), eps=1e-6, atol=1e-4)
    # print(test)

    # loss = dense_invh_approx(A).sum()
    loss = sparse_invh_approx(A)
    loss = loss.to_dense().sum()
    loss.backward()
    print(A.grad)

    loss = torch.linalg.inv(Ad).sum()
    loss.backward()
    print(Ad.grad)

    from IPython import embed; embed()










