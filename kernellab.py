'''Kernel function lab with Mahalanobis distance'''

import torch

# basic distance functions
def euclidean_distances(samples, centers, squared=True):
    samples_norm = torch.sum(samples**2, dim=1, keepdim=True)
    if samples is centers:
        centers_norm = samples_norm
    else:
        centers_norm = torch.sum(centers**2, dim=1, keepdim=True)
    centers_norm = torch.reshape(centers_norm, (1, -1))

    distances = samples.mm(torch.t(centers))
    distances.mul_(-2)
    distances.add_(samples_norm)
    distances.add_(centers_norm)
    if not squared:
        distances.clamp_(min=0)
        distances.sqrt_()

    return distances

def euclidean_distances_M(samples, centers, M, squared=True):
    
    samples_norm = (samples @ M)  * samples
    samples_norm = torch.sum(samples_norm, dim=1, keepdim=True)

    if samples is centers:
        centers_norm = samples_norm
    else:
        centers_norm = (centers @ M) * centers
        centers_norm = torch.sum(centers_norm, dim=1, keepdim=True)

    centers_norm = torch.reshape(centers_norm, (1, -1))

    distances = samples.mm(M @ torch.t(centers))
    distances.mul_(-2)
    distances.add_(samples_norm)
    distances.add_(centers_norm)
    if not squared:
        distances.clamp_(min=0)
        distances.sqrt_()

    return distances


# Gaussian kernel
def gaussian(samples, centers, bandwidth):
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers)
    kernel_mat.clamp_(min=0)
    gamma = 1. / (2 * bandwidth ** 2)
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat

def gaussian_M(samples, centers, bandwidth, M):
    assert bandwidth > 0
    kernel_mat = euclidean_distances_M(samples, centers, M)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat


# Laplacian kernel
def laplacian(samples, centers, bandwidth):
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, squared=False)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat

def laplacian_M(samples, centers, bandwidth, M):
    assert bandwidth > 0
    kernel_mat = euclidean_distances_M(samples, centers, M, squared=False)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat


# Laplacian kernel with smoothing parameter delta
def euclidean_distances_M_delta(samples, centers, M, delta, squared=True):
    samples_norm = (samples @ M) * samples
    samples_norm = torch.sum(samples_norm, dim=1, keepdim=True)

    if samples is centers:
        centers_norm = samples_norm
    else:
        centers_norm = (centers @ M) * centers
        centers_norm = torch.sum(centers_norm, dim=1, keepdim=True)

    centers_norm = torch.reshape(centers_norm, (1, -1))

    distances = samples.mm(M @ torch.t(centers))
    distances.mul_(-2)
    distances.add_(samples_norm)
    distances.add_(centers_norm)
    distances.add_(delta)
    if not squared:
        distances.clamp_(min=0)
        distances.sqrt_()

    return distances

def laplacian_M_delta(samples, centers, bandwidth, M, delta):
    assert bandwidth > 0
    kernel_mat = euclidean_distances_M_delta(samples, centers, M, delta, squared=False)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat


# L1 kernel
def l1_distances_M(samples, centers, M):
    diff = samples.unsqueeze(1) - centers.unsqueeze(0)
    mat = diff @ M
    distances = torch.norm(mat, p=1, dim=2)

    return distances

def l1_M(samples, centers, bandwidth, M):
    assert bandwidth > 0
    kernel_mat = l1_distances_M(samples, centers, M)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat


# Linear kernel
def linear_M(samples, centers, bandwidth, M):
    assert bandwidth > 0
    kernel_mat = samples @ M @ torch.t(centers)
    gamma = 1. / bandwidth
    kernel_mat.mul_(gamma)
    return kernel_mat


# square kernel
def square_M(samples, centers, bandwidth, M):
    assert bandwidth > 0
    kernel_mat = samples @ M @ torch.t(centers)
    gamma = 1. / bandwidth
    kernel_mat.mul_(gamma)
    kernel_mat.pow_(2)
    return kernel_mat


# cubic kernel
def cubic_M(samples, centers, bandwidth, M):
    assert bandwidth > 0
    kernel_mat = samples @ M @ torch.t(centers)
    gamma = 1. / bandwidth
    kernel_mat.mul_(gamma)
    kernel_mat.pow_(3)
    return kernel_mat


# IMQ kernel
def IMQ_M(samples, centers, bandwidth, M):
    assert bandwidth > 0
    kernel_mat = euclidean_distances_M(samples, centers, M)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(gamma)
    kernel_mat.add_(1)
    kernel_mat.pow_(-0.5)
    return kernel_mat


# Matern kernel
def Matern_M(samples, centers, bandwidth, M):
    assert bandwidth > 0
    kernel_mat = euclidean_distances_M(samples, centers, M, squared=False)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(gamma)
    kernel_mat.add_(1)
    kernel_mat.mul_(laplacian_M(samples, centers, bandwidth, M))
    return kernel_mat