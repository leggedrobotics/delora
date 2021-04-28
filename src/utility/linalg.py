#!/usr/bin/env python3
# Modified from: Modar M. Alfadly, https://discuss.pytorch.org/t/covariance-and-gradient-support/16217
import torch


def cov(point_neighbors, rowvar=True):
    '''
    Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if point_neighbors.dim() > 3:
        raise ValueError('m has more than 3 dimensions')
    elif point_neighbors.dim() < 2:
        point_neighbors = point_neighbors.view(1, -1)
    if not rowvar and point_neighbors.size(0) != 1:
        point_neighbors = point_neighbors.t()
    if point_neighbors.dim() == 3:
        point_neighbors_not_zero_bool = (point_neighbors[:, 0, :] != torch.zeros(1).to(
            point_neighbors.device)) | (point_neighbors[:, 1, :] != torch.zeros(1).to(
            point_neighbors.device)) | (point_neighbors[:, 2, :] != torch.zeros(1).to(
            point_neighbors.device))
        number_neighbours = torch.sum(point_neighbors_not_zero_bool, dim=1)
        factor = torch.ones(1).to(number_neighbours.device) / (number_neighbours - 1)
        # Mean assumes that all 9 neighbours are present, but some are zero
        mean = torch.mean(point_neighbors, dim=2, keepdim=True) * point_neighbors.shape[
            2] / number_neighbours.view(-1, 1, 1)
        difference = point_neighbors - mean
        difference.permute(0, 2, 1)[~point_neighbors_not_zero_bool] = torch.zeros(1).to(
            point_neighbors.device)
        difference_transpose = difference.permute(0, 2, 1)

    else:
        factor = 1.0 / (point_neighbors.shape[1] - 1)
        mean = torch.mean(point_neighbors, dim=1, keepdim=True)
        difference = point_neighbors - mean
        difference_transpose = difference.t()

    squared_difference = difference.matmul(difference_transpose)  # .squeeze()

    return factor.view(-1, 1, 1) * squared_difference, number_neighbours
