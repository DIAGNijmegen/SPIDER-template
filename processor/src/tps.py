import torch
import torch.nn as nn

import itertools


class ThinPlateSpline(nn.Module):
    def __init__(self, control_points, grid_shape, elasticity=0):  # controls points = (N, 2), grid shape = (2,)
        super().__init__()

        control_points = control_points
        n_control_points = control_points.size(0)

        # Assemble matrix L for the equation weights = inv(L) * z_coords
        L = torch.zeros(n_control_points + 3, n_control_points + 3)

        # TPS equation is a + x*b + y*c + ..., so fill in ones as weights for the constant offset a
        L[:-3, -3] = 1
        L[-3, :-3] = 1

        # Fill in x and y of all control points as weights for b and c
        L[:-3, -2:] = control_points
        L[-2:, :-3] = control_points.transpose(0, 1)

        # Calculate pairwise distances between all control points and fill into L
        K = ThinPlateSpline.calculate_pairwise_distances(control_points, control_points)
        L[:-3, :-3] = K + elasticity * torch.eye(n_control_points)

        # Determine pseudo inverse of L
        self.register_buffer('L_inverse', torch.pinverse(L))

        # Prepare the output grid
        n_grid_points = grid_shape[0] * grid_shape[1]
        self.grid_points = torch.Tensor(list(itertools.product(range(grid_shape[0]), range(grid_shape[1]))))
        K_grid = ThinPlateSpline.calculate_pairwise_distances(self.grid_points, control_points)
        self.register_buffer('grid_K1xy_t', torch.cat([
            K_grid, torch.ones(n_grid_points, 1), self.grid_points
        ], dim=1).t())

        # Keep a vector of zeros in memory for padding later
        # (do this here to make sure it's properly transfered to the GPU)
        self.register_buffer('padding', torch.zeros(1, 3))

    def forward(self, control_point_heights):  # heights = (batch_size, n_control_points)
        # Use control point heights to calculate TPS weight matrix
        batch_size = control_point_heights.size(0)
        z = torch.cat([control_point_heights, self.padding.expand(batch_size, -1)], dim=1)
        weights = z.mm(self.L_inverse)

        # Apply weight matrix to calculate grid point heights
        grid_heights = weights.mm(self.grid_K1xy_t)
        return grid_heights

    @staticmethod
    def calculate_pairwise_distances(points_a, points_b):
        n_a, n_b = points_a.size(0), points_b.size(0)
        pairwise_differences = points_a.view(n_a, 1, 2) - points_b.view(1, n_b, 2)
        pairwise_distances_squared = torch.sum(pairwise_differences**2, dim=2)

        K = pairwise_distances_squared * 0.5 * torch.log(pairwise_distances_squared)  # log(sqrt(x)) = 0.5 * log(x)

        # Fix cases with r = 0, which results into 0 * log(0) = NaN, but should be 0 actually
        K[K != K] = 0

        return K


class ThinPlateSplineMask(nn.Module):
    def __init__(self, shape, control_point_spacing, elasticity=0):
        super().__init__()

        self.grid_shape = (shape[0], shape[2])
        self.control_points = torch.Tensor(list(itertools.product(
            range(control_point_spacing // 2, self.grid_shape[0] - control_point_spacing // 2 + 1, control_point_spacing),
            range(control_point_spacing // 2, self.grid_shape[1] - control_point_spacing // 2 + 1, control_point_spacing)
        )))
        self.tps = ThinPlateSpline(self.control_points, self.grid_shape, elasticity)

        voxel_heights = torch.zeros(shape)
        for xz in self.tps.grid_points:
            x, z = xz
            for y in range(shape[1]):
                voxel_heights[int(x), y, int(z)] = y

        self.register_buffer('voxel_heights', voxel_heights)

    def forward(self, heights):  # in = control point heights, out = grid point distance (signed difference) to surface (1D)
        tps_heights = self.tps(heights)
        return self.voxel_heights - tps_heights.view(-1, self.grid_shape[0], 1, self.grid_shape[1])

    @property
    def n_control_points(self):
        return self.control_points.size(0)
