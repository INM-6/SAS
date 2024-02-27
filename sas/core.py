"""
SAS - Assessing the similarity of real matrices with arbitrary shape.
Copyright (C) 2024 Forschungszentrum Juelich GmbH, IAS-6

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.

SPDX-License-Identifier: GPL-3.0-or-later
"""

import numpy as np
from .utils import angle


def compare(matrix_a, matrix_b, method='match_values'):
    """
    Compares two matrices using SVD and calculates their similarity score.

    Parameters
    ----------
    matrix_a : ndarray
        First input matrix.
    matrix_b : ndarray
        Second input matrix.

    Returns
    -------
    float
        Similarity score between the input matrices.
    """

    U_a, S_a, V_at = np.linalg.svd(matrix_a)
    U_b, S_b, V_bt = np.linalg.svd(matrix_b)

    # if the matrices are rectangular, disregard the singular vectors of the larger singular matrix that map to 0
    dim_0, dim_1 = matrix_a.shape
    if dim_0 < dim_1:
        V_at = V_at[:dim_0, :]
        V_bt = V_bt[:dim_0, :]
    elif dim_0 > dim_1:
        U_a = U_a[:, :dim_1]
        U_b = U_b[:, :dim_1]

    if method == 'match_values':

        angles_noflip = (angle(U_a, U_b, method='columns') + angle(V_at, V_bt, method='rows')) / 2
        angles_flip = np.pi - angles_noflip
        angles = np.minimum(angles_noflip, angles_flip)
        weights = (S_a + S_b) / 2

        # if one singular vector projects to 0, discard it
        zero_mask = (S_a > np.finfo(float).eps) | (S_b > np.finfo(float).eps)
        weights = weights[zero_mask]
        angles = angles[zero_mask]

        weights /= np.sum(weights)
        smallness = 1 - angles / (np.pi / 2)
        weighted_smallness = smallness * weights
        similarity_score = np.sum(weighted_smallness)
        return similarity_score

    elif method == 'match_vectors':

        product_U = U_a.T @ U_b
        product_V = V_at @ V_bt.T

        # catch floats that are slightly above 1 or below -1 due to machine precision
        product_U[np.isclose(product_U, 1)] = 1
        product_V[np.isclose(product_V, 1)] = 1
        product_U[np.isclose(product_U, -1)] = -1
        product_V[np.isclose(product_V, -1)] = -1

        angles_U = np.arccos(product_U)
        angles_V = np.arccos(product_V)

        angles_noflip = (angles_U + angles_V) / 2
        angles_flip = np.pi - angles_noflip

        angles = np.minimum(angles_noflip, angles_flip)

        weights = np.zeros((angles.shape[0]))
        smallness = np.zeros((angles.shape[0]))
        for row in range(angles.shape[0]):

            # calculate matching vectors
            minimal_angle = np.min(angles)
            index_0, index_1 = np.where(angles == minimal_angle)

            # if multiple pairings exist with the same minimal value, pick random one
            if (len(index_0) != 1):
                index = np.random.choice(np.arange(len(index_0)))
                index_0 = index_0[index]
                index_1 = index_1[index]

            weights[row] = np.sqrt(S_a[index_0] * S_b[index_1])
            angles = np.delete(angles, index_0, axis=0)
            angles = np.delete(angles, index_1, axis=1)

            smallness[row] = 1 - minimal_angle / (np.pi / 2)

        return np.sum(weights * smallness) / np.sum(weights)

    else:
        raise ValueError('Undefined method. Please choose either match_values or match_vectors')


def effect_size(dist_a, dist_b):
    """
    Computes the effect size between two similarity distributions.

    Parameters
    ----------
    dist_a : list or ndarray
        First similarity distribution.
    dist_b : list or ndarray
        Second similarity distribution.

    Returns
    -------
    float
        The effect size between the two distributions.
    """
    def s_pooled(sample_a, sample_b):
        n = len(sample_a)
        s = np.std(sample_a)
        nn = len(sample_b)
        sn = np.std(sample_b)
        return np.sqrt(((n - 1.) * s ** 2 + (nn - 1.) * sn ** 2) / (n + nn - 2.))
    return abs(np.mean(dist_a) - np.mean(dist_b)) / s_pooled(dist_a, dist_b)
