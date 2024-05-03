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


def compare(matrix_a, matrix_b, decimal_places=None):
    """
    Compares two matrices using Singular Value Decomposition (SVD) and calculates their similarity score.
    The similarity score is based on the singular values and vectors of the matrices. If the matrices have
    degeneracies in their singular values, a specialized scoring method is used to account for these.

    Parameters
    ----------
    matrix_a : ndarray
        First input matrix to be compared. Must be two-dimensional.
    matrix_b : ndarray
        Second input matrix to be compared. Must be two-dimensional and have the same shape as `matrix_a`.
    decimal_places : int, optional
        Number of decimal places to consider for the singular values. If specified, the singular values
        will be rounded to this number of decimal places before calculating the similarity score.
        This can be useful for mitigating the effects of noise that might randomly alter the ordering of
        singular values, and instead treat close singular values as degenerate.
        If None, the singular values are used as is.

    Returns
    -------
    float
        A similarity score between the input matrices, ranging from 0 to 1, where 1 indicates identical
        matrices and 0 indicates minimally similar matrices.

    Raises
    ------
    ValueError
        If the input matrices do not have the same shape.
    """

    if np.shape(matrix_a) != np.shape(matrix_b):
        raise ValueError("Matrices must be of the same shape")

    # calculate SVD of provided matrices
    U_a, S_a, V_at = np.linalg.svd(matrix_a)
    U_b, S_b, V_bt = np.linalg.svd(matrix_b)

    # if decimal places are given, singular values are only considered up to that precision
    if decimal_places:
        S_a = np.round(S_a, decimal_places)
        S_b = np.round(S_b, decimal_places)

    # calculate weights according to chosen weight function
    # discard weights belonging to singular vectors that project to 0
    weights = (S_a + S_b) / 2
    weights[(S_a < np.finfo(float).eps) | (S_b < np.finfo(float).eps)] = 0
    weights /= np.sum(weights)

    # check whether degeneracy exists
    if (np.sum(np.isclose(np.diff(S_a), 0))) or (np.sum(np.isclose(np.diff(S_b), 0))):
        non_degenerate_indices, degenerate_windows = _degeneracy(S_a, S_b)

        score = _degenerate_score(SVD_a=(U_a, S_a, V_at), SVD_b=(U_b, S_b, V_bt), weights=weights,
                                  degenerate_windows=degenerate_windows)
        if len(non_degenerate_indices) > 0:
            score += _non_degenerate_score(SVD_a=(U_a, S_a, V_at), SVD_b=(U_b, S_b, V_bt), weights=weights,
                                           non_degenerate_indices=non_degenerate_indices)
        return score
    else:
        return _non_degenerate_score(SVD_a=(U_a, S_a, V_at), SVD_b=(U_b, S_b, V_bt), weights=weights)


def _degeneracy(S_a, S_b):
    """
    Identify degenerate and non-degenerate indices in sequences `S_a` and `S_b`.

    Parameters
    ----------
    S_a : array_like
        A sequence of values to be analyzed for degeneracy.
    S_b : array_like
        Another sequence of values to be analyzed for degeneracy.

    Returns
    -------
    non_degenerate_indices : list
        A list of indices where neither `S_a` nor `S_b` are degenerate.
    degenerate_windows : list of tuples
        A list of tuples, where each tuple contains the start and stop
        indices of a degenerate window.
    """
    degenerate = False
    degenerate_windows = []
    non_degenerate_indices = []

    for index in range(len(S_a) - 1):
        if np.isclose(S_a[index + 1], S_a[index]) or np.isclose(S_b[index + 1], S_b[index]):
            if not degenerate:
                start = index
                stop = index + 1
                degenerate = True
            else:
                stop = index + 1
        else:
            if degenerate:
                degenerate_windows.append((start, stop))
                degenerate = False
            else:
                non_degenerate_indices.append(index)

    # write information on last index
    if degenerate:
        degenerate_windows.append((start, stop))
    else:
        non_degenerate_indices.append(index + 1)

    return non_degenerate_indices, degenerate_windows


def _degenerate_score(SVD_a, SVD_b, weights, degenerate_windows):
    """
    Given the Singular Value Decompositions (SVDs) of two matrices and a set of 
    degenerate windows, this function computes a similarity score based on the 
    alignment of the subspaces corresponding to each degenerate window.

    Parameters
    ----------
    SVD_a : tuple of (U_a, S_a, V_at)
        SVD components of the first matrix, where U_a is the left singular vectors,
        S_a is the diagonal matrix of singular values, and V_at is the transpose of
        the right singular vectors.
    SVD_b : tuple of (U_b, S_b, V_bt)
        SVD components of the second matrix, analogous to SVD_a.
    weights : array_like
        Weights corresponding to each singular value, used to weight the contribution
        of each degenerate window to the overall similarity score.
    degenerate_windows : list of tuples
        List of tuples where each tuple contains the start and end indices (inclusive)
        defining a window of degenerate singular values.

    Returns
    -------
    similarity_score : float
        The weighted similarity score for the degenerate subspaces, where a higher 
        score indicates greater similarity between the subspaces.
    """

    U_a, S_a, V_at = SVD_a
    U_b, S_b, V_bt = SVD_b

    similarity_score = 0

    for degenerate_window in degenerate_windows:

        U_a = U_a[:, degenerate_window[0]:degenerate_window[1] + 1]
        V_at = V_at[degenerate_window[0]:degenerate_window[1] + 1, :]
        U_b = U_b[:, degenerate_window[0]:degenerate_window[1] + 1]
        V_bt = V_bt[degenerate_window[0]:degenerate_window[1] + 1, :]
        weights = weights[degenerate_window[0]:degenerate_window[1] + 1]

        O_U_a, S_U_U, O_U_b = np.linalg.svd(U_a.T @ U_b)
        O_V_a, S_V_V, O_V_b = np.linalg.svd(V_at @ V_bt.T)

        angles_U_aligned_U = np.nan_to_num(np.arccos(S_U_U))
        angles_V_aligned_V = np.nan_to_num(np.arccos(S_V_V))

        angles_U_aligned_V = angle(V_at.T @ O_U_a, V_bt.T @ O_U_b, method='columns')
        angles_V_aligned_U = angle(U_a @ O_V_a, U_b @ O_V_b, method='columns')

        angles_U_aligned = (angles_U_aligned_U + angles_U_aligned_V) / 2
        angles_V_aligned = (angles_V_aligned_V + angles_V_aligned_U) / 2

        if np.sum(angles_U_aligned) < np.sum(angles_V_aligned):
            angles = angles_U_aligned
        else:
            angles = angles_V_aligned

        smallness = 1 - angles / (np.pi / 2)
        weighted_smallness = smallness * weights
        similarity_score += np.sum(weighted_smallness)

    return similarity_score


def _non_degenerate_score(SVD_a, SVD_b, weights, non_degenerate_indices=None):
    """
    Calculate the similarity score for non-degenerate subspaces of two matrices.

    Given the Singular Value Decompositions (SVDs) of two matrices, this function computes
    a similarity score for the non-degenerate subspaces, optionally considering only a 
    subset of indices.

    Parameters
    ----------
    SVD_a : tuple of (U_a, S_a, V_at)
        SVD components of the first matrix, where U_a is the left singular vectors,
        S_a is the diagonal matrix of singular values, and V_at is the transpose of
        the right singular vectors.
    SVD_b : tuple of (U_b, S_b, V_bt)
        SVD components of the second matrix, analogous to SVD_a.
    weights : array_like
        Weights corresponding to each singular value, used to weight the contribution
        of each index to the overall similarity score.
    non_degenerate_indices : array_like, optional
        Indices of the non-degenerate singular values. If provided, the similarity score
        will be computed using only these indices.

    Returns
    -------
    similarity_score : float
        The weighted similarity score for the non-degenerate subspaces, where a higher 
        score indicates greater similarity between the subspaces.
    """

    U_a, S_a, V_at = SVD_a
    U_b, S_b, V_bt = SVD_b

    if non_degenerate_indices:
        U_a = U_a[:, non_degenerate_indices]
        V_at = V_at[non_degenerate_indices, :]
        U_b = U_b[:, non_degenerate_indices]
        V_bt = V_bt[non_degenerate_indices, :]
        weights = weights[non_degenerate_indices]

    dim_0 = np.shape(U_a)[0]
    dim_1 = np.shape(V_at)[1]

    if dim_0 < dim_1:
        V_at = V_at[:dim_0, :]
        V_bt = V_bt[:dim_0, :]
    elif dim_0 > dim_1:
        U_a = U_a[:, :dim_1]
        U_b = U_b[:, :dim_1]

    angles_noflip = (angle(U_a, U_b, method='columns') + angle(V_at, V_bt, method='rows')) / 2
    angles_flip = np.pi - angles_noflip
    angles = np.minimum(angles_noflip, angles_flip)
    smallness = 1 - angles / (np.pi / 2)

    weighted_smallness = smallness * weights
    similarity_score = np.sum(weighted_smallness)

    return similarity_score


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
