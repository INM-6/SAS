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


def angle(a, b, method='columns'):
    """
     Calculates the angles between the row or column vectors of two matrices.

     Parameters
     ----------
     a : ndarray
         First input matrix.
     b : ndarray
         Second input matrix.
     method : str, optional
         Defines the direction of the vectors (either 'rows' or 'columns'), by default 'columns'.

     Returns
     -------
     ndarray
         Array of angles.
     """
    if method == 'columns':
        axis = 0
    if method == 'rows':
        axis = 1

    dot_product = np.sum(a * b, axis=axis)
    magnitude_a = np.linalg.norm(a, axis=axis)
    magnitude_b = np.linalg.norm(b, axis=axis)
    angle = np.arccos(dot_product / (magnitude_a * magnitude_b))

    mask_pos1 = np.isnan(angle) & np.isclose(dot_product, 1)
    angle[mask_pos1] = 0
    mask_neg1 = np.isnan(angle) & np.isclose(dot_product, -1)
    angle[mask_neg1] = np.pi

    return angle
