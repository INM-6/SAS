import numpy as np


class SingularAngles():
    """
    A class used to compare the similarity of non-symmetric matrices based on Singular Value Decomposition (SVD).

    Methods
    -------
    compare(matrix_a, matrix_b):
        Compares two matrices using SVD and calculates their similarity score.

    _angle(a, b, method='columns'):
        Calculates the angles between the row or column vectors of two matrices.

    effect_size(sample_a, sample_b):
        Computes the effect size between two similarity distributions.
    """

    def __init__(self):
        super(SingularAngles, self).__init__()

    def compare(self, matrix_a, matrix_b):
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

        angles_noflip = (self._angle(U_a, U_b, method='columns') + self.angle(V_at, V_bt, method='rows')) / 2
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

    def _angle(self, a, b, method='columns'):
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

    def effect_size(self, dist_a, dist_b):
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
