import numpy as np
import sas


def test_identical_matrices():
    matrix = np.random.normal(0, 1, (100, 100))
    similarity = sas.compare(matrix, matrix)
    assert np.isclose(similarity, 1), "Identical matrices don't produce SAS=1."


def test_positive_scaling():
    matrix = np.random.normal(0, 1, (100, 100))
    similarity = sas.compare(matrix, np.random.uniform(1e-6, 1e6) * matrix)
    assert np.isclose(similarity, 1), "Positive scaling doesn't produce SAS=1."


def test_negative_scaling():
    matrix = np.random.normal(0, 1, (100, 100))
    similarity = sas.compare(matrix, -np.random.uniform(1e-6, 1e6) * matrix)
    assert np.isclose(similarity, 0), "Negative scaling doesn't produce SAS=0."
