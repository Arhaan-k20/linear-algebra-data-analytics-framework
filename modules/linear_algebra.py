import numpy as np


def calculate_covariance_matrix(data):
    data_matrix = np.asarray(data, dtype=float)

    if data_matrix.ndim != 2:
        raise ValueError("Data must be a two-dimensional matrix.")

    covariance_matrix = np.cov(
        data_matrix,
        rowvar=False
    )

    return covariance_matrix


def calculate_eigenvalues_eigenvectors(covariance_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance_matrix
    )

    sorted_indices = np.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvalues, eigenvectors