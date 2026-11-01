import numpy as np


def calculate_pca(standardized_data):
    data_matrix = np.asarray(
        standardized_data,
        dtype=float
    )

    covariance_matrix = np.cov(
        data_matrix,
        rowvar=False
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance_matrix
    )

    sorted_indices = np.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    explained_variance_ratio = (
        eigenvalues / np.sum(eigenvalues)
    )

    cumulative_variance = np.cumsum(
        explained_variance_ratio
    )

    return (
        covariance_matrix,
        eigenvalues,
        eigenvectors,
        explained_variance_ratio,
        cumulative_variance
    )


def transform_data(standardized_data, eigenvectors, n_components=2):
    components = eigenvectors[:, :n_components]

    transformed_data = np.dot(
        standardized_data,
        components
    )

    return transformed_data