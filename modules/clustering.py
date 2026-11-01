import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def perform_kmeans(data, n_clusters=3):
    """
    Perform K-Means clustering on the input data.
    """

    X = np.asarray(data)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    return model, labels


def calculate_silhouette_score(data, labels):
    """
    Calculate the Silhouette Score for the clustering result.
    """

    X = np.asarray(data)

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return None

    if len(unique_labels) >= len(X):
        return None

    score = silhouette_score(
        X,
        labels
    )

    return score