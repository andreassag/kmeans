"""Scikit-learn style KMeans estimator."""

import numpy as np

from kmeans import _kmeans
from kmeans._core import _fit, check_array


class KMeans:
    """
    K-Means clustering.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form.
    max_iter : int, optional
        Maximum number of iterations (default: 100).
    tol : float, optional
        Convergence tolerance (default: 1e-4).
    random_state : int, numpy.random.Generator or None, optional
        Seed for the k-means++ initialization. Pass an int for reproducible
        results (default: None).

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. Also available as ``centroids_``.
    labels_ : ndarray of shape (n_samples,)
        Cluster index of each training sample.
    inertia_ : float
        Sum of squared distances of samples to their closest center.
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def __repr__(self):
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"

    @property
    def centroids_(self):
        """Alias for ``cluster_centers_``."""
        return self.cluster_centers_

    def get_params(self, deep=True):
        """Return the constructor parameters as a dict.

        ``deep`` is accepted for scikit-learn compatibility and ignored.
        """
        return {
            "n_clusters": self.n_clusters,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """
        Set constructor parameters.

        Returns
        -------
        self : KMeans
            The estimator itself.

        Raises
        ------
        ValueError
            If a parameter name is not valid.
        """
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Invalid parameter: {key!r}")
            setattr(self, key, value)
        return self

    def fit(self, X, y=None):
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        self : KMeans
            Fitted estimator.
        """
        (
            self.cluster_centers_,
            self.labels_,
            self.inertia_,
            self.n_iter_,
        ) = _fit(
            X, self.n_clusters, self.max_iter, self.tol, self.random_state
        )
        return self

    def predict(self, X):
        """
        Predict the closest cluster for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.

        Raises
        ------
        ValueError
            If the model is not fitted or the number of features differs
            from the data used in ``fit``.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = check_array(X)
        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but KMeans was fitted with "
                f"{self.cluster_centers_.shape[1]}"
            )

        return _kmeans.predict(X, np.ascontiguousarray(self.cluster_centers_))

    def fit_predict(self, X, y=None):
        """
        Compute clustering and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X).labels_
