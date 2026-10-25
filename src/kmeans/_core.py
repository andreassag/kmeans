"""Core k-means clustering functionality."""

import numpy as np

try:
    from kmeans import _kmeans  # pragma: no cover
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The _kmeans C extension could not be imported. "
        "Please ensure the package is properly installed with: pip install ."
    ) from e


def check_array(data, name="X"):
    """
    Validate input and convert it to a C-contiguous float64 matrix.

    Parameters
    ----------
    data : array-like of shape (n_samples,) or (n_samples, n_features)
        Input data. 1-D input is treated as a single feature.
    name : str, optional
        Name used in error messages.

    Returns
    -------
    ndarray of shape (n_samples, n_features)
        Validated data.

    Raises
    ------
    ValueError
        If the data is not 1-D or 2-D, is empty, or has non-finite values.
    """
    array = np.ascontiguousarray(data, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(
            f"{name} must be 1-D or 2-D, got {array.ndim} dimensions"
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _seed_from(random_state):
    """Derive a 63-bit integer seed from ``random_state``."""
    rng = np.random.default_rng(random_state)
    return int(rng.integers(0, 2**63 - 1))


def kmeans(data, k, max_iterations=100, tolerance=1e-4, random_state=None):
    """
    Perform k-means clustering on the given data.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The input data to cluster.
    k : int
        The number of clusters.
    max_iterations : int, optional
        Maximum number of iterations (default: 100).
    tolerance : float, optional
        Convergence tolerance (default: 1e-4).
    random_state : int, numpy.random.Generator or None, optional
        Seed for the k-means++ initialization. Pass an int for reproducible
        results (default: None, i.e. a fresh random seed).

    Returns
    -------
    centroids : ndarray of shape (k, n_features)
        The final cluster centroids.
    labels : ndarray of shape (n_samples,)
        Index of the cluster each sample belongs to.

    Raises
    ------
    ValueError
        If ``k`` is not between 1 and ``n_samples``, ``max_iterations`` is
        not positive, ``tolerance`` is negative, or the data is invalid.
    """
    centroids, labels, _, _ = _fit(
        data, k, max_iterations, tolerance, random_state
    )
    return centroids, labels


def _fit(data, k, max_iterations, tolerance, random_state):
    """Run the C extension and return ``(centroids, labels, inertia, n_iter)``."""
    data = check_array(data, "data")
    return _kmeans.fit(
        data,
        int(k),
        int(max_iterations),
        float(tolerance),
        _seed_from(random_state),
    )
