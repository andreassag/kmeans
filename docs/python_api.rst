Python API Reference
====================

.. automodule:: kmeans
   :members:
   :undoc-members:
   :show-inheritance:

Functional API
--------------

.. autofunction:: kmeans.kmeans

Object-Oriented API
-------------------

KMeans Class
~~~~~~~~~~~~

.. autoclass:: kmeans.KMeans
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. automethod:: __init__
   .. automethod:: fit
   .. automethod:: predict
   .. automethod:: fit_predict

Attributes
^^^^^^^^^^

After calling :meth:`fit`, the following attributes are available:

.. py:attribute:: cluster_centers_

   :type: numpy.ndarray of shape (n_clusters, n_features)

   Coordinates of cluster centers. Also available as ``centroids_``.

.. py:attribute:: labels_

   :type: numpy.ndarray of shape (n_samples,)

   Labels of each point indicating cluster assignment.

.. py:attribute:: inertia_

   :type: float

   Sum of squared distances of samples to their closest cluster center.

.. py:attribute:: n_iter_

   :type: int

   Number of iterations run.

C Extension Module
------------------

.. note::
   The ``_kmeans`` module is a low-level C extension. Most users should use
   the high-level Python API instead.

.. py:module:: kmeans._kmeans

.. py:function:: fit(data, k, max_iter, tol, seed)

   Low-level k-means fitting function.

   :param numpy.ndarray data: Input data array (n_samples, n_features)
   :param int k: Number of clusters
   :param int max_iter: Maximum iterations (must be positive)
   :param float tol: Convergence tolerance (must be non-negative)
   :param int seed: Seed for the k-means++ initialization
   :return: Tuple of (centroids, labels, inertia, n_iter)
   :rtype: tuple[numpy.ndarray, numpy.ndarray, float, int]

   ``data`` must be a C-contiguous ``float64`` array; ``TypeError`` or
   ``ValueError`` is raised otherwise.

.. py:function:: predict(data, centroids)

   Predict cluster labels for data points.

   :param numpy.ndarray data: Input data array (n_samples, n_features)
   :param numpy.ndarray centroids: Cluster centroids (k, n_features)
   :return: Cluster labels
   :rtype: numpy.ndarray

Examples
--------

See :doc:`examples` for more usage examples.