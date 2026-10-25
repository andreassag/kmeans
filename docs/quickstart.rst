Quick Start Guide
=================

Basic Usage
-----------

Functional API
~~~~~~~~~~~~~~

The simplest way to use kmeans:

.. code-block:: python

   import numpy as np
   from kmeans import kmeans

   # Generate sample data
   data = np.random.randn(1000, 2)

   # Perform clustering
   centroids, labels = kmeans(data, k=5)

   print(f"Centroids:\n{centroids}")
   print(f"Labels: {labels}")

Object-Oriented API
~~~~~~~~~~~~~~~~~~~

For a scikit-learn compatible interface:

.. code-block:: python

   from kmeans import KMeans

   # Create and fit the model
   model = KMeans(n_clusters=5, max_iter=100)
   model.fit(data)

   # Access results
   print(f"Centroids:\n{model.centroids_}")
   print(f"Labels: {model.labels_}")

   # Predict on new data
   new_data = np.random.randn(100, 2)
   predictions = model.predict(new_data)

Complete Example
----------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from kmeans import KMeans

   # Generate three clusters
   np.random.seed(42)
   cluster1 = np.random.randn(100, 2) + [0, 0]
   cluster2 = np.random.randn(100, 2) + [5, 5]
   cluster3 = np.random.randn(100, 2) + [10, 0]
   data = np.vstack([cluster1, cluster2, cluster3])

   # Fit k-means
   kmeans_model = KMeans(n_clusters=3, max_iter=100, tol=1e-4)
   kmeans_model.fit(data)

   # Plot results
   plt.scatter(data[:, 0], data[:, 1], c=kmeans_model.labels_, cmap='viridis')
   plt.scatter(kmeans_model.centroids_[:, 0], 
               kmeans_model.centroids_[:, 1],
               c='red', marker='x', s=200, linewidths=3)
   plt.title('K-Means Clustering')
   plt.xlabel('Feature 1')
   plt.ylabel('Feature 2')
   plt.show()

Working with Different Dimensions
----------------------------------

1D Data
~~~~~~~

.. code-block:: python

   # 1D clustering
   data_1d = np.random.randn(500)
   centroids, labels = kmeans(data_1d, k=3)
   print(f"Shape: {centroids.shape}")  # (3, 1)

High-Dimensional Data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 10D clustering
   data_10d = np.random.randn(1000, 10)
   model = KMeans(n_clusters=5)
   model.fit(data_10d)
   print(f"Centroids shape: {model.centroids_.shape}")  # (5, 10)

scikit-learn conventions
------------------------

``KMeans`` follows scikit-learn conventions (``fit``/``predict``/``fit_predict``,
trailing-underscore attributes, ``get_params``/``set_params``, ``random_state``),
but it is not a scikit-learn ``BaseEstimator`` and is not guaranteed to work
inside scikit-learn ``Pipeline`` objects.

Reproducible results
--------------------

.. code-block:: python

   from kmeans import KMeans

   model = KMeans(n_clusters=3, random_state=42).fit(data)
   print(model.cluster_centers_, model.inertia_, model.n_iter_)
