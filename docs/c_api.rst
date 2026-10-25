C API Reference
===============

This page documents the C implementation details for developers who want to
understand or extend the underlying algorithm.

.. note::
   The C extension is implemented in ``src/kmeans/_kmeans.c``. 
   This documentation describes the algorithm and function signatures.

Standalone C Library
--------------------

``src/2d.c`` and ``src/3d.c`` build into ``libkmeans`` for use without Python
(``include/kmeans/kmeans.h``):

.. code-block:: c

   cluster_2d_t *kmeans_2d(observation_2d_t observations[], size_t size, size_t k);
   cluster_3d_t *kmeans_3d(observation_3d_t observations[], size_t size, size_t k);

Each observation's ``group`` is set to its cluster index. The returned array
of ``max(k, 1)`` clusters is heap-allocated and must be released with
``free()``. ``NULL`` is returned for empty input or on allocation failure.
``k <= 1`` yields one cluster at the mean, and ``k >= size`` yields one cluster
per observation. Otherwise Lloyd's algorithm runs from a random initial
assignment (using ``rand()``; seed it with ``srand()``) for at most
``KMEANS_MAX_ITERATIONS`` iterations, re-seeding empty clusters from a random
observation. Tests live in ``tests/c`` and run with ``make test-c``.

Python Extension Algorithm Overview
-----------------------------------

The k-means implementation uses the following approach:

1. **Initialization**: K-means++ algorithm for better initial centroids
2. **Assignment**: Assign each point to nearest centroid
3. **Update**: Recalculate centroids as mean of assigned points
4. **Convergence**: Repeat until centroids stop moving significantly

Core Functions
--------------

Distance Calculation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

   static double squared_distance(const double *a, const double *b, size_t dims)

Calculate squared Euclidean distance between two points.

:param a: First point coordinates
:param b: Second point coordinates  
:param dims: Number of dimensions
:return: Squared distance

Find Nearest Centroid
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

   static int find_nearest_centroid(const double *point, 
                                     const double *centroids,
                                     size_t k, size_t dims,
                                     double *out_dist)

Find the index of the nearest centroid for a given point.

:param point: Data point coordinates
:param centroids: Array of all centroid coordinates
:param k: Number of clusters
:param dims: Number of dimensions
:param out_dist: Optional output for the squared distance to that centroid
:return: Index of nearest centroid (0 to k-1)

K-means++ Initialization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

   static int init_centroids_kmeanspp(const double *data, 
                                       size_t n_samples, size_t dims,
                                       size_t k, double *centroids,
                                       rng_t *rng)

Initialize centroids using the k-means++ algorithm.

The k-means++ initialization chooses initial cluster centers that are
far apart, leading to better and faster convergence.

:param data: Input data array
:param n_samples: Number of data points
:param dims: Number of dimensions
:param k: Number of clusters
:param centroids: Output array for centroid coordinates

**Algorithm:**

1. Choose first centroid uniformly at random
2. For each remaining centroid:
   
   a. Calculate distance to nearest existing centroid for each point
   b. Choose new centroid with probability proportional to squared distance

Main K-means Algorithm
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

   static int kmeans_core(const double *data, size_t n_samples, size_t dims, 
                          size_t k, int max_iter, double tol, uint64_t seed,
                          double *centroids, int *labels,
                          double *inertia, int *n_iter)

Main k-means clustering algorithm.

:param data: Input data array (row-major, n_samples × dims)
:param n_samples: Number of data points
:param dims: Number of dimensions
:param k: Number of clusters
:param max_iter: Maximum number of iterations
:param tol: Convergence tolerance
:param centroids: Output array for final centroids
:param seed: Seed for the extension's internal PRNG (splitmix64)
:param labels: Output array for cluster assignments
:param inertia: Output for the sum of squared distances to the centroids
:param n_iter: Output for the number of iterations run
:return: 0 on success, -1 on memory allocation failure

**Convergence:**

The algorithm stops when the maximum centroid shift is less than ``tol``,
or when ``max_iter`` iterations have been reached.

Python Binding Functions
-------------------------

.. code-block:: c

   static PyObject *py_fit(PyObject *self, PyObject *args)

Python wrapper for the fit operation. Accepts NumPy arrays and returns 
tuple of (centroids, labels).

.. code-block:: c

   static PyObject *py_predict(PyObject *self, PyObject *args)

Python wrapper for the predict operation. Assigns new data points to 
existing centroids.

Memory Management
-----------------

The C implementation uses dynamic memory allocation for:

* Temporary centroid calculations
* Cluster point counts
* Distance calculations in k-means++

All allocations are freed before function return. Error checking ensures
proper cleanup on allocation failures.

Performance Considerations
--------------------------

Optimization Techniques
~~~~~~~~~~~~~~~~~~~~~~~

1. **Memory Layout**: Row-major array layout for cache efficiency
2. **Squared Distance**: Avoid expensive sqrt() in distance calculations
3. **Early Termination**: Stop when convergence tolerance is reached
4. **K-means++**: Better initialization reduces iterations needed

Complexity
~~~~~~~~~~

* **Time**: O(n × k × d × i) where:
  
  * n = number of samples
  * k = number of clusters
  * d = number of dimensions
  * i = number of iterations

* **Space**: O(n + k × d)

Building the C Extension
-------------------------

The C extension is built using CMake:

.. code-block:: cmake

   # From CMakeLists.txt
   python_add_library(_kmeans MODULE
       src/kmeans/_kmeans.c
       WITH_SOABI
   )

   target_include_directories(_kmeans PRIVATE
       ${Python_NumPy_INCLUDE_DIRS}
       ${CMAKE_SOURCE_DIR}/include
   )

   target_link_libraries(_kmeans PRIVATE Python::NumPy)

Source Code Location
--------------------

The complete C implementation can be found in:

* ``src/kmeans/_kmeans.c`` - Main implementation file
* ``CMakeLists.txt`` - Build configuration

See the source files directly for the most up-to-date implementation details.