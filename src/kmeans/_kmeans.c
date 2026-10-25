#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Small self-contained PRNG (splitmix64) so results are reproducible for a given seed and the
// global rand() state is never touched.
typedef struct
{
  uint64_t state;
} rng_t;

static uint64_t rng_next(rng_t *rng)
{
  uint64_t z = (rng->state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Uniform double in [0, 1) with 53 bits of resolution.
static double rng_uniform(rng_t *rng)
{
  return (double) (rng_next(rng) >> 11) * (1.0 / 9007199254740992.0);
}

// Calculate squared Euclidean distance between two points
static double squared_distance(const double *a, const double *b, size_t dims)
{
  double dist = 0.0;
  for (size_t i = 0; i < dims; i++)
  {
    double diff = a[i] - b[i];
    dist += diff * diff;
  }
  return dist;
}

// Find the nearest centroid for a data point; optionally report the squared distance
static int find_nearest_centroid(const double *point, const double *centroids, size_t k,
                                 size_t dims, double *out_dist)
{
  int nearest = 0;
  double min_dist = DBL_MAX;

  for (size_t i = 0; i < k; i++)
  {
    double dist = squared_distance(point, centroids + i * dims, dims);
    if (dist < min_dist)
    {
      min_dist = dist;
      nearest = (int) i;
    }
  }
  if (out_dist != NULL)
  {
    *out_dist = min_dist;
  }
  return nearest;
}

// Initialize centroids using k-means++ algorithm. Returns 0 on success, -1 on allocation failure.
static int init_centroids_kmeanspp(const double *data, size_t n_samples, size_t dims, size_t k,
                                   double *centroids, rng_t *rng)
{
  double *distances = (double *) malloc(n_samples * sizeof(double));
  if (!distances)
  {
    return -1;
  }

  // Choose first centroid randomly
  size_t first_idx = (size_t) (rng_uniform(rng) * (double) n_samples);
  if (first_idx >= n_samples)
  {
    first_idx = n_samples - 1;
  }
  memcpy(centroids, data + first_idx * dims, dims * sizeof(double));

  // Distance of every point to its nearest chosen centroid
  for (size_t i = 0; i < n_samples; i++)
  {
    distances[i] = squared_distance(data + i * dims, centroids, dims);
  }

  // Choose remaining centroids
  for (size_t c = 1; c < k; c++)
  {
    double total_dist = 0.0;
    for (size_t i = 0; i < n_samples; i++)
    {
      total_dist += distances[i];
    }

    // Choose next centroid with probability proportional to squared distance. If all points
    // coincide with existing centroids (total == 0), fall back to a uniform choice.
    size_t chosen = n_samples - 1;
    if (total_dist > 0.0)
    {
      double threshold = rng_uniform(rng) * total_dist;
      double cumsum = 0.0;
      for (size_t i = 0; i < n_samples; i++)
      {
        cumsum += distances[i];
        if (cumsum > threshold && distances[i] > 0.0)
        {
          chosen = i;
          break;
        }
      }
    }
    else
    {
      chosen = (size_t) (rng_uniform(rng) * (double) n_samples);
      if (chosen >= n_samples)
      {
        chosen = n_samples - 1;
      }
    }

    memcpy(centroids + c * dims, data + chosen * dims, dims * sizeof(double));

    for (size_t i = 0; i < n_samples; i++)
    {
      double dist = squared_distance(data + i * dims, centroids + c * dims, dims);
      if (dist < distances[i])
      {
        distances[i] = dist;
      }
    }
  }

  free(distances);
  return 0;
}

// Main k-means algorithm. Returns 0 on success, -1 on allocation failure.
static int kmeans_core(const double *data, size_t n_samples, size_t dims, size_t k, int max_iter,
                       double tol, uint64_t seed, double *centroids, int *labels, double *inertia,
                       int *n_iter)
{
  double *new_centroids = (double *) calloc(k * dims, sizeof(double));
  size_t *counts = (size_t *) calloc(k, sizeof(size_t));
  rng_t rng = {seed};

  if (!new_centroids || !counts)
  {
    free(new_centroids);
    free(counts);
    return -1;
  }

  // Initialize centroids using k-means++
  if (init_centroids_kmeanspp(data, n_samples, dims, k, centroids, &rng) != 0)
  {
    free(new_centroids);
    free(counts);
    return -1;
  }

  int iterations = 0;
  for (int iter = 0; iter < max_iter; iter++)
  {
    iterations = iter + 1;

    // Reset accumulators
    memset(new_centroids, 0, k * dims * sizeof(double));
    memset(counts, 0, k * sizeof(size_t));

    // Assign points to nearest centroid
    for (size_t i = 0; i < n_samples; i++)
    {
      int nearest = find_nearest_centroid(data + i * dims, centroids, k, dims, NULL);
      labels[i] = nearest;
      counts[nearest]++;

      for (size_t d = 0; d < dims; d++)
      {
        new_centroids[(size_t) nearest * dims + d] += data[i * dims + d];
      }
    }

    // Update centroids
    double max_shift = 0.0;
    for (size_t c = 0; c < k; c++)
    {
      if (counts[c] > 0)
      {
        for (size_t d = 0; d < dims; d++)
        {
          new_centroids[c * dims + d] /= (double) counts[c];
        }
      }
      else
      {
        // Empty cluster: keep its previous centroid instead of collapsing to the origin
        memcpy(new_centroids + c * dims, centroids + c * dims, dims * sizeof(double));
      }

      double shift = squared_distance(centroids + c * dims, new_centroids + c * dims, dims);
      if (shift > max_shift)
      {
        max_shift = shift;
      }
    }

    memcpy(centroids, new_centroids, k * dims * sizeof(double));

    // Check convergence
    if (sqrt(max_shift) < tol)
    {
      break;
    }
  }

  // Final assignment against the final centroids so labels and inertia are consistent with them
  double total = 0.0;
  for (size_t i = 0; i < n_samples; i++)
  {
    double dist;
    labels[i] = find_nearest_centroid(data + i * dims, centroids, k, dims, &dist);
    total += dist;
  }
  *inertia = total;
  *n_iter = iterations;

  free(new_centroids);
  free(counts);
  return 0;
}

// Validate that an object is a well-behaved float64 2-D array
static int check_float64_matrix(PyArrayObject *array, const char *name)
{
  if (PyArray_NDIM(array) != 2)
  {
    PyErr_Format(PyExc_ValueError, "%s must be 2-dimensional", name);
    return -1;
  }
  if (PyArray_TYPE(array) != NPY_FLOAT64)
  {
    PyErr_Format(PyExc_TypeError, "%s must be float64", name);
    return -1;
  }
  if (!PyArray_ISCARRAY_RO(array))
  {
    PyErr_Format(PyExc_ValueError, "%s must be C-contiguous and aligned", name);
    return -1;
  }
  if (PyArray_DIM(array, 0) > INT_MAX || PyArray_DIM(array, 1) > INT_MAX)
  {
    PyErr_Format(PyExc_ValueError, "%s is too large", name);
    return -1;
  }
  return 0;
}

// Python wrapper for fit function
static PyObject *py_fit(PyObject *Py_UNUSED(self), PyObject *args)
{
  PyArrayObject *data_array;
  Py_ssize_t k;
  int max_iter;
  double tol;
  unsigned long long seed;

  if (!PyArg_ParseTuple(args, "O!nidK", &PyArray_Type, &data_array, &k, &max_iter, &tol, &seed))
  {
    return NULL;
  }

  if (check_float64_matrix(data_array, "Input data") != 0)
  {
    return NULL;
  }

  size_t n_samples = (size_t) PyArray_DIM(data_array, 0);
  size_t dims = (size_t) PyArray_DIM(data_array, 1);

  if (n_samples == 0 || dims == 0)
  {
    PyErr_SetString(PyExc_ValueError, "Input data must not be empty");
    return NULL;
  }

  if (k <= 0 || (size_t) k > n_samples)
  {
    PyErr_SetString(PyExc_ValueError, "k must be between 1 and n_samples");
    return NULL;
  }

  if (max_iter <= 0)
  {
    PyErr_SetString(PyExc_ValueError, "max_iter must be positive");
    return NULL;
  }

  if (!(tol >= 0.0))
  {
    PyErr_SetString(PyExc_ValueError, "tol must be non-negative");
    return NULL;
  }

  // Create output arrays
  npy_intp centroid_dims[2] = {(npy_intp) k, (npy_intp) dims};
  npy_intp label_dims[1] = {(npy_intp) n_samples};

  PyArrayObject *centroids = (PyArrayObject *) PyArray_SimpleNew(2, centroid_dims, NPY_FLOAT64);
  PyArrayObject *labels = (PyArrayObject *) PyArray_SimpleNew(1, label_dims, NPY_INT32);

  if (!centroids || !labels)
  {
    Py_XDECREF(centroids);
    Py_XDECREF(labels);
    return PyErr_NoMemory();
  }

  // Run k-means
  const double *data = (const double *) PyArray_DATA(data_array);
  double *centroids_data = (double *) PyArray_DATA(centroids);
  int *labels_data = (int *) PyArray_DATA(labels);
  double inertia = 0.0;
  int n_iter = 0;

  int result;
  Py_BEGIN_ALLOW_THREADS result = kmeans_core(data, n_samples, dims, (size_t) k, max_iter, tol,
                                              (uint64_t) seed, centroids_data, labels_data,
                                              &inertia, &n_iter);
  Py_END_ALLOW_THREADS

  if (result != 0)
  {
    Py_DECREF(centroids);
    Py_DECREF(labels);
    return PyErr_NoMemory();
  }

  // "N" steals the references, so the arrays are not leaked
  return Py_BuildValue("(NNdi)", centroids, labels, inertia, n_iter);
}

// Python wrapper for predict function
static PyObject *py_predict(PyObject *Py_UNUSED(self), PyObject *args)
{
  PyArrayObject *data_array, *centroids_array;

  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &data_array, &PyArray_Type, &centroids_array))
  {
    return NULL;
  }

  if (check_float64_matrix(data_array, "Input data") != 0 ||
      check_float64_matrix(centroids_array, "Centroids") != 0)
  {
    return NULL;
  }

  size_t n_samples = (size_t) PyArray_DIM(data_array, 0);
  size_t dims = (size_t) PyArray_DIM(data_array, 1);
  size_t k = (size_t) PyArray_DIM(centroids_array, 0);

  if ((size_t) PyArray_DIM(centroids_array, 1) != dims)
  {
    PyErr_SetString(PyExc_ValueError, "Data and centroids must have same dimensions");
    return NULL;
  }

  if (k == 0)
  {
    PyErr_SetString(PyExc_ValueError, "At least one centroid is required");
    return NULL;
  }

  npy_intp label_dims[1] = {(npy_intp) n_samples};
  PyArrayObject *labels = (PyArrayObject *) PyArray_SimpleNew(1, label_dims, NPY_INT32);

  if (!labels)
  {
    return PyErr_NoMemory();
  }

  const double *data = (const double *) PyArray_DATA(data_array);
  const double *centroids = (const double *) PyArray_DATA(centroids_array);
  int *labels_data = (int *) PyArray_DATA(labels);

  Py_BEGIN_ALLOW_THREADS for (size_t i = 0; i < n_samples; i++)
  {
    labels_data[i] = find_nearest_centroid(data + i * dims, centroids, k, dims, NULL);
  }
  Py_END_ALLOW_THREADS

  return (PyObject *) labels;
}

// Method definitions
static PyMethodDef kmeans_methods[] = {
  {"fit", py_fit, METH_VARARGS,
   "fit(data, k, max_iter, tol, seed) -> (centroids, labels, inertia, n_iter)\n\n"
   "Fit k-means to a C-contiguous float64 array of shape (n_samples, n_features)."},
  {"predict", py_predict, METH_VARARGS,
   "predict(data, centroids) -> labels\n\n"
   "Assign each row of `data` to its nearest centroid."},
  {NULL, NULL, 0, NULL}};

// Module definition
static struct PyModuleDef kmeans_module = {PyModuleDef_HEAD_INIT, "_kmeans",
                                           "K-means clustering C extension", -1, kmeans_methods};

// Module initialization
PyMODINIT_FUNC PyInit__kmeans(void)
{
  import_array();
  return PyModule_Create(&kmeans_module);
}
