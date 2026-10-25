#ifndef KMEANS_KMEANS_H
#define KMEANS_KMEANS_H

#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Upper bound on Lloyd iterations performed by kmeans_2d()/kmeans_3d(). */
#define KMEANS_MAX_ITERATIONS 300

typedef struct observation_2d
{
  double x, y;
  size_t group;
} observation_t, observation_2d_t;

typedef struct observation_3d
{
  double x, y, z;
  size_t group;
} observation_3d_t;

typedef struct cluster_2d
{
  double x, y;
  size_t count;
} cluster_t, cluster_2d_t;

typedef struct cluster_3d
{
  double x, y, z;
  size_t count;
} cluster_3d_t;

/**
 * Cluster 2D observations with Lloyd's algorithm.
 *
 * On return, each observation's `group` holds the index of its cluster.
 *
 * - k <= 1: a single cluster at the mean of all observations.
 * - k >= size: one cluster per observation (clusters beyond `size`, if any,
 *   have a count of zero).
 * - Otherwise: random initial assignment, then iterate until assignments no
 *   longer change or KMEANS_MAX_ITERATIONS is reached. A cluster that becomes
 *   empty is re-seeded from a random observation.
 *
 * @param observations Observations to cluster (their `group` is overwritten).
 * @param size         Number of observations.
 * @param k            Requested number of clusters.
 * @return A heap-allocated array of max(k, 1) clusters that the caller must
 *         free(), or NULL if `observations` is NULL, `size` is 0, or
 *         allocation fails.
 */
cluster_2d_t *kmeans_2d(observation_2d_t observations[], size_t size, size_t k);

/** Same as kmeans_2d(), for 3D observations. */
cluster_3d_t *kmeans_3d(observation_3d_t observations[], size_t size, size_t k);

/** Alias for kmeans_2d(). */
cluster_t *kmeans(observation_t observations[], size_t size, size_t k);

#ifdef __cplusplus
}
#endif

#endif /* KMEANS_KMEANS_H */
