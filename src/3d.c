#include "kmeans/kmeans.h"
#include <float.h>
#include <stdint.h>
#include <string.h>

static size_t calculate_nearest(const observation_3d_t *observation, const cluster_3d_t clusters[],
                                size_t k)
{
  double min_distance = DBL_MAX;
  size_t nearest_index = 0;

  for (size_t index = 0; index < k; index++)
  {
    double dx = clusters[index].x - observation->x;
    double dy = clusters[index].y - observation->y;
    double dz = clusters[index].z - observation->z;
    double distance = dx * dx + dy * dy + dz * dz;

    if (distance < min_distance)
    {
      min_distance = distance;
      nearest_index = index;
    }
  }
  return nearest_index;
}

static void calculate_centroid(observation_3d_t observations[], size_t size, cluster_3d_t *centroid)
{
  centroid->x = 0;
  centroid->y = 0;
  centroid->z = 0;
  centroid->count = size;

  for (size_t index = 0; index < size; index++)
  {
    centroid->x += observations[index].x;
    centroid->y += observations[index].y;
    centroid->z += observations[index].z;
    observations[index].group = 0;
  }

  centroid->x /= (double) size;
  centroid->y /= (double) size;
  centroid->z /= (double) size;
}

cluster_3d_t *kmeans_3d(observation_3d_t observations[], size_t size, size_t k)
{
  if (observations == NULL || size == 0)
  {
    return NULL;
  }

  if (k <= 1)
  {
    cluster_3d_t *cluster = calloc(1, sizeof(cluster_3d_t));
    if (cluster != NULL)
    {
      calculate_centroid(observations, size, cluster);
    }
    return cluster;
  }

  if (k > SIZE_MAX / sizeof(cluster_3d_t))
  {
    return NULL;
  }
  cluster_3d_t *clusters = calloc(k, sizeof(cluster_3d_t));
  if (clusters == NULL)
  {
    return NULL;
  }

  if (k >= size)
  {
    for (size_t index = 0; index < size; index++)
    {
      clusters[index].x = observations[index].x;
      clusters[index].y = observations[index].y;
      clusters[index].z = observations[index].z;
      clusters[index].count = 1;
      observations[index].group = index;
    }
    return clusters;
  }

  for (size_t index = 0; index < size; index++)
  {
    observations[index].group = (size_t) rand() % k;
  }

  for (int iteration = 0; iteration < KMEANS_MAX_ITERATIONS; iteration++)
  {
    for (size_t index = 0; index < k; index++)
    {
      clusters[index].x = 0;
      clusters[index].y = 0;
      clusters[index].z = 0;
      clusters[index].count = 0;
    }

    for (size_t index = 0; index < size; index++)
    {
      cluster_3d_t *cluster = &clusters[observations[index].group];
      cluster->x += observations[index].x;
      cluster->y += observations[index].y;
      cluster->z += observations[index].z;
      cluster->count++;
    }

    for (size_t index = 0; index < k; index++)
    {
      if (clusters[index].count == 0)
      {
        // Re-seed an empty cluster from a random observation.
        const observation_3d_t *seed = &observations[(size_t) rand() % size];
        clusters[index].x = seed->x;
        clusters[index].y = seed->y;
        clusters[index].z = seed->z;
      }
      else
      {
        clusters[index].x /= (double) clusters[index].count;
        clusters[index].y /= (double) clusters[index].count;
        clusters[index].z /= (double) clusters[index].count;
      }
    }

    size_t changed = 0;
    for (size_t index = 0; index < size; index++)
    {
      size_t nearest = calculate_nearest(&observations[index], clusters, k);

      if (nearest != observations[index].group)
      {
        changed++;
        observations[index].group = nearest;
      }
    }

    if (changed == 0)
    {
      break;
    }
  }

  // Report counts consistent with the final assignments.
  for (size_t index = 0; index < k; index++)
  {
    clusters[index].count = 0;
  }
  for (size_t index = 0; index < size; index++)
  {
    clusters[observations[index].group].count++;
  }

  return clusters;
}
