#include "kmeans/kmeans.h"
#include <math.h>
#include <stdio.h>

static int failures = 0;

#define CHECK(cond)                                                                                \
  do                                                                                               \
  {                                                                                                \
    if (!(cond))                                                                                   \
    {                                                                                              \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);                     \
      failures++;                                                                                  \
    }                                                                                              \
  } while (0)

#define N 40

static void test_two_blobs_2d(void)
{
  observation_2d_t obs[N];
  for (size_t i = 0; i < N; i++)
  {
    double offset = (i < N / 2) ? 0.0 : 100.0;
    obs[i].x = offset + (double) (i % 5) * 0.1;
    obs[i].y = offset + (double) (i % 3) * 0.1;
    obs[i].group = 0;
  }

  cluster_2d_t *c = kmeans_2d(obs, N, 2);
  CHECK(c != NULL);
  if (c == NULL)
  {
    return;
  }

  CHECK(c[0].count + c[1].count == N);
  CHECK(c[0].count == N / 2 && c[1].count == N / 2);
  CHECK(fabs(fabs(c[0].x - c[1].x) - 100.0) < 1.0);
  for (size_t i = 1; i < N / 2; i++)
  {
    CHECK(obs[i].group == obs[0].group);
    CHECK(obs[N / 2 + i].group == obs[N / 2].group);
  }
  CHECK(obs[0].group != obs[N / 2].group);
  free(c);
}

static void test_two_blobs_3d(void)
{
  observation_3d_t obs[N];
  for (size_t i = 0; i < N; i++)
  {
    double offset = (i < N / 2) ? 0.0 : 100.0;
    obs[i].x = offset + (double) (i % 5) * 0.1;
    obs[i].y = offset + (double) (i % 3) * 0.1;
    obs[i].z = offset + (double) (i % 7) * 0.1;
    obs[i].group = 0;
  }

  cluster_3d_t *c = kmeans_3d(obs, N, 2);
  CHECK(c != NULL);
  if (c == NULL)
  {
    return;
  }

  CHECK(c[0].count == N / 2 && c[1].count == N / 2);
  CHECK(fabs(fabs(c[0].z - c[1].z) - 100.0) < 1.0);
  CHECK(obs[0].group != obs[N / 2].group);
  free(c);
}

static void test_k_one_and_k_ge_size(void)
{
  observation_2d_t obs[4] = {{0, 0, 9}, {2, 0, 9}, {0, 2, 9}, {2, 2, 9}};

  cluster_2d_t *one = kmeans_2d(obs, 4, 1);
  CHECK(one != NULL);
  if (one != NULL)
  {
    CHECK(one[0].count == 4);
    CHECK(fabs(one[0].x - 1.0) < 1e-12 && fabs(one[0].y - 1.0) < 1e-12);
    CHECK(obs[3].group == 0);
    free(one);
  }

  cluster_2d_t *many = kmeans_2d(obs, 4, 6);
  CHECK(many != NULL);
  if (many != NULL)
  {
    for (size_t i = 0; i < 4; i++)
    {
      CHECK(many[i].count == 1 && obs[i].group == i);
    }
    CHECK(many[4].count == 0 && many[5].count == 0);
    free(many);
  }
}

static void test_identical_points_have_no_nan(void)
{
  observation_3d_t obs[10];
  for (size_t i = 0; i < 10; i++)
  {
    obs[i] = (observation_3d_t){1.0, 2.0, 3.0, 0};
  }

  cluster_3d_t *c = kmeans_3d(obs, 10, 3);
  CHECK(c != NULL);
  if (c == NULL)
  {
    return;
  }

  size_t total = 0;
  for (size_t i = 0; i < 3; i++)
  {
    CHECK(isfinite(c[i].x) && isfinite(c[i].y) && isfinite(c[i].z));
    total += c[i].count;
  }
  CHECK(total == 10);
  free(c);
}

static void test_invalid_input(void)
{
  observation_2d_t obs[1] = {{0, 0, 0}};
  CHECK(kmeans_2d(obs, 0, 2) == NULL);
  CHECK(kmeans_2d(NULL, 3, 2) == NULL);
  CHECK(kmeans_3d(NULL, 3, 1) == NULL);
}

int main(void)
{
  test_two_blobs_2d();
  test_two_blobs_3d();
  test_k_one_and_k_ge_size();
  test_identical_points_have_no_nan();
  test_invalid_input();

  if (failures != 0)
  {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  puts("all C tests passed");
  return 0;
}
