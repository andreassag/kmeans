import numpy as np
import pytest
from kmeans import KMeans, kmeans


@pytest.fixture
def sample_data():
    """Generate sample clustered data."""
    rng = np.random.default_rng(42)
    return np.vstack(
        [
            rng.standard_normal((50, 2)) + [0, 0],
            rng.standard_normal((50, 2)) + [10, 10],
        ]
    )


def test_kmeans_function(sample_data):
    """Test the functional kmeans API."""
    centroids, labels = kmeans(sample_data, k=2)

    assert centroids.shape == (2, 2)
    assert labels.shape == (100,)
    assert set(labels) == {0, 1}


def test_recovers_blob_centers(sample_data):
    centroids, labels = kmeans(sample_data, k=2, random_state=0)

    ordered = centroids[np.argsort(centroids[:, 0])]
    np.testing.assert_allclose(ordered, [[0, 0], [10, 10]], atol=0.5)
    assert len(set(labels[:50])) == 1
    assert len(set(labels[50:])) == 1


def test_kmeans_class(sample_data):
    """Test the KMeans class."""
    model = KMeans(n_clusters=2)
    model.fit(sample_data)

    assert model.cluster_centers_.shape == (2, 2)
    assert model.centroids_ is model.cluster_centers_
    assert model.labels_.shape == (100,)
    assert model.inertia_ > 0
    assert 1 <= model.n_iter_ <= model.max_iter


def test_kmeans_predict(sample_data):
    """Test prediction on new data."""
    model = KMeans(n_clusters=2)
    model.fit(sample_data)

    new_data = np.array([[0.0, 0.0], [10.0, 10.0]])
    predictions = model.predict(new_data)

    assert len(predictions) == 2
    assert predictions[0] != predictions[1]


def test_predict_matches_fit_labels(sample_data):
    model = KMeans(n_clusters=2, random_state=1).fit(sample_data)
    np.testing.assert_array_equal(model.predict(sample_data), model.labels_)


def test_kmeans_fit_predict(sample_data):
    """Test fit_predict method."""
    model = KMeans(n_clusters=2)
    labels = model.fit_predict(sample_data)

    assert labels.shape == (100,)
    np.testing.assert_array_equal(labels, model.labels_)


def test_random_state_is_reproducible(sample_data):
    a = KMeans(n_clusters=4, random_state=7).fit(sample_data)
    b = KMeans(n_clusters=4, random_state=7).fit(sample_data)

    np.testing.assert_array_equal(a.cluster_centers_, b.cluster_centers_)
    np.testing.assert_array_equal(a.labels_, b.labels_)


def test_get_set_params_and_repr():
    model = KMeans(n_clusters=3, random_state=1)
    assert model.get_params() == {
        "n_clusters": 3,
        "max_iter": 100,
        "tol": 1e-4,
        "random_state": 1,
    }
    assert model.set_params(n_clusters=5) is model
    assert model.n_clusters == 5
    assert "n_clusters=5" in repr(model)
    with pytest.raises(ValueError):
        model.set_params(bogus=1)


def test_predict_before_fit():
    with pytest.raises(ValueError, match="not fitted"):
        KMeans(n_clusters=2).predict(np.zeros((3, 2)))


def test_predict_feature_mismatch(sample_data):
    model = KMeans(n_clusters=2).fit(sample_data)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((3, 5)))


def test_noncontiguous_and_other_dtypes_match(sample_data):
    expected, _ = kmeans(sample_data, k=2, random_state=3)

    fortran = np.asfortranarray(sample_data)
    c_fortran, _ = kmeans(fortran, k=2, random_state=3)
    np.testing.assert_array_equal(expected, c_fortran)

    f32, _ = kmeans(sample_data.astype(np.float32), k=2, random_state=3)
    np.testing.assert_allclose(expected, f32, atol=1e-3)

    strided = np.repeat(sample_data, 2, axis=0)[::2]
    s, _ = kmeans(strided, k=2, random_state=3)
    np.testing.assert_array_equal(expected, s)


def test_integer_input():
    data = [[0, 0], [0, 1], [10, 10], [10, 11]]
    centroids, labels = kmeans(data, k=2, random_state=0)
    assert centroids.dtype == np.float64
    assert labels[0] == labels[1] != labels[2] == labels[3]


def test_one_dimensional_input():
    data = np.array([0.0, 0.1, 0.2, 9.9, 10.0, 10.1])
    centroids, labels = kmeans(data, k=2, random_state=0)
    assert centroids.shape == (2, 1)
    assert labels[0] == labels[2] != labels[3]


@pytest.mark.parametrize("dims", [1, 3, 10])
def test_arbitrary_dimensions(dims):
    rng = np.random.default_rng(0)
    data = np.vstack(
        [rng.standard_normal((30, dims)), rng.standard_normal((30, dims)) + 20]
    )
    centroids, labels = kmeans(data, k=2, random_state=0)
    assert centroids.shape == (2, dims)
    assert len(set(labels[:30])) == 1
    assert labels[0] != labels[30]


def test_k_equals_one_is_mean(sample_data):
    centroids, labels = kmeans(sample_data, k=1)
    np.testing.assert_allclose(centroids[0], sample_data.mean(axis=0))
    assert (labels == 0).all()


def test_k_equals_n_samples():
    data = np.arange(12, dtype=float).reshape(6, 2)
    centroids, labels = kmeans(data, k=6, random_state=0)
    assert sorted(map(tuple, centroids)) == sorted(map(tuple, data))
    assert sorted(labels) == list(range(6))


def test_duplicate_points_give_finite_centroids():
    data = np.ones((20, 2))
    model = KMeans(n_clusters=3, random_state=0).fit(data)
    assert np.isfinite(model.cluster_centers_).all()
    assert model.inertia_ == 0


def test_invalid_k():
    """Test that invalid k raises an error."""
    data = np.random.randn(10, 2)

    with pytest.raises(ValueError):
        kmeans(data, k=0)

    with pytest.raises(ValueError):
        kmeans(data, k=20)


@pytest.mark.parametrize(
    "kwargs",
    [{"max_iterations": 0}, {"max_iterations": -1}, {"tolerance": -1}],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        kmeans(np.random.randn(10, 2), k=2, **kwargs)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_input(bad):
    data = np.random.randn(10, 2)
    data[3, 1] = bad
    with pytest.raises(ValueError, match="NaN or infinity"):
        kmeans(data, k=2)


def test_invalid_shapes():
    with pytest.raises(ValueError):
        kmeans(np.zeros((0, 2)), k=1)
    with pytest.raises(ValueError):
        kmeans(np.zeros((4, 2, 2)), k=2)


def test_low_level_extension_validates_directly():
    from kmeans import _kmeans

    good = np.random.randn(10, 2)
    with pytest.raises(TypeError):
        _kmeans.fit(good.astype(np.float32), 2, 10, 1e-4, 0)
    with pytest.raises(ValueError):
        _kmeans.fit(np.asfortranarray(good), 2, 10, 1e-4, 0)
    with pytest.raises(TypeError):
        _kmeans.predict(good.astype(np.int64), good)
    with pytest.raises(ValueError):
        _kmeans.predict(good, np.zeros((0, 2)))


def test_repeated_fits_do_not_leak(sample_data):
    import tracemalloc

    kmeans(sample_data, k=2)  # warm up
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for _ in range(500):
        kmeans(sample_data, k=2)
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    assert after - before < 100_000
