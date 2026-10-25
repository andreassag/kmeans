> [!NOTE]
> **Archived.** This project is feature-complete and no longer maintained.
> The C library and the Python bindings work as documented; no further
> development, issue triage or dependency updates are planned.

# KMeans

A fast k-means clustering algorithm implemented in C with Python bindings.

## Installation

### Using UV (recommended)

```bash
uv pip install .
```

### Using pip

```bash
pip install .
```

### Development installation

```bash
uv pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from kmeans import KMeans, kmeans

# Generate sample data
np.random.seed(42)
data = np.vstack([
    np.random.randn(100, 2) + [0, 0],
    np.random.randn(100, 2) + [5, 5],
    np.random.randn(100, 2) + [10, 0],
])

# Using the functional API
centroids, labels = kmeans(data, k=3)

# Using the class-based API
model = KMeans(n_clusters=3, random_state=0)
model.fit(data)
predictions = model.predict(data)
print(model.cluster_centers_, model.inertia_, model.n_iter_)
```

## Features

* Fast k-means clustering with C implementation
* K-means++ initialization
* NumPy integration
* Scikit-learn style API (`fit`, `predict`, `fit_predict`, `get_params`, `set_params`)
* Reproducible results via `random_state`
* Support for arbitrary dimensions

## Standalone C library

`src/2d.c` and `src/3d.c` build into a shared library (`libkmeans`) that can
be used without Python. See `include/kmeans/kmeans.h` for the API.

```bash
cmake -S . -B build -DKMEANS_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Development

```bash
uv sync --group dev
make check   # ruff, pytest and the C library tests
make docs    # Sphinx documentation
```

## License

MIT
