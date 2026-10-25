Contributing
============

We welcome contributions! This guide will help you get started.

Development Setup
-----------------

1. Clone the repository:

   .. code-block:: bash

      git clone <repository-url>
      cd kmeans

2. Install in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

3. Run tests:

   .. code-block:: bash

      pytest tests/

4. Run the standalone C library tests:

   .. code-block:: bash

      make test-c

Code Style
----------

Python
~~~~~~

Follow PEP 8 guidelines. Use type hints where appropriate.

C Code
~~~~~~

The C code uses a custom clang-format style. Format code with:

.. code-block:: bash

   clang-format -i src/kmeans/_kmeans.c src/2d.c src/3d.c include/kmeans/kmeans.h tests/c/test_kmeans_lib.c

Key style points:

* 2-space indentation
* Braces on new lines for functions
* 100 character line limit
* Pointer alignment to the right: ``double *ptr``

Testing
-------

Write tests for new features:

.. code-block:: python

   # tests/test_new_feature.py
   import pytest
   from kmeans import KMeans

   def test_new_feature():
       model = KMeans(n_clusters=3)
       # ... test code ...
       assert model.centroids_ is not None

Run tests with coverage:

.. code-block:: bash

   pytest --cov=kmeans tests/

Documentation
-------------

Update documentation for new features:

1. Add docstrings to Python code (NumPy style)
2. Add usage examples to ``docs/examples.rst``
3. Update API reference if needed

Build documentation locally:

.. code-block:: bash

   cd docs
   make html
   # Open _build/html/index.html

Pull Request Process
--------------------

1. Create a feature branch
2. Make your changes
3. Add tests
4. Update documentation
5. Ensure all tests pass
6. Submit pull request

Commit Messages
~~~~~~~~~~~~~~~

Use clear, descriptive commit messages:

.. code-block:: text

   Add support for weighted k-means

   - Implement sample weights in C core
   - Add weight parameter to Python API
   - Add tests for weighted clustering

Reporting Issues
----------------

When reporting bugs, include:

* Python version
* NumPy version
* Operating system
* Minimal code to reproduce
* Error messages and stack traces

Feature Requests
----------------

For feature requests, describe:

* Use case and motivation
* Proposed API
* Example usage
* Any related work or references