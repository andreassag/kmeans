kmeans: Fast K-Means Clustering
================================

A high-performance k-means clustering library with a C implementation and Python bindings.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   python_api
   c_api
   examples
   contributing

Overview
--------

The ``kmeans`` package provides:

* Fast k-means clustering algorithm implemented in C
* Python bindings with NumPy integration
* Scikit-learn style API
* K-means++ initialization for better convergence
* Support for arbitrary dimensions

Features
--------

* **Performance**: Core algorithm written in C for speed
* **Easy to Use**: Simple Python API with sensible defaults
* **Flexible**: Both functional and object-oriented interfaces
* **Compatible**: Works with NumPy arrays and follows scikit-learn conventions (``fit``/``predict``, ``get_params``, ``random_state``)
* **Well-tested**: Comprehensive test suite

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`