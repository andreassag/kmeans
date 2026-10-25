Installation
============

Requirements
------------

* Python >= 3.10
* NumPy
* CMake >= 3.18
* C compiler (gcc, clang, or MSVC)

Using pip
---------

.. code-block:: bash

   pip install .

Using UV
--------

UV is a fast Python package installer and resolver:

.. code-block:: bash

   uv pip install .

Development Installation
------------------------

For development, install in editable mode with development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

Or with UV:

.. code-block:: bash

   uv pip install -e ".[dev]"

Building from Source
--------------------

1. Clone the repository:

   .. code-block:: bash

      git clone <repository-url>
      cd kmeans

2. Build and install:

   .. code-block:: bash

      pip install .

The build process will:

1. Configure CMake
2. Compile the C extension
3. Build the Python wheel
4. Install the package

Troubleshooting
---------------

CMake not found
~~~~~~~~~~~~~~~

Install CMake:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install cmake

   # macOS
   brew install cmake

   # Windows
   # Download from https://cmake.org/download/

Compiler not found
~~~~~~~~~~~~~~~~~~

Install a C compiler:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install build-essential

   # macOS (installs clang)
   xcode-select --install

   # Windows
   # Install Visual Studio with C++ support