Installation
============

.. meta::
   :description: Install MAMUT from PyPI or from source with uv for automated tabular classification workflows.
   :keywords: MAMUT installation, Python 3.12, uv, pip

MAMUT targets Python 3.12. The repository pins the local development runtime in
``.python-version`` and resolves dependencies with ``uv``.

Install From PyPI
-----------------

.. code-block:: sh

   pip install mamut

Install From Source
-------------------

Clone the repository and install the package in editable mode:

.. code-block:: sh

   git clone https://github.com/przybytniowskaj/Mamut.git
   cd Mamut
   pip install -e .

Development Environment
-----------------------

For repository development, install all dependency groups from ``uv.lock``:

.. code-block:: sh

   uv sync --all-groups

Common validation commands:

.. code-block:: sh

   uv run deptry .
   scripts/audit_dependencies.sh
   uv run pytest
   uv run pre-commit run --all-files
   uv run make -C docs html

Next Steps
----------

After installation, continue with the :doc:`quickstart` to fit a first model
and inspect the generated outputs.
