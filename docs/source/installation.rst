Installation
============

Python 3.12 is the target runtime (see ``.python-version``).

From PyPI:

.. code-block:: sh

   pip install mamut

From source (editable install):

.. code-block:: sh

   pip install -e .

For development with uv:

.. code-block:: sh

   uv sync --all-groups

Quickstart
==========

.. code-block:: python

   from sklearn.datasets import load_iris
   from mamut.wrapper import Mamut

   X, y = load_iris(as_frame=True, return_X_y=True)
   mamut = Mamut(n_iterations=5, optimization_method="bayes")
   mamut.fit(X, y)
   preds = mamut.predict(X)
