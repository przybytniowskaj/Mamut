Preprocessing API
=================

.. meta::
   :description: API reference for MAMUT preprocessing, including the Preprocessor class and preprocessing handlers.
   :keywords: MAMUT preprocessing, Preprocessor, tabular data preprocessing

The preprocessing API transforms tabular data before model selection. Most
users configure preprocessing through :class:`mamut.wrapper.Mamut`, which
forwards preprocessing keyword arguments to
:class:`mamut.preprocessing.preprocessing.Preprocessor`.

Preprocessor
------------

.. autoclass:: mamut.preprocessing.preprocessing.Preprocessor
   :show-inheritance:

Common Methods
~~~~~~~~~~~~~~

.. automethod:: mamut.preprocessing.preprocessing.Preprocessor.fit_transform
   :no-index:

.. automethod:: mamut.preprocessing.preprocessing.Preprocessor.transform
   :no-index:

.. automethod:: mamut.preprocessing.preprocessing.Preprocessor.report
   :no-index:

Handlers
--------

The handler functions below are lower-level building blocks used by
``Preprocessor``.

.. automodule:: mamut.preprocessing.handlers
   :members:
   :show-inheritance:

Settings
--------

.. automodule:: mamut.preprocessing.settings
   :members:
   :show-inheritance:
