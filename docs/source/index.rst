.. ventu documentation master file, created by
   sphinx-quickstart on Wed May  6 19:31:44 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ventu's documentation!
=================================

|pypi| |versions| |Python Test| |Python document|

Serving the deep learning models easily.

Install
-------

.. code:: sh

   pip install ventu

Features
--------

-  Only need to implement Model(``preprocess``, ``postprocess``,
   ``inference`` or ``batch_inference``)
-  request & response data validation using
   `pydantic <https://pydantic-docs.helpmanual.io>`__
-  API document using
   `SpecTree <https://github.com/0b01001001/spectree>`__ (when run with
   ``run_http``)
-  backend service using `falcon <falcon.readthedocs.io/>`__ supports
   both JSON and `msgpack <https://msgpack.org/>`__
-  dynamic batching with
   `batching <https://github.com/kemingy/batching>`__ using Unix Domain
   Socket

   -  errors in one request won’t affect others in the same batch

-  support all the runtime
-  health check
-  inference warm-up

.. |pypi| image:: https://img.shields.io/pypi/v/ventu.svg
   :target: https://pypi.python.org/pypi/ventu
.. |versions| image:: https://img.shields.io/pypi/pyversions/ventu.svg
   :target: https://github.com/zenchars/ventu
.. |Python Test| image:: https://github.com/kemingy/ventu/workflows/Python%20package/badge.svg
.. |Python document| image:: https://github.com/kemingy/ventu/workflows/Python%20document/badge.svg
   :target: https://kemingy.github.io/ventu


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
