==================
 Cobaya utilities
==================

A set of tools to deal with MCMC chains and a complement to `cobaya
<https://github.com/CobayaSampler/cobaya>`_ and `getdist <https://github.com/cmbant/getdist>`_.

.. image:: https://img.shields.io/pypi/v/cobaya-utilities.svg?style=flat
   :target: https://pypi.python.org/pypi/cobaya-utilities

.. image:: https://img.shields.io/github/actions/workflow/status/xgarrido/cobaya_utilities/testing.yml?branch=master
   :target: https://github.com/xgarrido/cobaya_utilities/actions
   :alt: GitHub Workflow Status

.. image:: https://readthedocs.org/projects/cobaya-utilities/badge/?version=latest
   :target: https://cobaya-utilities.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/xgarrido/cobaya_utilities/master

.. image:: https://codecov.io/gh/xgarrido/cobaya_utilities/branch/master/graph/badge.svg?token=qrrVcbNCs5
   :target: https://codecov.io/gh/xgarrido/cobaya_utilities

.. image:: https://results.pre-commit.ci/badge/github/xgarrido/cobaya_utilities/master.svg
   :target: https://results.pre-commit.ci/latest/github/xgarrido/cobaya_utilities/master
   :alt: pre-commit.ci status

Installing the code
-------------------

The easiest way to install the module is

.. code:: shell

    pip install [--user] cobaya-utilities

If you plan to made some modifications, to improve or to correct some bugs, then you need to clone
the following repository

.. code:: shell

    git clone https://github.com/xgarrido/cobaya_utilities.git /where/to/clone

Then you can install the code and its dependencies *via*

.. code:: shell

    pip install -e [--user] /where/to/clone


Running/testing the code
------------------------

You can test the ``cobaya_utilities`` module (assuming you have ``pytest`` installed) by doing

.. code:: shell

    pytest


.. end_before_documentation

Documentation
-------------

Read the docs at `cobaya-utilities.readthedocs.io <http://cobaya-utilities.readthedocs.io>`_.
