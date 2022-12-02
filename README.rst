==================
 Cobaya utilities
==================

A set of tools to deal with MCMC chains and a complement to `cobaya
<https://github.com/CobayaSampler/cobaya>`_ and `getdist <https://github.com/cmbant/getdist>`_.

.. image:: https://img.shields.io/github/workflow/status/xgarrido/cobaya_utilities/Unit%20test/feature-github-actions
   :target: https://github.com/xgarrido/cobaya_utilities/actions

..
   .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/simonsobs/LAT_MFLike/master?filepath=notebooks%2Fmflike_tutorial.ipynb


   .. image:: https://codecov.io/gh/simonsobs/LAT_MFLike/branch/master/graph/badge.svg?token=qrrVcbNCs5
      :target: https://codecov.io/gh/simonsobs/LAT_MFLike


Installing the code
-------------------

The easiest way to install the module is

.. code:: shell

    $ pip install [--user] cobaya-utilities

If you plan to made some modifications, to improve or to correct some bugs, then you need to clone
the following respoitory

.. code:: shell

    $ git clone https://github.com/xgarrido/cobaya_utilities.git /where/to/clone

Then you can install the code and its dependencies *via*

.. code:: shell

    $ pip install -e [--user] /where/to/clone

..
   Running/testing the code
   ------------------------

   You can test the ``cobaya_utilities`` by doing