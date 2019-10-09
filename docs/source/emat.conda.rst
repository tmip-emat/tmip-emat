
=========================
Installing using Anaconda
=========================

Quick Start
-----------

To get started with TMIP-EMAT, you'll need to follow a few simple steps.

1. Install `Anaconda Python 3.7 <https://www.anaconda.com/download>`_.
2. Open the 'Anaconda Prompt' that was installed and run the following
   commands:

.. code-block:: console

    conda env create TMIP/EMAT
    conda activate EMAT
    jupyter-notebook

More detailed instructions appear below.

Installing Python
-----------------

To use TMIP-EMAT, you'll need to have Python 3.7, plus a handful
of other useful statistical packages.  The easiest way to get the basics
is to download and install the `Anaconda <https://www.anaconda.com/download>`_
version of Python 3.7. This comes with everything you'll need to get started,
and the Anaconda folks have helpfully curated a selection of useful tools for you,
so you don't have the sort through the huge selection of tools, both good and bad,
that are available for Python.

.. note::

    Python has two versions (2 and 3) that are available and currently maintained.
    TMIP-EMAT is compatible *only* with version 3.

You should usually install Anaconda for the local user,
which does not require administrator permissions.
You can also install Anaconda system wide, which does require
administrator permissions -- but even if you have those permissions,
you may find that installing only for one user prevents problems
arising over multiple users editing common packages.

If you already have Python installed, either by itself or
as a companion to any one of a variety of common transportation planning
tools (e.g., ArcGIS), you can still install and use Anaconda.
You do not need to uninstall, move, or change any existing
Python installation.  Just use the standard Anaconda installer
and let the installer add the conda installation of Python
to your PATH environment variable. There is no need to set the
PYTHONPATH environment variable.

Once Anaconda is installed, it can be accessed from the
Anaconda Prompt (on Windows) or the Terminal (linux and macOS).


Managing Environments
---------------------

When you use conda to install Python, by default a `base` environment is
created and packages are installed in that environment.  However, in general you should
almost never undertake project work in the `base` environment, especially if your
project involves installing any custom Python packages.  Instead,
you should create a new environment for each project, and install the
necessary packages and dependencies in that environment.  This will help
prevent software conflicts, and ensure that tools installed for one project
will not break another project.

The instructions below provide only the most basic steps to
set up and use an environment.  Much more extensive documentation
on :doc:`managing environments <conda:user-guide/tasks/manage-environments>`
is available in the conda documentation itself.


Creating a New Environment for TMIP-EMAT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    If you installed the "Miniconda" version of the anaconda package, or
    if your main conda installation is a bit out of date, you
    may need to install or update the *conda* and *anaconda-client* packages
    before the remote environment installation below will work:

    .. code-block:: console

        conda install -n base -c defaults conda anaconda-client

If you'd like one command to just install TMIP-EMAT and
the suite of related tools relevant for exploratory modeling and analysis
analysis, you can create a new environment for EMAT with one line.

.. code-block:: console

    conda env create TMIP/EMAT

If you've already installed the *EMAT* environment and want to update it to the latest
version, you can use:

.. code-block:: console

    conda env update TMIP/EMAT --prune

The *prune* option here will remove packages that are not ordinarily included in the
*EMAT* environment; omit that function if you've installed extra packages that you
want to keep.


Installing TMIP-EMAT in an Existing Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have an existing environment you want to use, or if you'd like to
skip the advice above and install TMIP-EMAT into the base environment, you can
do so using the regular `conda install` tool.  Activate the environment you want
to install into, and then run:

.. code-block:: console

    conda install emat -c tmip -c defaults -c conda-forge

The extra channels (`-c channel_name`) here are required as TMIP-EMAT depends on
other packages from a variety of places.  Because of these dependencies, there
is a fair chance that installing TMIP-EMAT into an existing environment may
cause incompatibilities with other tools, so installing in this manner is not
recommended.


Using an Environment
~~~~~~~~~~~~~~~~~~~~

When using the terminal (MacOS/Linux) or an Anaconda Prompt (Windows), the
current environment name will be shown as part of the prompt:

.. code-block:: console

    (base) C:\Users\cfinley>


By default, when opening a new terminal the environment is set as the
``base`` environment, although this is typically not where you want to
be if you have followed the advice above.  Instead, to switch environments
use the ``conda activate`` command.  For example, to activate the ``EMAT``
environment installed in the quick start, run:

.. code-block:: console

    (base) C:\Users\cfinley> conda activate EMAT
    (EMAT) C:\Users\cfinley>



Running Jupyter
---------------

The most convenient interface for interactive use of TMIP-EMAT is within
a `Jupyter Notebook <https://jupyter.org>`_. The notebook provides a
convenient interactive interface, allowing you to enter Python commands
and see (and interact with) the output in a web browser.
To use Jupyter Notebook, open the terminal (MacOS/Linux) or an Anaconda
Prompt (Windows), activate the EMAT environment, navigate to the
directory where you can find your notebook file, and run it the the
`jupyter-notebook` command.  For example:

.. code-block:: console

    (base) C:\Users\cfinley> conda activate EMAT
    (EMAT) C:\Users\cfinley> cd Documents\Modeling
    (EMAT) C:\Users\cfinley\Documents\Modeling> jupyter-notebook myfilename.ipynb

Alternatively, the next generation interface of Jupyter is called
`JupyterLab <https://jupyterlab.readthedocs.io/en/stable/>`_.
JupyterLab integrates many more features and provides for running
multiple notebooks, and multiple views of the same notebook.
It is in general compatible with TMIP-EMAT, although some of the
interactive exploratory visualizations may be less responsive in
JupyterLab than the Notebook interface alone.  You may also need
to install one or more JupyterLab extensions to enable the full
suite of TMIP-EMAT functionality.

If it's not already installed in your base or working
environments, you can install JupyterLab using conda:

.. code-block:: console

    conda install -c conda-forge jupyterlab

Then to start JupyterLab,

.. code-block:: console

    jupyter lab

JupyterLab will open automatically in your browser.
