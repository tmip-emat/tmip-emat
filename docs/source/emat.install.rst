

Installation and Configuration
==============================

**Installation**

Installation of TMIP-EMAT can be done through the Anaconda package manager,
or from source.  Installing from the package manager is recommended for
users who are interested in the exploratory modeling tools, but are not
responsible for conducting core model runs.  For modelers who wish to
integrate a new core model into EMAT, installing from and editing the
source code is recommended.

.. toctree::

    emat.conda
    emat.git


.. note::

	Python has two versions (2 and 3) but .
	`only version 3 is currently maintained for most tools <https://python3statement.org>`_.
	TMIP-EMAT is compatible *only* with version 3.6 or later.

**Core Model Configuration**

Depending on the particular model being used, a core model configuration file
may be necessary to define the folder locations, paths to executables and
other necessary information about the core model installation on the local
system. The actual contents of this file will be determined by the requirements
of the core model. The file is loaded when the core model is initialized.

For an example of a configuration file in use, see the :ref:`GBNRTC example model <gbnrtc-example-model>`

Each TMIP-EMAT deployment will have a single core model configuration file.
Settings that are specific to a particular application are defined in the Scope
File.


