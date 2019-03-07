

Installation and Configuration
==============================

**Installation**

Installation of TMIP-EMAT is currently a manual process.

Instructions for installing via `Anaconda <https://www.anaconda.com/download>`_
will be published here when that installation path is available.

.. note::

	Python has two versions (2 and 3) that are available and currently maintained,
	but only through the end of 2019.  From 2020 on,
	`only Python 3 will be supported <https://python3statement.org>`_.
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


