.. py:currentmodule:: emat.model


Core Models
===========

.. toctree::
    :hidden:

    abstract
    python_based
    excel
    files
    gbnrtc


The :class:`AbstractCoreModel` provides a basic interface structure
for interacting with models used in exploratory analysis. All
specific model types included with TMIP-EMAT and all new model connections
should inherit from this base class.

.. rubric:: :doc:`Python Models <python_based>`

The simplest instantiation of an exploratory model for use with TMIP-EMAT
is a Python function.  Such a function can be wrapped with a
:class:`PythonCoreModel`.

.. rubric:: :doc:`Excel Models <excel>`

TMIP-EMAT can also be used with Excel-based models, using the
:class:`ExcelCoreModel` interface.

.. rubric:: :doc:`Bespoke Models <gbnrtc>`

Most advanced travel models can be set up to run programmatically, generating
a variety of performance measures to one or more arbitrarily defined files on
disk.  TMIP-EMAT provides a :class:`FilesCoreModel` to serve as the starting
point for developing model-specific implementations for custom models than can
be operated in this manner.

For example, the model for the Greater Buffalo-Niagara Regional Transportation Council
is a TransCAD based model.  The :class:`GBNRTCModel` is included in TMIP-EMAT as an example
of how a more complex external travel demand model can be integrated into
the TMIP-EMAT framework.




