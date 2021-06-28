.. TMIP-EMAT documentation master file, created by
   sphinx-quickstart on Mon Dec 31 17:19:33 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/TMIP.png
    :target: https://www.fhwa.dot.gov/planning/tmip/
    :align: left
    :alt: TMIP
    :class: banner

.. image:: _static/BetterMethodsBetterOutcomes.png
    :target: https://www.fhwa.dot.gov/planning/tmip/
    :align: left
    :alt: Better Methods, Better Outcomes
    :class: banner

TMIP-EMAT documentation
=======================

.. warning:: The views expressed in this documentation do not necessarily
    represent the opinions of FHWA, and do not constitute an endorsement,
    recommendation, or specification by FHWA.

TMIP-EMAT is a *methodological approach* to exploratory modeling and analysis.
It provides a window to rigorous analytical methods for handling uncertainty
and making well informed decisions using travel forecasting models of all
types. It is designed to integrate with an existing transportation model or
tool to perform exploratory analysis of a range of possible scenarios.

TMIP-EMAT provides the following features to enhance the functionality of the
underlying core model:

- A structure to formalize and distill an exploratory **scope**, in a manner
  suitable for translating the abstraction of the "XLRM" robust decision making
  framework into a concrete, application-specific form,
- A systematic process for **designing experiments** to be evaluated
  using the core model, and support for running those experiments in an
  automated fashion.
- A **database** structure to organize and store the results from a large
  number of experiments.
- A facility to automatically create a **metamodel** from experimental
  results, which uses machine learning techniques to rapidly approximate
  new outputs of the core model for without actually running it for every
  relevant combination of inputs.
- A suite a **analytical and visualization tools** to explore the relationships
  between modeled inputs and outputs, and develop robust policy strategies that
  might be effective across a range of possible future scenarios.

The methodologies demonstrated in this toolset build on and extend the
capabilities of the `EMA Workbench, version 2 <https://github.com/quaquel/EMAworkbench/tree/v2>`_.
A brief discussion of the differences between these projects is
:doc:`available <source/emat.vs.workbench>`.

For an introduction to the concepts and tools contained in TMIP-EMAT,
you can view our TMIP Webinars and Screencasts:

* `Introducing the Exploratory Modeling and Analysis Tool <https://tmip.org/content/tmip-webinar-introducing-exploratory-modeling-and-analysis-tool-tmip-emat>`_
* `Using TMIP-EMAT for Exploratory Analysis <https://tmip.org/content/using-tmip-emat-exploratory-analysis>`_
* `TMIP-EMAT: Setting up a Model <https://www.youtube.com/watch?v=bQ5ITYb_uHU>`_
* `TMIP-EMAT: Running a Model <https://www.youtube.com/watch?v=UwmHeRO2tXg>`_
* `TMIP-EMAT: Scatter Plot Matrixes and Feature Scoring <https://www.youtube.com/watch?v=e_ZDBFxRC6g>`_
* `How to Get Started with TMIP-EMAT <https://tmip.org/content/how-get-started-tmip-emat>`_
* `Using TMIP-EMAT for Decision Making under Deep Uncertainty <https://tmip.org/content/tmip-dmdu-project-meeting-tmip-emat>`_
* `Connecting TMIP-EMAT and VisionEval for Exploratory Analysis <https://tmip.org/content/connecting-tmip-ematand-visioneval-exploratoryanalysis>`_

In addition to this documentation, there are several helpful example
repositories hosted on GitHub:

* `TMIP-EMAT Source <https://github.com/tmip-emat/tmip-emat>`_
  The main repository for TMIP-EMAT source code that demonstrates various
  aspects of the exploratory modeling methodological approach.

* `Using TMIP-EMAT with VisionEval <https://github.com/tmip-emat/tmip-emat-ve>`_
  This repository demonstrates the use of TMIP-EMAT with the VisionEval's
  Regional Strategic Planning Model (RSPM).

* `Using TMIP-EMAT with a Bespoke External Model <https://github.com/tmip-emat/tmip-emat-bespoke-demo>`_
  This repository demonstrates connecting TMIP-EMAT with an arbitrary
  custom model that is controlled by configuration files and invoked on
  the command line. A similar approach can be used to connect TMIP-EMAT
  to most commercial and proprietary travel demand models.

TMIP-EMAT is provided under a `BSD 3-Clause License license <https://github.com/tmip-emat/tmip-emat/blob/master/LICENSE>`_.
It also includes similarly licensed code from the `EMA Workbench <https://github.com/quaquel/EMAworkbench>`_.


.. toctree::
   :maxdepth: 3
   :caption: Contents:
   :includehidden:

   source/emat.intro
   source/emat.install
   source/emat.scope/index
   source/emat.design/emat.design
   source/emat.models/index
   source/emat.metamodels/index
   source/emat.database/index
   source/emat.analysis/index
   source/emat.examples/index
   source/emat.glossary
   source/emat.method-index
   source/emat.references



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


