
.. py:currentmodule:: emat.analysis


Analysis
========

Visualization
-------------

Visualization tools are an important part of the exploratory
modeling process.  They allow analysts and other stakeholders
to conceptualize the potentially complex relationships represented
in transportation models.

.. toctree::

    splom
    corruption
    interactive-explorer-v2


Scenario Discovery
------------------

Scenario discovery in exploratory modeling is focused on finding
scenarios that are interesting to the user.
The process generally begins through the identification
of particular outcomes that are “of interest”, and then discovering
what factor or combination of factors can result in those outcomes.
There are a variety of methods to use for scenario discovery.
We illustrate a few here.

.. toctree::

    feature-scoring
    prim
    cart


Directed Search
---------------

The scenario discovery tools outlined above are focused on
exploring parameters and outcomes across a pre-set sample of
model runs.  This sample can be small or quite large, but the
tools only consider cases that have already been evaluated.
In directed search, we transition to tools that will propose
and execute *new* cases.

.. toctree::

    policy-contrast
    optimization


