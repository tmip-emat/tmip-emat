
.. py:currentmodule:: emat.analysis


Analysis
========

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

Robust Optimization
-------------------

Optimization is finding a set of policy decisions that give the
best outcome on one particular measure.  Robust optimization
generalizes this, seeking to find a set of policy decisions that
gives good outcomes on multiple measures.  This is generally
set up as a multi-objective optimization, so that decision makers
can see and understand the tradeoffs between different outcomes.


