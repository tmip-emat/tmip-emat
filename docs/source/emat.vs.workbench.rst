

Difference between TMIP-EMAT and EMA Workbench
==============================================

The methodologies demonstrated in this toolset build on and extend the
capabilities of the `EMA Workbench, version 2 <https://github.com/quaquel/EMAworkbench/tree/v2>`_.
This document outlines some of the principal differences between these two
projects.

1. TMIP-EMAT includes an automated :term:`meta-model` generation capability.

    Many transportation planning models have exceptionally long computation
    times -- requiring hours-to-days of computer time to generate a single
    model run.  It is infeasible to run models like this enough times to
    conduct a thorough exploratory analysis directly.

#. TMIP-EMAT directly incorporates a database system for storing results.

    Although long term storage of results is not precluded when using
    the EMA workbench, persisting model results to long-term storage is
    left up to the user.  This is important for transportation planning
    especially in light of point 1 above.

#. TMIP-EMAT simplifies the processing of outcome measures by using only scalar values.

    Although input measures in version 2 are processing in a pandas.DataFrame,
    outcomes generated using the EMA workbench are not currently processed in the
    same manner, to preserve the flexibility of outputs to include time series and
    array types.  This may change in the future, as the capability of pandas to
    accommodate time series and array types as
    :ref:`native cell types <extending.extension-types>` matures.

#. TMIP-EMAT provides documentation and guides with a transportation-planning focus.

    As it is designed to serve as a demonstration of the methodological
    approaches to exploratory modeling in transportation, the examples and
    guides provided may be more useful to transportation planners than the
    more general documentation available for the EMA Workbench.  Moreover,
    the TMIP-EMAT includes an example built from a TransCAD model, to
    demonstrate the potential and provide a road map for integrating other
    bespoke transportation models.

