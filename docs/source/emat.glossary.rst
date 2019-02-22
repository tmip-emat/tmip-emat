.. _glossary:

Glossary
========

This glossary of terms used in TMIP-EMAT is based on, and heavily influenced by, the
`robust decision making glossary <https://www.rand.org/methods/rdmlab/glossary.html>`_
published by The RAND Corporation.  There are some minor difference and additions
that are specifically tailored to the TMIP-EMAT approach demonstrated here.

.. glossary::

    box
        A box is a subset of cases in a :term:`design` of experiments, containing
        only those :term:`cases <case>` that meet certain restrictions.  Typically,
        these restrictions are expressed as limited ranges on a particular set of
        parameters or |measures|.  A box can be found as the result of certain
        EMA methodologies (e.g. PRIM, CART) or simply generated manually during
        an exploration of the data.

    CART
        Classification and Regression Trees is a relatively greedy but long-standing
        approach to developing interesting boxes for model exploration.
        :ref:`(Breiman, 1984) <Breiman1984>`.

    case
        A case is a single modeling experiment, defined by a particular :term:`policy`,
        :term:`scenario`, and set of |measures| outcomes.

    core model
        A model that represents some relationships between |uncertainties|, |levers|, and
        |measures|.  The model can be implemented as
        a Python function, an Excel workbook, or any other computer-based model
        that can be evaluated automatically from Python (e.g., at the command line).

    design
        A design of experiments is a group of cases generated in some systematic manner.
        Designs can be as simple as a Monte Carlo sample, where a number of independent
        random draws are made from the possible input parameters, or more structured
        random (e.g. latin hypercube) or non-random (e.g. factorial) designs

    experiment
        The execution of the model to calculate |measures|.
        See also :term:`case`.

    lever
        The *L* in |XLRM|, a lever is a single policy strategy that might
        be implemented by decision maker(s).  Like
        |uncertainties|, a lever is an input to the underlying |model|.  Unlike
        |uncertainties|, a lever does not have any distributional assumptions.

    measure
        The *M* in |XLRM|, this is a single performance measure that
        can be used to evaluate the performance of the system, including
        whether or not decision maker's goals are being met.  The current version of
        TMIP-EMAT assumes that each measure is a single scalar outcome, although any number
        of performance measures can be included, so that a vector of outcomes (e.g.,
        traffic volumes across a sequence of screen lines along a corridor) can be
        jointly considered.  Future version of TMIP-EMAT may incorporate time series
        and/or array-based performace measures, which are explicitly available in the
        underlying EMA Workbench source code.

    meta-model
        A meta-model is an analytical approximation of some |core model|. Typically
        a meta-model is used when the execution of the |core model| is computationally
        expensive.  The meta-model complements the |core model| and can approximate
        the results of the |core model| in a fraction of the time, but it is not a
        replacement for having a |core model| in the first place.

    model
        A model that represents some relationships between |uncertainties|, |levers|, and
        |measures|.  The model can be a |core model| used directly, or a :term:`meta-model`
        derived from a |core model|.

    parameter
        A parameter is an input to a |model|.  This is a generic term that groups
        together |uncertainties|, |levers|, as well as any other constant values
        that are passed as inputs to a |model| (as may happen for model inputs that
        are potentially changeable but are set to fixed values within the current
        exploratory |scope|).

    policy
        A set of values for all the various |levers| defined in an
        exploratory modeling scope.

    PRIM
        The Patient Rule Induction Method, a "bump hunting" technique introduced by
        :ref:`Friedman and Fisher (1999) <FriedmanFisher1999>`.

    scenario
        A set of values for all the various |uncertainties|
        defined in an exploratory modeling scope.

    scope
        An exploratory modeling scope is a collection of definitions for the |uncertainties|,
        |levers|, and
        |measures| under consideration.

    uncertainty
        The *X* in |XLRM|, and uncertainty is a single exogenous risk factor that
        represents an unknown and uncontrolled future
        state of the system under study.  An uncertainty is an input to the underlying
        |model|, and is characterized by a random variable distribution.  In
        many exploratory modeling contexts and for certain types of exploratory modeling
        analysis, analysts explicitly disavow making probabilistic
        statements about the possible values of the uncertainty factors, and instead
        merely characterize uncertainty as a range (for numeric-type uncertainties) or
        a set of possible values (for boolean or categorical type uncertainties).

    XLRM
        A general framework for conducting exploratory analysis, as proposed in
        :ref:`Lembert et al (2003) <Lempert2003>`.
        The letters refer two four principal components of the analysis:

        - **X**: |uncertainties|
        - **L**: |levers|
        - **R**: relationships between inputs and outputs as represented by a |model|.
        - **M**: |measures|





.. |measures| replace:: performance :term:`measures <measure>`
.. |uncertainties| replace:: exogenous :term:`uncertainties <uncertainty>`
.. |levers| replace:: policy :term:`levers <lever>`
.. |core model| replace:: :term:`core model`
.. |model| replace:: :term:`model`
.. |XLRM| replace:: :term:`XLRM`
.. |scope| replace:: :term:`scope`
