.. _methodindex:

Methodology Index
=================

This listing provides a single alphabetical listing of the various methodological
tools that are a part of TMIP-EMAT.

.. glossary::

    :ref:`Classification and Regression Trees (CART) <methodology-cart>`
        Classification and Regression Trees, or CART, is a simple machine learning
        technique for predicting a target variable. Within TMIP-EMAT, the CART
        algorithm is implemented as a |ScenarioDiscovery| method. It is a relatively
        greedy but long-standing approach to developing interesting boxes for model
        exploration. :ref:`(Breiman, 1984) <Breiman1984>`.

    :ref:`Contrast Experiments <methodology-contrast-experiments>`
        The contrast experiments method renders two different set of experiments
        on a common |SPLOM|. This visualization approach makes it easy
        to see if the overall shape of the distriubution of experiment inputs and
        outputs is similar or different. It is particularly useful in validating
        that TMIP-EMAT’s automatically generated meta-models are performing
        correctly.

    :ref:`Design of Experiments <methodology-experimental-design>`
        A "design" of experiments is simply a set or list of particular experiments
        to be run, generated in some prescribed manner.  Designs can be completely
        random (e.g., Monte Carlo simulation), partly random (e.g. Latin Hypercube),
        or completely deterministic (e.g., univariate sensitivity testing, reference
        experiments).

    :ref:`Display Experiments <methodology-display-experiments>`
        The `display_experiments` method generates a |SPLOM| that diplays model
        inputs (uncertainties and policy levers) in one dimension and model outputs
        (performance measures) in the other.

    :ref:`Feature Scoring <methodology-feature-scoring>`
        This is a |ScenarioDiscovery| method for identifying what model inputs have the
        greatest relationship to the outputs, by computing a numerical value
        that summarizes the relative importance of each input in determining
        the level of the output.

    :ref:`Experimental Design <methodology-display-experiments>`
        See :term:`Design of Experiments`.

    :ref:`Exploratory Scoping <methodology-scoping>`
        While not a methodological approach per se, TMIP-EMAT provides a notational
        structure to concretely define the manner in which the XLRM framework is
        to be operationalized for a given travel model (R) and its uncertainties (X),
        policy levers (L), and performance measures (M).

    :ref:`Interactive Visualizer <methodology-interactive-explorer>`
        The Interactive Visualizer in TMIP-EMAT provides a set of tools that can
        display a dynamically generated selection of experiments in a number of
        visualizations, including histograms, scatter plots, and SPLOMs.  The
        dimensional bounds of the select (the "box") can be manipulated by a user
        programmatically or by clicking and dragging directly on the figures.

    :ref:`Latin Hypercube Design of Experiments <methodology-latin-hypercube>`
        A Latin Hypercube is a space-filling mathematical process for making pseudo-random
        draws from a multi-dimensional space.  This kind of design is not formally
        "random" but approximates a random distribution while ensuing a reasonable
        coverage across the spectrum of possible values in each dimension. Meta-models
        for deterministic simulation experiments, such as most transportation models,
        are best supported by a “space filling” design of experiments such as this.

    :ref:`Meta-model Creation <methodology-metamodel-creation>`
        A main feature of TMIP-EMAT is the ability to automatically generate meta-models
        that provide a good approximation of the underlying core model in most situations.
        By default, metamodels derived through TMIP-EMAT include two stages, a linear
        regression model to capture overall trends and a gaussian process regression
        (GPR) model that can capture a wide variety of non-linear effects.

    :ref:`Monte Carlo Simulation <methodology-monte-carlo-design>`
        A Monte Carlo simulation is a simple random (or in more precise computer
        science terminology, pseudo-random) process for generating a design of
        experiments.  It is not generally an efficient design, but with a large
        enough sample size efficiency is less relevant and simplicity can be
        valuable.

    :ref:`Multi-objective Optimization <methodology-multiobjective-optimization>`
        With exploratory modeling, optimization is also often undertaken as a
        multi-objective optimization exercise, where multiple and possibly
        conflicting performance measures need to be addressed simultaneously.
        Instead of generating one unique "optimal" solution, this TMIP-EMAT
        method can be used to find a spectrum of different solutions.  Each of
        them solves the problem at a different weighting of the various objectives.
        Decision makers can then review the various different solutions, and
        make judgements about the various trade-offs implicit in choosing one
        over another.

    :ref:`Patient Rule Induction Method (PRIM) <methodology-prim>`
        The Patient Rule Induction Method, or |PRIM|, is a |ScenarioDiscovery| method.
        This method is a "bump hunting" technique introduced by
        :ref:`Friedman and Fisher (1999) <FriedmanFisher1999>`, which often provides
        insightful results for complex models.

    :ref:`Policy Contrast <methodology-policy-contrast>`
        The Policy Contrast method in TMIP-EMAT allows an analysst to compare the
        outcomes of two different sets of policies. The tool runs the model across
        a distribution of inputs, and displays the resulting distribution of
        performance measure outputs. Two sets of model runs are generated with the
        same design of experiments for all the non-contrasted distributions, so
        any variation in the performance measures can be unambiguously linked to
        the changes in the specific-value inputs, instead of being a result of
        input stocasticity.

    :ref:`Reference Experiment <methodology-reference-design>`
        A "design of expermiments" which contains only a single experiment, with all
        input values set to their default parameters.

    :ref:`Robust Optimization <methodology-robust-optimization>`
        Robust optimization is a variant of more traditional optimization problems.
        Rather than seeking a solution that provides the best outcome, a robust
        optimization problem is one where we try to find policies that yield good
        outcomes across a broad range of possible futures.  It is common to employ
        various different criteria for what constitutes "good" or "broad", by also
        borrowing methods from from the Multi-objective Optimization tools.

    :ref:`Scatter Plot Matrix (SPLOM) <methodology-splom>`
        The Scatter Plot Matrix, or |SPLOM|, is a visualization method. It is
        a collection of two-dimensional scatter plots arranged in a matrix, where each
        column of plots shares a common x-axis definition, and each row shares a common
        y-axis definition.

    :ref:`Scenario Discovery <Scenario Discovery>`
        This is a broad category of different methodologies used to discover important
        relationships between inputs and outputs across multiple dimensions.

    :ref:`Search over Levers <methodology-search-over-levers>`
        A Search over Levers is a particular style of multi-objective optimization
        for exploratory modeling in the XLRM framework, where the uncertainties are
        held constant at some particular value, and only the policy levers are
        manipulated by the search algorithm.

    :ref:`Threshold Scoring <methodology-threshold-scoring>`
        A variant of |FeatureScoring|, where inputs are scored not with just a single
        numerical value, but with a range of values representing the relative importance
        of inputs for getting the output to be above or below various possible threshold
        values.

    :ref:`Univariate Sensitivity Testing <methodology-univariate-sensitivity>`
        One of the simplest experimental designs is a set of univariate sensitivity
        tests. In this design, a set of baseline model inputs is used as a starting
        point, and then input parameters are changed one at a time to non-default
        values. Univariate sensitivity tests are excellent tools for debugging and
        quality checking the model code, as they allow modelers to confirm that each
        modeled input is (or is intentionally not) triggering some change in the model
        outputs.

    :ref:`Worst Case Discovery <methodology-worst-case-discovery>`
        Worst Case Discovery is a particular style of multi-objective optimization
        for exploratory modeling in the XLRM framework. In this analysis, the policy
        levers are held constant at some particular value, and only the exogenous
        uncertainties are manipulated by the search algorithm.  In addition, the
        directionality of all objective dimensions is inverted, so that the search
        algorthim seeks to find values for the input that lead to worse outcomes
        instead of better ones.



.. |PRIM| replace:: :term:`PRIM`
.. |SPLOM| replace:: :term:`SPLOM`
.. |ScenarioDiscovery| replace:: :term:`Scenario Discovery`
.. |FeatureScoring| replace:: :term:`Feature Scoring`