
Introduction to EMAT
====================

TMIP-EMAT is a methodological approach to exploratory modeling and analysis.
It provides a window to rigorous analytical methods for handling uncertainty
and making well informed decisions using travel forecasting models of all
types. It is designed to integrate with and enhance an **existing transportation model** or
tool to perform exploratory analysis of a range of possible scenarios.  In the
documentation of TMIP-EMAT, we refer to the existing model or tool as the
"core model".

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

To be clear, TMIP-EMAT is *not* a standalone model or tool by itself, it *must*
be integrated with a separate core model.  Moreover, the quality of any analysis undertaken
with TMIP-EMAT depends on the quality and capabilities of the underlying core
model.  If the core model does not contain an explicit representation of the
transportation network, then TMIP-EMAT will not allow an analyst to study policy
questions that hinge on the microscopic details of traffic congestion.

TMIP-EMAT is presented as a flexible, methodological approach applicable to many
different core models. It is not a fully-developed end-to-end software solution.
Because of this, developing a new implementation of TMIP-EMAT to connect with a
new core model will require at least one developer with some technical expertise.
The documentation and code published with TMIP-EMAT is meant as guide and a start,
but someone with detailed knowledge of the technical operation of the core model
and at least basic Python skills will need to write a connector between the core
model and the TMIP-EMAT tools.

The Core Model
--------------

The core model itself does not need to be in Python, it can be created and run in
any computer language. It should take a collection of inputs and generate one or more
outputs, or "performance metrics", of interest.  Inputs can include variable
inputs (e.g., fuel cost) as well as model parameter inputs (e.g., the elasticity
of vehicle travel with respect to fuel cost).  Examples of a core model include,
but are not necessarily limited to the following:

    * Regional or statewide travel demand models
    * Activity-based travel models
    * Trip-based travel models
    * Sketch planning or spreadsheet model
    * Microsimulation models
    * Corridor-level travel model

TMIP-EMAT can be used to systematically explore uncertainties in input variables
and model parameters, and the impact that those uncertainties have on performance
metrics.  It is useful for examining model forecasts as a range of model outcomes
rather than a single outcome, and it provides a mechanism for defining
uncertainties and visualizing outputs.

TMIP-EMAT can also be used to understand how uncertainties interact with policy
decisions (e.g., extending a transit line), where *uncertainties* relate to model
inputs and variables that are outside of the policy-maker's control, and policy
*levers* are model inputs that are within the policy-maker’s control.

If the existing tool or model is computationally expensive to run, TMIP-EMAT can
generate meta-models of the core model that describe how a set of model inputs
impact specific performance metrics.  These meta-models are formulated as
regression models of core model outputs that run very quickly (microseconds)
and allow model input uncertainties to be explored systematically while limiting
the number of computationally expensive core model runs.


End User Requirements
---------------------

First and foremost, an existing core model is required, and there should exist
a desire to explore the model systematically to better understand uncertainties
and their impact on potential future outcomes.

The core model should be *fully calibrated and validated* prior to integrating
with TMIP-EMAT.  Since TMIP-EMAT uses the core model as the basis for analysis,
any deficiencies with the core model will propagate to TMIP-EMAT results and
potentially lead the user to draw inappropriate conclusions.

The ability to run the core model programmatically, rather than manually, is
strongly recommended.  Although it is possible to conduct this kind of analysis
starting from manual operation of the core model, that process tends to be
error-prone, and automating the execution of the core model will both reduce
errors and increase overall modeling efficiency.

As noted above, deploying TMIP-EMAT on a programmatically executable core model requires
the existence or development of an application programming interface (API) to the
existing core model.  The API enables TMIP-EMAT to programmatically define
scenarios, launch and run the core model, retrieve errors and status, and
import metrics from the core model.  This API must also have a Python-facing
interface to connect with TMIP-EMAT, even if the core model itself does not
use Python.

Lastly, the core model should be directly sensitive to the policies under study,
or capable of being adjusted in some manner to be sensitive to those policies.
For example, a traditional travel demand model might be made sensitive to some
of the impacts of the introduction of autonomous cars though adjustments to
parameters for vehicle availability, highway capacity, or the value of in-vehicle
travel time.  On the other hand, making such a model sensitive to the introduction
of personal aerial vehicles might not be possible without major changes to the
basic structure of the model.


Model Inputs, Outputs, and Configurations
-----------------------------------------

Inputs
~~~~~~

*   **Core Model**.
    The core model is interfaced with TMIP-EMAT using an API.  The API enables
    TMIP-EMAT to programmatically define scenarios, launch, retrieve errors and
    status, and import metrics from the core model.  The API should allow for
    configuration of all uncertainties and policy levers that are input to the
    system as well as configuration of the desired performance metrics.  The
    core model should be well-validated to ensure model sensitivities are
    reasonable.
*   **Uncertainty Definitions**.
    Uncertainty definitions include the overall range, correlation, and
    distribution of the risk variables that were selected for the analysis.
    Uncertainties represent exogenous inputs to the core model that impact the
    forecasts of the core model, and may include input variables, model
    parameters, or model structures.  The set of uncertainties input to TMIP-EMAT
    will typically be smaller than the full domain of inputs to the core model.
    Uncertainties should be selected based on the importance of the variable in
    the context of the scope of the analysis (considering policy levers and
    metrics of interest) and the relative impact the variable has on relevant
    performance metrics.
*   **Policy Lever Definitions**.
    Policy lever definitions include the specification of specific strategies/choices
    to test in the analysis, including the range in potential lever options.
    Levers (i.e., policy levers) represent inputs to the core model that impact
    the model’s forecasts, but are controllable by planners or decision makers.
    They can include individual variable inputs to the model (e.g., toll price)
    or can represent a portfolio of changes to the model (e.g., a transit line
    extension).
*   **Performance Metric Definitions**.
    The set of metrics that will be analyzed must be defined.  A performance
    metric is an output of the core model and represents a gage by which the
    impact of changes in uncertainties and levers can be measured.  Often core
    models will have a large number of intermediate and final outputs that could
    be considered here.  Metrics should be selected based on their relevance to
    the analysis and for decision makers.

Outputs
~~~~~~~

*   A primary output of TMIP-EMAT is a database of simulation runs of
    the model, including the associated uncertainty and policy lever inputs for
    each simulation and the performance metric outputs for each run.
    TMIP-EMAT uses either a Monte Carlo or Latin Hypercube approach to sample across the uncertainty inputs
    from their defined distributions and to sample across the potential values of
    each lever.  For each simulation, the collection of model inputs is used along
    with the core model (or meta-model representations of the core model) to generate
    the set of performance metric outputs for that simulation.  The user has the
    ability to specify the number of simulations that are performed.

*   The meta-models developed to describe the relationships between uncertainties,
    levers, and metrics are themselves outputs of TMIP-EMAT (this only applies when
    meta-models are used).  The meta-models serve as direct input to the EMA Workbench
    to support guided exploratory analysis and can be useful for validating the core
    model runs (e.g., to verify that model sensitivities to input variables and/or
    parameters are reasonable and appropriate).

*   A variety of tools are built into TMIP-EMAT that can be used to generate
    visualizations and tables to better understand the outputs.  Such tools include
    the following:

    -   Risk Analysis Visualizations

        +   Tabulations of performance metrics can be generated showing percentile
            ranges of each metric.  Tabulations can be segmented across different
            values of levers included in the analysis.
        +   Two-way scatter plots of model run results can be produced, showing
            the impact of uncertainties on performance metrics for different policy
            levers.
        +   The relative importance/contribution of uncertainties to the overall
            range in metrics can be plotted to gain an understanding for which
            uncertainties carry the greatest impacts.

    -   Open-Ended Exploratory Analysis Visualizations

        +   Interactive tools are available in TMIP-EMAT to examine the relationships
            between performance metrics and policy levers.  Interactive sliders and
            toggle functions can be used to refine the exploratory analysis to
            specific sets of simulation runs.
        +   Conditional uncertainty distribution charts can be used to illustrate
            the importance (or lack thereof) of uncertainty variables for achieving
            specific targets in metrics under different policies.

    -   Guided Exploratory Analysis Visualizations

        +   Patient Rule Induction Method (:term:`PRIM`) tradeoff curves and
            performance tabulations illustrate the number of
            scenarios meeting specific performance metric criteria for scenario
            discovery.
        +   Other :term:`PRIM` visualizations indicate the restricted ranges of uncertainties
            and levers that were identified by the algorithm.
        +   Robust search outputs can be tabulated or plotted in parallel axis charts
            to show the relationships between different policies and corresponding
            objectives. Such visualizations illustrate the set of scenarios that represent
            the Pareto optimal solutions to achieve a set of optimization criteria
            specified by the user for the robust search algorithm.

Configurations
~~~~~~~~~~~~~~

*   **EMA Scoping File**.
    A single input configuration file is used to specify the configurations for each
    TMIP-EMAT analysis.  The input configuration file uses a specific format to specify
    the required inputs TMIP-EMAT, including the following:

    -   Uncertainty definitions.
    -   Lever definitions.
    -   Metric definitions.

*   **Guided Search Configurations**.
    A number of configurations must be set by the analyst when working with the guided
    search tools, which are linked to the EMA Workbench.

    -   *PRIM Configurations*.
        PRIM requires that the analyst preselect a set of scenarios of interest.
        The set of scenarios can be defined based upon any of the uncertainties,
        levers, or metrics, though usually they are set based upon metrics to better
        understand the inputs that result in a particular set of scenarios meeting
        performance metric criteria.
    -   *Robust Search Configurations*.  Robust search requires several configurations,
        including the following:

        +   Robustness metrics define the multi-criteria optimization, specifying the
            set of metrics (or functions thereof) that should be optimized.
        +   Constraints define criteria for which a metric must meet (rather than
            be optimized).
        +   The maximum number of iterations define stopping criteria for the
            algorithm.
        +   The number of scenarios define how many draws the algorithm uses of the
            TMIP-EMAT model to perform the optimization.


