.. py:currentmodule:: emat


Scope File
==========

The Scope File is where the scope is defined for the analysis. The file is
implemented in YAML and contains the selected model inputs (uncertainties, levers) and
and outputs (measures) for the analysis and the necessary meta-data on each. The scope
file is used when initializing an EMAT Scope.

Scope Name and Description
--------------------------

Each scope file begins with a **scope:** section that defines the name and description of the
scope. Both the **name:** and **desc:** are input as strings, for example:

::

    scope:
        name: EMAT demo test
        desc: A demo test to fall in love with EMAT


Inputs
------

The **inputs:** section contains the scoped inputs including uncertainties, levers, and
constants that are to be set in the core model. Depending on the input types, there are
slightly different parameters to be set.

For a full set of input options, see the :ref:`Parameters <input_parameters>` documentation.

All inputs are defined by a string name and **ptype:** set where the name is a string value
and the parameter is either *uncertainty*, *lever*, or *constant*.

Uncertainties
#############

Uncertainties are defined by the range or categories of values, and the distribution shape and
parameters across the range. Correlations may also be defined between multiple uncertainty
measures.

An example uncertainty input is below. This example implements an uncertainty variable with a
range between 0.82 and 1.37 with a PERT distribution shaped by a relative peak 33% of the
distance between lower and upper bounds, and a
gamma value of 4. There is also a defined correlation between the distribution of this uncertainty variable
and two other variables in the scope. The default value of this variable is 1.0, which is used for
sensitivity testing as the "baseline" value for this parameter on tests where it is not the
focus (i.e., when examining single-variable sensitivity on other parameters).

::

    Uncertainty Variable 1:
        ptype: uncertainty
        desc: A test uncertainty variable
        dtype: float
        default: 1.0
        min: 0.82
        max: 1.37
        dist:
            name: pert
            rel_peak: 0.33
            gamma: 4
        corr:
            Uncertainty Variable 2: 0.5
            Uncertainty Variable 3: -0.25

Notes on Correlations
~~~~~~~~~~~~~~~~~~~~~

Correlations are by definition bi-directional -- if 'Uncertainty Variable 1' is
set to be correlated at +0.5 with 'Uncertainty Variable 2', then that same correlation
automatically applies in reverse as well, and it is not necessary to give it explicitly
for both parameters (and, if they are both given but at different values, an error is generated
when using the scope file).

Additionally, correlations in aggregate across the entire scope must make sense:  if 'A'
is highly correlated with 'B' and 'B' is highly correlated with 'C', then there must
be at least some correlation between 'A' and 'C'.  Exactly how much is required is a
matter of linear algebra (i.e. the correlation matrix must be positive semi-definite).
In practical applications, logical and consistent assumptions about correlation across
uncertainties should naturally come out fine, and if not, an ``LinAlgError`` will be generated.

Levers
######

Levers differ from Uncertainties as inputs because they do not have a distribution per se,
rather levers are set in a deterministic fashion. However, it is still important for the analysis
that scenarios are tested with levers in all possible positions.

Three example lever definitions are shown below. The first has a simple boolean value, the second has
several categorical values, and the third has a continuous value.

::

    boolean lever:
        ptype: policy lever
        desc: Example boolean lever input
        dtype: bool
        default: False

    categorical lever:
        ptype: policy lever
        desc: Example lever with categorical value
        dtype: cat
        default: category 1
        values:
            - category 1
            - category 2
            - category 3

    continuous value lever:
        ptype: policy lever
        desc: Example lever with a continuous value
        dtype: int
        default: 30
        min: 15
        max: 50

Constants
#########

Constants are inputs with a fixed value in the analysis. This input option is provided to allow the
modeler to fix a set of inputs. This may also be useful as a placeholder in the scoping file. Note
that both the ptype and the dist need to be set to 'constant' in the definition.

::

    constant input:
        ptype: constant
        desc: Example constant input
        dtype: float
        default: 60
        dist: constant


Outputs
-------

The **outputs:** section of the scoping file lists all the measures to be captured from the core model.
Each output is declared individually along with information about any transformation that should be
taken in development of the meta-model and how the output should be treated in the automated analysis,
for example with `EMA Workbench <https://github.com/quaquel/EMAworkbench/tree/v2>`_.

For a full set of output options, see the :ref:`Measures <output_measures>` documentation.  Within
each output measure definition, any keyword argument for the constructor of a :class:`Measure` can
be given in this scope file (with the exception of *function* arguments, which are disabled for
security reasons; use the *transform* argument instead).

The example output below shows an output measure that has a log transformation (the meta-model will
be based on the log of the value of this measure, instead of the original raw value) and the measure
should be minimized in an automated analysis.

::

    Example output measure 1:
        metamodeltype: log
        kind: minimize
