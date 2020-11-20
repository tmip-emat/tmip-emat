# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [raw] {"raw_mimetype": "text/restructuredtext"}
# .. py:currentmodule:: emat

# %% [markdown]
# # Road Test

# %%
import emat, os, numpy, pandas, functools, asyncio
emat.versions()

# %%
logger = emat.util.loggers.log_to_stderr(30, True)

# %% [markdown]
# ## Defining the Exploratory Scope

# %% [raw] {"raw_mimetype": "text/restructuredtext"}
# The model scope is defined in a YAML file.  For this Road Test example, the scope file is named 
# :ref:`road_test.yaml <road_test_scope_file>` and is included in the model/tests directory.

# %%
road_test_scope_file = emat.package_file('model','tests','road_test.yaml')

# %% [raw] {"raw_mimetype": "text/restructuredtext"}
# The filename for the YAML file is the first argument when creating a :class:`Scope`
# object, which will load and process the content of the file.

# %%
road_scope = emat.Scope(road_test_scope_file)
road_scope

# %% [markdown]
# A short summary of the scope can be reviewed using the `info` method.

# %%
road_scope.info()

# %% [markdown]
# Alternatively, more detailed information about each part of the scope can be
# accessed in four list attributes:

# %%
road_scope.get_constants()

# %%
road_scope.get_uncertainties()

# %%
road_scope.get_levers()

# %%
road_scope.get_measures()

# %% [markdown]
# ## Using a Database
#
# The exploratory modeling process will typically generate many different sets of outputs,
# for different explored modeling scopes, or for different applications.  It is convenient
# to organize these outputs in a database structure, so they are stored consistently and 
# readily available for subsequent analysis.
#
# The `SQLiteDB` object will create a database to store results.  When instantiated with
# no arguments, the database is initialized in-memory, which will not store anything to
# disk (which is convenient for this example, but in practice you will generally want to
# store data to disk so that it can persist after this Python session ends).

# %%
emat_db = emat.SQLiteDB('tempfile')

# %% [markdown]
# An EMAT Scope can be stored in the database, to provide needed information about what the 
# various inputs and outputs represent.

# %%
road_scope.store_scope(emat_db)

# %% [markdown]
# Trying to store another scope with the same name (or the same scope) raises a KeyError.

# %%
try:
    road_scope.store_scope(emat_db)
except KeyError as err:
    print(err)

# %% [markdown]
# We can review the names of scopes already stored in the database using the `read_scope_names` method.

# %%
emat_db.read_scope_names()

# %% [markdown]
# ## Experimental Design
#
# Actually running the model can be done by the user on an *ad hoc* basis (i.e., manually defining every 
# combination of inputs that will be evaluated) but the real power of EMAT comes from runnning the model
# using algorithm-created experimental designs.
#
# An important experimental design used in exploratory modeling is the Latin Hypercube.  This design selects
# a random set of experiments across multiple input dimensions, to ensure "good" coverage of the 
# multi-dimensional modeling space.
#
# The `design_latin_hypercube` function creates such a design based on a `Scope`, and optionally
# stores the design of experiments in a database.

# %%
from emat.experiment.experimental_design import design_experiments

# %%
design = design_experiments(road_scope, db=emat_db, n_samples_per_factor=10, sampler='lhs')
design.head()

# %%
large_design = design_experiments(road_scope, db=emat_db, n_samples=5000, sampler='lhs', design_name='lhs_large')
large_design.head()

# %% [markdown]
# We can review what experimental designs have already been stored in the database using the 
# `read_design_names` method of the `Database` object.

# %%
emat_db.read_design_names('EMAT Road Test')

# %% [markdown]
# ## Core Model in Python

# %% [raw] {"raw_mimetype": "text/restructuredtext"}
# Up until this point, we have been considering a model in the abstract, defining in the :class:`Scope` what the inputs 
# and outputs will be, and designing some experiments we would like to run with the model.  Now we will actually 
# interface with the model itself. 

# %% [markdown]
# ### Model Definition
#
# In the simplest approach for EMAT, a model can be defined as a basic Python function, which accepts all
# inputs (exogenous uncertainties, policy levers, and externally defined constants) as named keyword
# arguments, and returns a dictionary where the dictionary keys are names of performace measures, and 
# the mapped values are the computed values for those performance measures.  The `Road_Capacity_Investment`
# function provided in EMAT is an example of such a function.  This made-up example considers the 
# investment in capacity expansion for a single roadway link.  The inputs to this function are described
# above in the Scope, including uncertain parameters in the volume-delay function,
# traffic volumes, value of travel time savings, unit construction costs, and interest rates, and policy levers including the 
# amount of capacity expansion and amortization period.

# %%
from emat.model.core_python import PythonCoreModel, Road_Capacity_Investment

# %% [raw] {"raw_mimetype": "text/restructuredtext"}
# The :class:`PythonCoreModel <emat.model.core_python.core_python_api.PythonCoreModel>` object 
# provides an interface that links the basic Python function that represents 
# the model, the :class:`Scope <emat.scope.scope.Scope>`, and optionally the 
# :class:`Database <emat.database.database.Database>` used to manage data storage.

# %%
m = PythonCoreModel(Road_Capacity_Investment, scope=road_scope, db=emat_db)

# %% [raw] {"raw_mimetype": "text/restructuredtext"}
# From the :class:`PythonCoreModel`, which links the model, scope, design, and database, we can run the design of experiments.  
# This will systematically run the core model with each set of input parameters in the design, store the results in
# the database, and return a pandas.DataFrame containing the results.

# %% [markdown]
# ### Model Execution

# %%
lhs_results = m.run_experiments(design_name='lhs')
lhs_results.head()

# %% [markdown]
# If running a large number of experiments, it may be valuable to parallelize the 
# processing using a DistributedEvaluator instead of the default SequentialEvaluator.
# The DistributedEvaluator uses dask.distributed to distribute the workload to
# a cluster of processes, which can all be on the same machine or distributed
# over multiple networked computers. (The details of using dask.distributed in 
# more complex environments are beyond this scope of this example, but interested
# users can refer to that package's [documentation](https://distributed.dask.org/).)

# %%
lhs_large_async = m.async_experiments(large_design, max_n_workers=8, batch_size=157)

# %%
lhs_large_results = await lhs_large_async.final_results()

# %% [markdown]
# Once a particular design has been run once, the results can be recovered from the database without re-running the model itself.

# %%
reload_results = m.read_experiments(design_name='lhs')
reload_results.head()

# %% [markdown]
# It is also possible to load only the parameters, or only the performance meausures.

# %%
lhs_params = m.read_experiment_parameters(design_name='lhs')
lhs_params.head()

# %%
lhs_outcomes = m.read_experiment_measures(design_name='lhs')
lhs_outcomes.head()

# %% [markdown]
# ## Feature Scoring

# %%
m.get_feature_scores('lhs')

# %% [markdown]
# ## Visualization

# %%
from emat.analysis import display_experiments
display_experiments(road_scope, lhs_results, rows=['time_savings', 'net_benefits', 'input_flow'])

# %% [markdown]
# ## Scenario Discovery

# %% [markdown]
# Scenario discovery in exploratory modeling is focused on finding scenarios that are interesting to the user.  
# The process generally begins through the identification of particular outcomes that are "of interest",
# and the discovery process that can seek out what factor or combination of factors can result in
# those outcomes.
#
# There are a variety of methods to use for scenario discovery.  We illustrate a few here.
#

# %% [markdown]
# ### PRIM
#
# Patient rule induction method (PRIM) is an algorithm that operates on a set of data with inputs and outputs.  
# It is used for locating areas of an outcome space that are of particular interest, which it does by reducing 
# the data size incrementally by small amounts in an iterative process as follows:
#     
# - Candidate boxes are generated.  These boxes represent incrementally smaller sets of the data.  
#   Each box removes a portion of the data based on the levels of a single input variable.
#   * For categorical input variables, there is one box generated for each category with each box 
#     removing one category from the data set.
#   * For integer and continuous variables, two boxes are generated – one box that removes a 
#     portion of data representing the smallest set of values for that input variable and another 
#     box that removes a portion of data representing the largest set of values for that input.  
#     The step size for these variables is controlled by the analyst.
# - For each candidate box, the relative improvement in the number of outcomes of interest inside 
#   the box is calculated and the candidate box with the highest improvement is selected.
# - The data in the selected candidate box replaces the starting data and the process is repeated.
#
# The process ends based on a stopping criteria.  For more details on the algorithm, 
# see [Friedman and Fisher (1999)](http://statweb.stanford.edu/~jhf/ftp/prim.pdf) or 
# [Kwakkel and Jaxa-Rozen (2016)](https://www.sciencedirect.com/science/article/pii/S1364815215301092).
#
# The PRIM algorithm is particularly useful for scenario discovery, which broadly is the process of 
# identifying particular scenarios of interest in a large and deeply uncertain dataset.   
# In the context of exploratory modeling, scenario discovery is often used to obtain a better understanding 
# of areas of the uncertainty space where a policy or collection of policies performs poorly because it is 
# often used in tandem with robust search methods for identifying policies that perform well 
# ([Kwakkel and Jaxa-Rozen (2016)](https://www.sciencedirect.com/science/article/pii/S1364815215301092)).

# %%
from emat.analysis.prim import Prim

# %%
of_interest = lhs_large_results['net_benefits']>0

# %%
discovery = Prim(
    m.read_experiment_parameters(design_name='lhs_large'),
    of_interest,
    scope=road_scope,
)

# %%
box1 = discovery.find_box()

# %%
box1.tradeoff_selector()

# %%
box1.inspect(45)

# %%
box1.select(45)

# %%
box1.splom()

# %% [markdown]
# ### CART
#
# Classification and Regression Trees (CART) can also be used for scenario discovery. 
# They partition the explored space (i.e., the scope) into a number of sections, with each partition
# being added in such a way as to maximize the difference between observations on each 
# side of the newly added partition divider, subject to some constraints.

# %%
from emat.workbench.analysis import cart

cart_alg = cart.CART(
    m.read_experiment_parameters(design_name='lhs_large'),
    of_interest,
)
cart_alg.build_tree()

# %%
from emat.util.xmle import Show
Show(cart_alg.show_tree(format='svg'))

# %%
cart_alg.boxes_to_dataframe(include_stats=True)

# %% [markdown]
# ## Robust Search

# %%
from emat import Measure

MAXIMIZE = Measure.MAXIMIZE
MINIMIZE = Measure.MINIMIZE

robustness_functions = [
    Measure(
        'Expected Net Benefit', 
        kind=Measure.INFO, 
        variable_name='net_benefits', 
        function=numpy.mean,
    ),
    
    Measure(
        'Probability of Net Loss', 
        kind=MINIMIZE, 
        variable_name='net_benefits', 
        function=lambda x: numpy.mean(x<0),
    ),

    Measure(
        '95%ile Travel Time', 
        kind=MINIMIZE, 
        variable_name='build_travel_time',
        function=functools.partial(numpy.percentile, q=95),
    ),

    Measure(
        '99%ile Present Cost', 
        kind=Measure.INFO, 
        variable_name='present_cost_expansion', 
        function=functools.partial(numpy.percentile, q=99),
    ),

    Measure(
        'Expected Present Cost', 
        kind=Measure.INFO, 
        variable_name='present_cost_expansion', 
        function=numpy.mean,
    ),

]

# %% [markdown]
# ### Constraints
#
# The robust optimization process solutions can be constrained to only include solutions that 
# satisfy certain constraints.  These constraints can be based on the policy lever parameters
# that are contained in the core model, the aggregate performance measures identified in 
# the list of robustness functions, or some combination of levers and aggregate measures.

# %%
from emat import Constraint

# %% [markdown]
# The common use case for constraints in robust optimation is imposing requirements
# on solution outcomes. For example, we may want to limit our robust search only
# to solutions where the expected present cost of the capacity expansion is less
# than some particular value (in our example here, 4000).  

# %%
constraint_1 = Constraint(
    "Maximum Log Expected Present Cost", 
    outcome_names="Expected Present Cost",
    function=Constraint.must_be_less_than(4000),
)

# %% [markdown]
# Our second constraint is based exclusively on an input: the capacity expansion
# must be at least 10.  We could also achieve this kind of constraint by changing
# the exploratory scope, but we don't necessarily want to change the scope to 
# conduct a single robust optimization analysis with a constraint on a policy lever.

# %%
constraint_2 = Constraint(
    "Minimum Capacity Expansion", 
    parameter_names="expand_capacity",
    function=Constraint.must_be_greater_than(10),
)

# %% [markdown]
# It is also possible to impose constraints based on a combination of inputs and outputs.
# For example, suppose that the total funds available for pay-as-you-go financing are
# only 1500.  We may thus want to restrict the robust search to only solutions that
# are almost certainly within the available funds at 99% confidence (a model output) but only 
# if the Paygo financing option is used (a model input).  This kind of constraint can
# be created by giving both `parameter_names` and `outcomes_names`, and writing a constraint
# function that takes two arguments.

# %%
constraint_3 = Constraint(
    "Maximum Paygo", 
    parameter_names='debt_type',
    outcome_names='99%ile Present Cost',
    function=lambda i,j: max(0, j-1500) if i=='Paygo' else 0,
)

# %%
robust_results = m.robust_optimize(
    robustness_functions,  
    scenarios=200, 
    nfe=2500, 
    constraints=[
        constraint_1,
        constraint_2,
        constraint_3,
    ],
    #evaluator=get_client(),
    cache_file="./cache_road_test_robust_opt"
)

# %% [markdown]
# The robust_results include all the non-dominated solutions which satisfy the contraints that are found.
# Note that robust optimization is generally a hard problem, and the algorithm may need to run for
# a very large number of iterations in order to arrive at a good set of robust solutions.

# %%
robust_results.par_coords()

# %% [markdown]
# ## Creating a Meta-Model

# %% [raw] {"raw_mimetype": "text/restructuredtext"}
# Creating a meta-model requires an existing model, plus a set of 
# experiments (including inputs and outputs) run with that model
# to support the estimation of the meta-model.  After having completed
# sufficient initial runs of the core model, instantiating a meta-model
# is as simple as calling a `create_metamodel_*` method on the core
# model, either giving a design of experiments stored in the database
# with :meth:`create_metamodel_from_design <emat.AbstractCoreModel.create_metamodel_from_design>`
# or passing the experimental results directly with
# :meth:`create_metamodel_from_data <emat.AbstractCoreModel.create_metamodel_from_data>`.

# %%
mm = m.create_metamodel_from_design('lhs')
mm

# %% [markdown]
# To demonstrate the performance of the meta-model, we can create an
# alternate design of experiments.  Note that to get different random values,
# we set the `random_seed` argument to something other than the default value.

# %%
design2 = design_experiments(
    scope=road_scope, 
    db=emat_db, 
    n_samples_per_factor=10, 
    sampler='lhs', 
    random_seed=2,
)

# %%
design2_results = mm.run_experiments(design2)
design2_results.head()

# %%
mm.cross_val_scores()

# %% [markdown]
# ### Compare Core vs Meta Model Results
#
# We can generate a variety of plots to compare the distribution of meta-model outcomes
# on the new design against the original model's results.

# %%
from emat.analysis import contrast_experiments
contrast_experiments(road_scope, lhs_results, design2_results)

# %%
