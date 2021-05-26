# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [raw] raw_mimetype="text/restructuredtext"
# .. py:currentmodule:: emat

# %% [markdown]
# # GBNRTC Example Model

# %%
import emat
emat.require_version('0.5.1')

# %%
import pandas as pd

# %% [raw] raw_mimetype="text/restructuredtext"
# The model scope is defined in a YAML file.  For this GBNRTC example, the scope file is named 
# :ref:`gbnrtc_scope.yaml <gbnrtc_scope_file>`.

# %%
scope = emat.Scope('gbnrtc_scope.yaml')

# %%
db = emat.SQLiteDB()

# %%
scope.store_scope(db)

# %% [markdown]
# The basic operation of the GBNRTC model can be controlled by EMAT through a custom developed 
# class, which defines the input and output "hooks" that are consistent with the defined 
# scope file.  The `GBNRTCModel` class is able to call to TransCAD, setup the input parameters
# (exogenous uncertainties, policy levers, and constants defined in the scope), exceute the 
# model, and retrieve the performance measure results.  

# %%
from emat.model import GBNRTCModel

# %%
g = GBNRTCModel(
    configuration='gbnrtc_model_config.yaml',
    scope=scope,
    db=db,
)
g

# %% [markdown]
# The GBNRTC model takes a couple of hours for each run, and runs in TransCAD, which 
# is a proprietary software package that is not included with the EMAT distribution.
# However, for demonstration purposes, the definition and results of a particular set 
# of experiments is included in the file `buffalo.csv`.  We can use 
# the `write_experiment_all` method to pre-load these results into the database.

# %%
lhs = pd.read_csv('buffalo.csv')

# %%
lhs.info()

# %%
db.write_experiment_all(
    'GBNRTC', 
    'lhs', 
    emat.SOURCE_IS_CORE_MODEL, 
    lhs,
)

# %% [raw] raw_mimetype="text/restructuredtext"
# We can check that the pre-loaded data includes the results of the experiments
# by checking the number of rows in the :meth:`read_experiments <model.core_model.AbstractCoreModel.read_experiments>` 
# DataFrame, both in total and when only loading pending experiments (those without stored performance
# meaures):

# %%
len(g.read_experiments('lhs'))

# %%
len(g.read_experiments('lhs', only_pending=True))

# %% [markdown]
# The example data contains a large variety of output performance measures, as 
# TransCAD models can potentially output a lot of data.

# %%
g.scope.get_measure_names()

# %% [markdown]
# The high level scope
# definition is designed to capture all of this data for later analysis, but
# in this demonstration we will only evaluate a few of these performance measures.
# In part, this is because creating meta-models for each performance measure is 
# relatively inexpensive (computationally speaking) but not free -- it can take 
# a few seconds to create the meta-model and it is not needed here if we are not 
# interested in all these results for this analysis.
#
# Creating a meta-model for analysis of an existing model with a completed 
# design of experiments can be done using the `create_metamodel_from_design` 
# method. To create a meta-model on a more limited scope, we can use the 
# `include_measures` argument to list out a subset of measures that will be
# included in this metamodel.

# %%
mm = g.create_metamodel_from_design(
    'lhs',
    include_measures=[
        'Region-wide VMT', 
        'AM Trip Time (minutes)',
        'Downtown to Airport Travel Time',
        'Total Transit Boardings',
        'Peak Transit Share', 
        'Peak NonMotorized Share',
        'Kensington Daily VMT',
        'Corridor 190 Daily VMT',
        'Corridor 33_west Daily VMT',
        'Corridor I90_south Daily VMT',
    ],
    suppress_converge_warnings=True,
)
mm

# %% [markdown]
# You might notice that the class of the meta-model is no longer a `GBNRTCModel`
# but instead now it is a `PythonCoreModel`.  This is because at its heart, the
# meta-model is a Python function that wraps the gaussian process regression that
# has been fit to the available experimental data.  Also, although the scope still
# has 46 measures, only 10 are active in the actual meta-model:

# %%
mm.function

# %%
callable(mm.function)

# %% [raw] raw_mimetype="text/restructuredtext"
# To access this regression directly, we can use the :meth:`regression <emat.MetaModel.regression>` attribute 
# of the :class:`MetaModel <emat.MetaModel>`.

# %%
mm.function.regression

# %%
mm.function.regression.lr.r2

# %%
mm.function.regression.lr.coefficients_summary()

# %% [raw] raw_mimetype="text/restructuredtext"
# We can also generate cross-validation scores for the :class:`MetaModel <emat.MetaModel>` to verify that the
# meta-model is performing well.

# %%
mm.function.cross_val_scores()

# %% [markdown]
# To use the metamodel for exploratory analysis, we can design and run a large
# number of experiments.

# %%
design = mm.design_experiments(n_samples=10000, sampler='lhs')

# %% [markdown]
# The meta-model evaluates pretty quickly.

# %%
result = mm.run_experiments(design)

# %% [markdown]
# If we inspect the results, we see that among the performance measures, only the 
# active measures have non-null computed values:

# %%
result.info()

# %% [markdown]
# The results of these meta-model experiments can be used for visualization and
# other exploratory modeling applications.

# %%
from emat.viz import scatter_graphs
scatter_graphs('Downtown to Airport Travel Time', result, scope=mm.scope, render='png')
