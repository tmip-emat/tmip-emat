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

# %% [markdown]
# # Database Walkthrough

# %%
import os
import numpy as np
import pandas as pd
import seaborn; seaborn.set_theme()
import plotly.io; plotly.io.templates.default = "seaborn"
import emat
import yaml
from emat.util.show_dir import show_dir
from emat.analysis import display_experiments
emat.versions()

# %% [markdown]
# For this walkthrough of database features, we'll work in a temporary directory. 
# (In real projects you'll likely want to save your data somewhere less ephemeral,
# so don't just copy this tempfile code into your work.)

# %%
import tempfile
tempdir = tempfile.TemporaryDirectory()
os.chdir(tempdir.name)

# %% [markdown]
# We begin our example by populating a database with some experimental data, by creating and
# running a single design of experiments for the Road Test model.

# %%
import emat.examples
scope, db, model = emat.examples.road_test()
design = model.design_experiments()
model.run_experiments(design);

# %% [markdown]
# ## Single-Design Datasets

# %% [markdown]
# ### Writing Out Raw Data
#
# When the database has only a single design of experiments, or if we
# don't care about any differentiation between multiple designs that we
# may have created and ran, we can dump the entire set of model runs,
# including uncertainties, policy levers, and performance measures, all
# consolidated into a single pandas DataFrame using the 
# `read_experiment_all` function.  The constants even appear in this DataFrame
# too, for good measure.

# %%
df = db.read_experiment_all(scope.name)
df

# %% [markdown]
# Exporting this data is simply a matter of using the usual pandas 
# methods to save the dataframe to a format of your choosing.  We'll
# save our data into a gzipped CSV file, which is somewhat compressed
# (we're not monsters here) but still widely compatible for a variety of uses.

# %%
df.to_csv("road_test_1.csv.gz")

# %% [markdown]
# This table contains most of the information we want to export from
# our database, but not everything.  We also probably want to have access
# to all of the information in the exploratory scope as well.  Our example
# generator gives us a `Scope` reference directly, but if we didn't have that
# we can still extract it from the database, using the `read_scope` method.

# %%
s = db.read_scope()
s

# %%
s.dump(filename="road_test_scope.yaml")

# %%
show_dir('.')

# %% [markdown]
# ### Reading In Raw Data
#
# Now, we're ready to begin anew, constructing a fresh database from scratch,
# using only the raw formatted files.
#
# First, let's load our scope from the yaml file, and initialize a clean database
# using that scope.

# %%
s2 = emat.Scope("road_test_scope.yaml")

# %%
db2 = emat.SQLiteDB("road_test_2.sqldb")

# %%
db2.store_scope(s2)

# %% [markdown]
# Just as we used pandas to save out our consolidated DataFrame of experimental results,
# we can use it to read in a consolidated table of experiments.

# %%
df2 = pd.read_csv("road_test_1.csv.gz", index_col='experiment')
df2

# %% [markdown]
# Writing experiments to a database is not quite as simple as reading them.  There
# is a parallel `write_experiment_all` method for the `Database` class, but to use 
# it we need to provide not only the DataFrame of actual results, but also a name for
# the design of experiments we are writing (all experiments exist within designs) and
# the source of the performance measure results (zero means actual results from a 
# core model run, and non-zero values are ID numbers for metamodels). This allows many
# different possible sets of performance measures to be stored for the same set
# of input parameters.

# %%
db2.write_experiment_all(
    scope_name=s2.name, 
    design_name='general',
    source=0,
    xlm_df=df2,
)

# %%
display_experiments(s2, 'general', db=db2, rows=['time_savings'])

# %% [markdown]
# ## Multiple-Design Datasets
#
# The EMAT database is not limited to storing a single design of experiments.  Multiple designs 
# can be stored for the same scope.  We'll add a set of univariate sensitivity test to our
# database, and a "ref" design that contains a single experiment with all inputs set to their
# default values.

# %%
design_uni = model.design_experiments(sampler='uni')
model.run_experiments(design_uni)
model.run_reference_experiment();

# %% [markdown]
# We now have three designs stored in our database. We can confirm this
# by reading out the set of design names.

# %%
db.read_design_names(s.name)

# %% [markdown]
# The design names we se here are the default names given when designs are created with each of the given samplers.  When creating new designs, we can override the default names with other names of our choice using the `design_name` argument.  The names can be any string not already in use.

# %%
design_b = model.design_experiments(sampler='lhs', design_name='bruce')
db.read_design_names(s.name)

# %% [markdown]
# If you try to re-use a name you'll get an error, as having multiple designs with the same name does not allow you to make it clear which design you are referring to.

# %%
try:
    model.design_experiments(sampler='lhs', design_name='bruce')
except ValueError as err:
    print(err)

# %% [markdown]
# As noted above, the design name, which can be any string, is separate from the sampler method. A default design name based on the name of the sampler method is used if no design name is given.  The selected sampler must be one available in EMAT, as the sampler defines a particular logic about how to generate the design.

# %%
try:
    model.design_experiments(sampler='uni')
except ValueError as err:
    print(err)

# %% [markdown]
# Note that there 
# can be some experiments that are in more than one design.  This is
# not merely duplicating the experiment and results, but actually 
# assigning the same experiment to both designs.  We can see this
# for the 'uni' and 'ref' designs -- both contain the all-default 
# parameters experiment, and when we read these designs out of the 
# database, the same experiment number is reported out in both 
# designs.

# %%
db.read_experiment_all(scope.name, design_name='uni').head()

# %%
db.read_experiment_all(scope.name, design_name='ref')

# %% [markdown]
# One "gotcha" to be wary of is unintentionally replicating experiments.  By default, the `random_seed` for randomly generated experiemnts is set to 0 for reproducibility.  This means that, for example, the 'bruce' design is actually the same as the original 'lhs' design:

# %%
db.read_experiment_all(scope.name, design_name='lhs').equals(
    db.read_experiment_all(scope.name, design_name='bruce')
)

# %% [markdown]
# If we want a new set of random experiments with the same sampler and other parameters, we'll need to provide a different `random_seed`.

# %%
design_b = model.design_experiments(sampler='lhs', design_name='new_bruce', random_seed=42)
db.read_experiment_all(scope.name, design_name='lhs').equals(design_b)

# %% [markdown]
# ### Writing Out Raw Data
#
# We can read a single dataframe containing all the experiments associated with
# this scope by omitting the `design_name` argument, just as if there was only
# one design.

# %%
df = db.read_experiment_all(scope.name)
df

# %% [markdown]
# This dataframe is different than the one we saw earlier with the same command, as we have since added a few more experiments to the database in a few different designs.  If we don't give a `design_name` argument, we'll retrieve every (unique) experiment from every design currently stored in the database.  

# %%
df.to_csv("road_test_2.csv.gz")

# %% [markdown]
# If we want to be able to reconstruct the various designs of experiments later, 
# we'll also need to write out instructions for that.  The `read_all_experiment_ids`
# method can give us a dictionary of all the relevant information.

# %%
design_experiments = db.read_all_experiment_ids(scope.name, design_name='*',grouped=True)
design_experiments

# %% [markdown]
# We can write this dictionary to a file in 'yaml' format.

# %%
with open("road_test_design_experiments.yaml", 'wt') as f:
    yaml.dump(design_experiments, f)

# %% [markdown]
# ### Reading In Raw Data

# %% [markdown]
# To construct a new emat Database with multiple designs of experients,...

# %%
db3 = emat.SQLiteDB("road_test_3.sqldb")
db3.store_scope(s2)

# %%
df3 = pd.read_csv("road_test_2.csv.gz", index_col='experiment')
df3

# %%
with open("road_test_design_experiments.yaml", 'rt') as f:
    design_experiments2 = yaml.safe_load(f)
design_experiments2

# %%
db3.write_experiment_all(
    scope_name=s2.name, 
    design_name=design_experiments2,
    source=0,
    xlm_df=df3,
)

# %%
db3.read_design_names(s.name)

# %%
db3.read_all_experiment_ids(scope.name, design_name='*',grouped=True)

# %%
db3.read_experiment_all(scope.name, design_name='uni').head()

# %% [markdown]
# ## Re-running Experiments

# %% [markdown]
# This section provides a short walkthrough of how to handle mistakes 
# in an EMAT database.  By "mistakes" we are referring to incorrect
# values that have been written into the database by accident, generally 
# arising from core model runs that were misconfigured or suffered 
# non-fatal errors that caused the results to be invalid.
#
# One approach to handling such problems is to simply start over with a
# brand new clean database file.  However, this may be inconvenient if
# the database already includes a number of valid results, especially if
# those valid results were expensive to generate.  It may also be desirable
# to keep prior invalid results on hand, so as to easily recognized when
# errors recur.

# %% [markdown]
# We begin this example by populating our database with some more experimental data, by creating and
# running a single design of experiments for the Road Test model, except these experiments will be
# created with a misconfigured model (lane_width = 11, it should be 10), so the results will be bad.
# (In general, you probably won't intentionally create corrupt data, but we're doing so here for 
# expository purposes, so we'll give this design a name of 'oops' so we can readily recall what we've done.)

# %%
model.lane_width = 10.3
oops = model.design_experiments(design_name='oops', random_seed=12345)
model.run_experiments(oops);

# %% [markdown]
# We can review a dataframe of results as before, using the `read_experiment_all`
# method. This time we will add `with_run_ids=True`, which will add an extra
# column to the index, showing a universally unique id attached to each row
# of results.

# %%
oops_result1 = db.read_experiment_all(scope.name, 'oops', with_run_ids=True)
oops_result1.head()

# %%
display_experiments(scope, oops_result1, rows=['time_savings'])

# %% [markdown]
# Some of these results are obviously problematic.  Increasing capacity cannot possibly
# result in a negative travel time savings. (Braess paradox doesn't apply here because 
# it's just one link, not a network.)  So those negative values are clearly wrong.  We 
# can fix the model so they won't be wrong, but by default the `run_experiments` method
# won't actually re-run models when the results are already available in the database.
# To solve this conundrum, we can mark the incorrect results as invalid, using a query
# to pull out the rows that can be flagged as wrong.

# %%
db.invalidate_experiment_runs(
    queries=['time_savings < 0']
)

# %% [markdown]
# The `[73]` returned here indicates that 73 sets of results were invalidated by this command.
# The invalidation command actually sets a "valid" flag in the database to False for these
# experiment runs, so that a persistant record that they are bad is stored in the database.
# Now we can fix our model, and then use the `run_experiments` method to get new model runs for
# the invalidated results.

# %%
model.lane_width = 10
oops_result2 = model.run_experiments(oops)

# %%
display_experiments(scope, 'oops', db=db, rows=['time_savings'])

# %% [markdown]
# The re-run fixed the negative values, although it left in place the other 
# experimental runs in the database. By the way we constructed this example, 
# we know those are wrong too, and it's evident in the apparent discontinuity
# in the input flow graph, which we can zoom in on.

# %%
ax = oops_result2.plot.scatter(x='input_flow', y='time_savings', color='r')
ax.plot([109,135], [0,35], '--',color='y');

# %% [markdown]
# Those original results are bad too, and we want to invalidate them as well.
# In addition to giving conditional queries to the `invalidate_experiment_runs`
# method, we can also give a dataframe of results that have run ids attached, 
# and those unique ids will be used to to find and invalidate results in the 
# database.  Here, we pass in the dataframe of all the results, which contains
# all 110 runs, but only 37 runs are newly invalidated (77 were invalidated 
# previously).

# %%
db.invalidate_experiment_runs(
    oops_result1
)

# %% [markdown]
# Now when we run the experiments again, those 37 experiments are re-run.

# %%
oops_result3 = model.run_experiments(oops)

# %%
display_experiments(scope, 'lhs', db=db, rows=['time_savings'])

# %% [markdown]
# ### Writing Out All Runs
#
# By default, the `read_experiment_all` method returns the most recent valid set of 
# performance measures for each experiment, but we can override this behavior to
# ask for `'all'` run results, or all `'valid'` or `'invalid'` results, by setting the 
# `runs` argument to those literal values.  This allows us to easily
# write out data files containing all the results stored in the database.

# %%
db.read_experiment_all(scope.name, with_run_ids=True, runs='all')

# %% [markdown]
# In the resulting dataframe above, we can see that we have retrieved two different runs for some of the experiments.
# Only one of them is valid for each. If we want to get all the stored runs and also mark the valid and invalid runs, we can read them 
# seperately and attach a tag to the two dataframes.

# %%
runs_1 = db.read_experiment_all(scope.name, with_run_ids=True, runs='valid')
runs_1['is_valid'] = True
runs_0 = db.read_experiment_all(scope.name, with_run_ids=True, runs='invalid', only_with_measures=True)
runs_0['is_valid'] = False
all_runs = pd.concat([runs_1, runs_0])
all_runs.sort_index()

# %% [markdown]
# These mechanisms can be use to write out results of multiple runs, 
# and to repopulate a database with both valid
# and invalid raw results. This can be done multiple ways (seperate
# files, one combined file, keeping track of invalidation queries, etc.).
# The particular implementations of each are left as an exercise for
# the reader.
