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
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import emat
emat.versions()

# %% [raw] {"raw_mimetype": "text/restructuredtext"}
# .. _methodology-cart:

# %% [markdown]
# # CART
#
# Classification and Regression Trees (CART) can be used for scenario discovery. 
# They partition the explored space (i.e., the scope) into a number of sections, with each partition
# being added in such a way as to maximize the difference between observations on each 
# side of the newly added partition divider, subject to some constraints.

# %% [markdown]
# ## The Mechanics of using CART

# %% [markdown]
# In order to use CART for scenario discovery, the analyst must
# first conduct a set of experiments.  This includes having both
# the inputs and outputs of the experiments (i.e., you've already
# run the model or meta-model).

# %%
import emat.examples
scope, db, model = emat.examples.road_test()
designed = model.design_experiments(n_samples=5000, sampler='mc', random_seed=42)
results = model.run_experiments(designed, db=False)

# %% [markdown]
# In order to use CART for scenario discovery, the analyst must
# also identify what constitutes a case that is "of interest".
# This is essentially generating a True/False label for every 
# case, using some combination of values of the output performance 
# measures as well as (possibly) the values of the inputs.
# Some examples of possible definitions of "of interest" might
# include:
#
# - Cases where total predicted VMT (a performance measure) is below some threshold.
# - Cases where transit farebox revenue (a performance measure) is above some threshold.
# - Cases where transit farebox revenue (a performance measure) is above above 50% of
#   budgeted transit operating cost (a policy lever).
# - Cases where the average speed of tolled lanes (a performance measure) is less 
#   than free-flow speed but greater than 85% of free-flow speed (i.e., bounded both
#   from above and from below).
# - Cases that meet all of the above criteria simultaneously.
#
# The salient features of a definition for "of interest" is that
# (a) it can be calculated for each case if given the set 
# of inputs and outputs, and (b) that the result is a True or False value.
#
# For this example, we will define "of interest" as cases from the 
# Road Test example that have positive net benefits.

# %%
of_interest = results['net_benefits']>0

# %% [markdown]
# Having defined the cases of interest, to use CART we pass the
# explanatory data (i.e., the inputs) and the 'of_interest' variable
# to the `CART` object, and then we can invoke the `tree_chooser` method.

# %%
from emat.analysis import CART

cart = CART(
    model.read_experiment_parameters(design_name='mc'),
    of_interest,
    scope=scope,
)

# %%
chooser = cart.tree_chooser()
chooser

# %% [markdown]
# The CART algorithm develops a tree that seeks to make the "best" split
# at each decision point, generating two datasets that are subsets of the original
# data and which provides the best (weighted) improvement in the target criterion,
# which can either be gini impurity or information gain (i.e., entropy reduction).
#
# The `tree_chooser` method returns an interactive widget that allows an analyst
# to manipulate selected hyperparameters for the decision tree used
# by CART.  The analyst can set the branch splitting criteria
# (gini impurity or information gain / entropy reduction), the maximum tree depth, and
# the minimum fraction of observations in any leaf node.
#
# The display shows the decision tree created by CART, including the branching 
# rule at each step, and a short summary of the data in each branch.  The coloration
# of each tree node highlights the progress, with increasing saturation representing
# improvements in the branching criterion (gini or entropy) and the hue indicating 
# the dominant result in each node.  In the example above, the "of interest" cases 
# are most densely collected in the blue nodes.
#
# It is also possible to review the collection leaf nodes in a tabular display, 
# by accessing the `boxes_to_dataframe` method, which reports out the total dimensional 
# restrictions for each box.  Here, we provide a `True` argument to include box statistics as well.

# %%
cart.boxes_to_dataframe(True)

# %% [markdown]
# This table shows various leaf node "boxes" as well as the trade-offs 
# between coverage and density in each.
#
# - **Coverage** is percentage of the cases of interest that are in each box
#   (i.e., number of cases of interest in the box divided by total number of 
#   cases of interest).
# - **Density** is the share of cases in each box that are case of interest
#   (i.e., number of cases of interest in the box divided by the total 
#   number of cases in the box). 
#
# For the statistically minded, this tradeoff can also be interpreted as
# the tradeoff between Type I (false positive) and Type II (false negative)
# error.  High coverage minimizes the false negatives, while high density
# minimizes false positives.
#
# As we can for PRIM, we can make a selection of a particular box, and then
# generate a number of visualizations around that selection.

# %%
box = cart.select(6)
box

# %% [markdown]
# To help visualize these restricted dimensions better, we can 
# generate a plot of the resulting box,
# overlaid on a 'pairs' scatter plot matrix (`splom`) of the various restricted 
# dimensions.
#
# In the figure below, each of the three restricted dimensions represents
# both a row and a column of figures.  Each of the off-diagonal charts show 
# bi-dimensional distribution of the data across two of the actively
# restricted dimensions.  These charts are overlaid with a green rectangle
# denoting the selected box.  The on-diagonal charts show the relative
# distribution of cases that are and are not of interest (unconditional
# on the selected box).

# %%
box.splom()

# %% [markdown]
# Depending on the number of experiments in the data and the number 
# and distribution of the cases of interest, it may be clearer to
# view these figures as a heat map matrix (`hmm`) instead of a splom.

# %%
box.hmm()

# %% [raw] {"raw_mimetype": "text/restructuredtext"}
# .. include:: cart-api.irst
