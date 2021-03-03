# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import emat
emat.versions()

# %% [markdown]
# # Interactive Explorer
#
# TMIP-EMAT includes an interactive explorer, inspired by a 
# [similar tool](https://htmlpreview.github.io/?https://github.com/VisionEval/VisionEval/blob/master/sources/VEScenarioViewer/verpat.html) 
# provided with the [VisionEval](https://visioneval.org) package.
# To demonstrate the interactive explorer, we will use the Road Test example model.
# First, we need to develop and run a design of experiments to have some
# data to explore.  We'll run 5,000 experiments to get a good size sample of 
# data points to visualize.

# %%
import emat.examples
scope, db, model = emat.examples.road_test()
design = model.design_experiments(n_samples=5000)
results = model.run_experiments(design)

# %% [markdown]
# The interactive explorer class can be imported from the `emat.analysis` package.
# To use it, we create an `Explore` instance, giving a scope and a set of 
# experimental results.

# %%
from emat.analysis import Explore
xp = Explore(scope=scope, data=results)

# %% [markdown]
# <span style="color:red; font-size:80%;">
# Note: The interactivity of the figures displayed directly on the
# TMIP-EMAT website is not enabled.  This interactivity requires
# a live running Python kernel to support the calculations to
# update the figures.
# You can try to open a live interactive version in
# <a href="https://mybinder.org/v2/gh/tmip-emat/tmip-emat/ab5ea96371751d7d3ddd95dd5599b384d781f92e?filepath=docs%2Fsource%2Femat.analysis%2Finteractive-explorer.ipynb">Binder</a>
# but it may take a bit of time to load.</span>

# %% [markdown]
# ## Single Dimension Figures
#
# To build a complete interactive workspace similar to that provided by VisionEval, we
# can use the `complete` method of the `Explore` instance we created above. This will
# create a set of histograms illustrating the data in the results computed above. There
# is one histogram for each policy lever, exogenous uncertainty, and performance measure.
#
# Each histogram is accompanied by a range slider or toggle buttons, depending on the 
# data type (boolean and categorical data get toggle buttons).  These controls can 
# be used interactively to select and highlight only a subset of the experiments in
# the complete data set.  By manipulating these controls, users can explore the 
# interaction across various inputs and outputs.

# %%
xp.complete()

# %% [markdown]
# In addition to manipulating the controls interactively, they can also be
# set programatically from Python code.  For example, to clear the selections,
# use the `clear` method.

# %%
xp.clear()

# %% [markdown]
# To set the lower or upper bound on a range slider, use `set_lower_bound` or `set_upper_bound`, 
# each of which takes the name of the scope parmeter or measure to set, and the bound.

# %%
xp.set_lower_bound('net_benefits', 0)

# %% [markdown]
# To set toggle buttons, use `remove_from_allowed_set` or `add_to_allowed_set`, again giving as 
# arguments the name of the scope parmeter or measure, and the value to remove or add.

# %%
xp.remove_from_allowed_set('debt_type', 'Rev Bond')

# %% [markdown]
# In addition to the histogram views, TMIP-EMAT can also generate kernel density plots,
# by specifying the `style` keyword argument as `'kde'` when using the `viewers` or 
# `selectors` methods.
# These plots differ from the histograms in two ways: the discrete bars are replaced
# by smoothed curves, and the selected (orange) area is renormalized to the same scale 
# as the overall (blue) curves, similar to a probability density function (although the
# figures generated are not actually PDF's, as they incorporate a selection of experiments
# that vary based on policy levers, which are not probabilistic distributions).

# %%
xp.viewers(['time_savings', 'net_benefits'], style='kde')

# %%
xp.viewers(['input_flow', 'net_benefits', 'value_of_time'], style='kde')

# %% [markdown]
# ## Two Dimension Figures

# %% [markdown]
# The `Explore` object can also create an interactive two-dimensional scatter plot,
# using the `two_way` method. This method allows the user to specify the variables
# for both the `x` and `y` axis, and either can be any policy lever, exogenous 
# uncertainty, or performance measure.  These dimensions can be changed interactively
# later as well.  The resulting scatter plot is linked to the same selection of
# experiments in the interactive one-dimensional figures shown above, and by default
# the same experiments are highlighted in blue and orange in all of these related
# figures.

# %%
xp.two_way(x='expand_capacity', y='time_savings')

# %% [markdown]
# ## Using PRIM with the Interactive Explorer
#
# The PRIM tools are available directly within the interactive explorer. Simply 
# set a target as shown.

# %%
prim = xp.prim(target=xp.data['net_benefits'] > 0)

# %% [markdown]
# The tradeoff selector is directly integrated into the explorer.  In addition
# to the information visible by hovering over any point in the tradeoff selector
# figure, clicking on that point will set all of the interactive constraints 
# to the bounds given by that particular point.

# %%
prim.tradeoff_selector()
