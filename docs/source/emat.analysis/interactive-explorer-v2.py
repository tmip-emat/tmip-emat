# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
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
# TMIP-EMAT includes an interactive visualizer, inspired by a 
# [similar tool](https://htmlpreview.github.io/?https://github.com/VisionEval/VisionEval/blob/master/sources/VEScenarioViewer/verpat.html) 
# provided with the [VisionEval](https://visioneval.org) package.
# To demonstrate the second generation interactive visualizer, we will use the Road Test example model.
# First, we need to develop and run a design of experiments to have some
# data to explore.  We'll run 5,000 experiments to get a good size sample of 
# data points to visualize.

# %%
import emat.examples
scope, db, model = emat.examples.road_test()
design = model.design_experiments(n_samples=5000)
results = model.run_experiments(design)

# %% [markdown]
# One feature of the visualizer is the ability to display not only a number of results,
# but also to contrast those results against a given "reference" model that represents
# a more traditional single-point forecast of inputs and results.  We'll prepare a
# reference point here using the input parameter defaults (as defined in the scope),
# and the `io_experiment` method of our model, which returns both inputs and outputs 
# in a single dictionary, suitable for use as the reference point marker in our
# visualizations.

# %%
refpoint = model.io_experiment(scope.get_parameter_defaults())

# %% [markdown]
# The interactive visualizer class can be imported from the `emat.analysis` package.
# To use it, we create an `Visualizer` instance, giving a scope and a set of 
# experimental results, as well as the reference point.

# %%
from emat.analysis import Visualizer

# %%
viz = Visualizer(scope=scope, data=results, reference_point=refpoint)

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
# can use the `complete` method of the `Visualizer` instance we created above. This will
# create a set of histograms illustrating the data in the results computed above. There
# is one histogram for each policy lever, exogenous uncertainty, and performance measure.
#
# Each histogram is accompanied by a range slider or toggle buttons, depending on the 
# data type (boolean and categorical data get toggle buttons).  These controls can 
# be used interactively to select and highlight only a subset of the experiments in
# the complete data set.  By manipulating these controls, users can explore the 
# interaction across various inputs and outputs.

# %%
viz.complete()

# %% [markdown]
# In addition to manipulating the controls interactively, they can also be
# set programatically from Python code.  For example, to clear the selections,
# use the `clear_box` method, which by default clears the settings on the active
# box selection; give a name to clear the settings from a different box selection.

# %%
viz.clear_box()

# %% [markdown]
# To create a new selection from a box, we can use `new_box` and define the selection box based on the values in the scope.  The `new_box` method takes all the same keyword arguments as the typical `Box` contructor.

# %%
viz.new_box('Profitable', lower_bounds={'net_benefits':0})

# %% [markdown]
# To manipulate an existing box selection, we can access the Box object, 
# manipulate it (e.g. by using `remove_from_allowed_set` or `add_to_allowed_set`), 
# and write it back into the Visualizer.

# %%
box = viz['Profitable']
box.remove_from_allowed_set('debt_type', 'Rev Bond')
viz['Profitable'] = box

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
viz.two_way(x='expand_capacity', y='time_savings')

# %% [markdown]
# ## Using PRIM with the Interactive Explorer
#
# The PRIM tools are available directly within the interactive explorer. Simply 
# set a target as shown.

# %%
prim = viz.prim(target="Profitable")

# %% [markdown]
# The tradeoff selector is directly integrated into the explorer.  In addition
# to the information visible by hovering over any point in the tradeoff selector
# figure, clicking on that point will set all of the interactive constraints 
# to the bounds given by that particular point.

# %%
prim.tradeoff_selector()