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

# %%
import emat
emat.versions()

# %% [markdown]
# # Interactive Explorer
#
# TMIP-EMAT includes an interactive visualizer, inspired by a 
# [similar tool](https://htmlpreview.github.io/?https://github.com/VisionEval/VisionEval/blob/master/sources/VEScenarioViewer/verpat.html) 
# provided with the [VisionEval](https://visioneval.org) package.
# To demonstrate the interactive visualizer, we will use the Road Test example model.
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
# reference point here using the `run_reference_experiment` method of the `CoreModel`
# class, which reads the input parameter defaults (as defined in the scope),
# and returns both inputs and outputs in a DataFrame (essentially, an experimental
# design with only a single experiment), suitable for use as the reference point marker in our
# visualizations.

# %%
refpoint = model.run_reference_experiment()

# %% [markdown]
# The interactive visualizer class can be imported from the `emat.analysis` package.
# To use it, we create an `Visualizer` instance, giving a scope and a set of 
# experimental results, as well as the reference point.

# %%
from emat.analysis import Visualizer

# %%
viz = Visualizer(scope=scope, data=results, reference_point=refpoint)

# %% [markdown]
# ## Single Dimension Figures
#
# To build a complete interactive workspace similar to that provided by VisionEval, we
# can use the `complete` method of the `Visualizer` instance we created above. This will
# create a set of histograms illustrating the data in the results computed above. There
# is one histogram for each policy lever, exogenous uncertainty, and performance measure.
#
# A range of data in each histogram can be selected by dragging horizonatally across the 
# figure. For continuous parameters (i.e. float or integer valued parameters) you can 
# select a single contiguous range by dragging across that range, and deselect by double 
# clicking on the figure (or by selecting the entire possible range).  For discrete 
# parameters (i.e. boolean or categorical parameters, identifiable by the larger gaps
# between the bars in the figure) dragging across the center of any bar toggles whether
# that bar is selected or not.  This allows non-contiguous selections in categories that
# have 3 or more possible values.  Like the other figures, any selection can be cleared 
# by double-clicking.
#
# Selections can be made simultaneously over any combination of uncertainties, policy levers,
# and performance measures.  The combination of controls offered can 
# be used interactively to select and highlight only a subset of the experiments in
# the complete data set.  By manipulating these controls, users can explore the 
# interaction across various inputs and outputs.

# %% [markdown]
# ![Selecting from histograms](interactive-gifs/select-from-histograms-.gif)

# %%
viz.complete()

# %% [markdown]
# It is also possible to display just a small subset of the figures of this interactive viewer.
# This could be convenient, for example, if there are a very large number of performance measures.

# %%
viz.selectors(['input_flow', 'expand_capacity', 'net_benefits'])

# %% [raw] raw_mimetype="text/restructuredtext"
# In addition to manipulating the controls interactively, they can also be
# set programatically from Python code.  To do so, we can define a new :class:`emat.Box`
# that declares lower and/or upper bounds for any continuous dimensions,
# as well as the set of allowed (included) value for any discrete dimensions,
# and then add that new box to this visualizer using the :meth:`Visualizer.add_box` command.

# %%
box = emat.Box("Passable", scope=scope)
box.set_upper_bound('cost_of_capacity_expansion', 400)
box.set_lower_bound('time_savings', 5)
box.remove_from_allowed_set('debt_type', 'GO Bond')
viz.add_box(box)

# %% [markdown]
# Alternatively, a new box can be created and added to the Visualier
# with a single :meth:`Visualizer.new_box` command, which
# passes most keyword arguments through to the :class:`emat.Box` constuctor.

# %%
viz.new_box('Profitable', lower_bounds={'net_benefits':0});

# %% [markdown]
# Each of these new boxes is added to the `Visualizer` seperately. You can
# switch between different active boxes using the dropdown selector at the top 
# of the `complete` interface -- this same selector is available within the
# smaller `status` widget:

# %%
viz.status()

# %% [markdown]
# You can also programatically find and change the active box from Python:

# %%
viz.active_selection_name()

# %%
viz.set_active_selection_name("Passable")
viz.active_selection_name()

# %% [markdown]
# When interactively changing bounds by dragging on figures, the currently 
# "active" box is modified with the revised bounds.  The entire set of
# bounds can be cleared at once with the `clear_box` method, which by default 
# clears the settings on the active box selection; give a name to clear the 
# settings from a different box selection.

# %%
viz.clear_box()

# %% [markdown]
# If instead we want to manipulate an existing box selection, we can access the Box object, 
# manipulate it (e.g. by using `remove_from_allowed_set` or `add_to_allowed_set`), 
# and write it back into the Visualizer.

# %%
box = viz['Profitable']
box.remove_from_allowed_set('debt_type', 'Rev Bond')
viz['Profitable'] = box

# %% [markdown]
# ## Two Dimension Figures

# %% [markdown]
# The `Visualizer` object can also create an interactive two-dimensional scatter plot,
# using the `two_way` method. This method allows the user to specify the variables
# for both the `x` and `y` axis, and either can be any policy lever, exogenous 
# uncertainty, or performance measure.  These dimensions can be changed interactively
# later as well.  The resulting scatter plot is linked to the same selection of
# experiments in the interactive one-dimensional figures shown above, and by default
# the same experiments are highlighted in the same color scheme in all of these related
# figures.

# %%
viz.two_way(x='expand_capacity', y='time_savings')

# %% [markdown]
# One useful feature of the `two_way` is the ability to manually "lasso" a selection of 
# data points. This lasso selection does *not* need to be anything like a rectangular 
# box selection, as we have seen so far.  Once a lasso selection of data points is made 
# in the figure above, you can choose "Use Manual Selection" from the `Edit Selection...`
# menu at right, which will create a new `Visualizer` selection from the selected data.
# The highlight color changes to signify that this is not an editable rectangular box,
# and the selected data will be highlighted in *all* figures linked to this `Visualizer`,
# including the histograms above.

# %% [raw] raw_mimetype="text/restructuredtext"
# In addition to the `two_way`, which offers a feature-packed view of two dimensions at a time,
# there is also a scatter plot matrix :meth:`Visualizer.splom` option, which displays a configurable matrix of similar 
# two dimensional views.

# %%
viz.splom(
    rows=('expand_capacity','time_savings','net_benefits'), 
    cols='L',
    reset=True
)

# %%
viz.hmm(
    rows=('time_savings','net_benefits'), 
    cols='L',
    show_points=100,
    reset=True,
)

# %%
viz.new_selection(
    "time_savings * input_flow > 1000 & cost_of_capacity_expansion < 300",
    name="TimeSaved"
)

# %% [markdown]
# ## Dynamic Feature Scoring
#
# EMAT can score the relative importance of inputs for an experiment being within the selection, either for a typical rectangular selection based on thresholds, or for any arbitrary selection. These scores are recomputed and updated in near-real-time as the thresholds are adjusted.
#
# When the selection includes rectangular thresholds set on both inputs and outputs, the thresholded inputs are automatically excluded from the scoring algorithm.

# %%
viz.selection_feature_scores()

# %% [markdown]
# ## Using PRIM with the Interactive Explorer
#
# The PRIM tools are available directly within the interactive explorer. Simply 
# set a target as shown.

# %%
prim = viz.prim(target="net_benefits >= 0")

# %%
box1 = prim.find_box()

# %% [markdown]
# The tradeoff selector is directly integrated into the explorer.  In addition
# to the information visible by hovering over any point in the tradeoff selector
# figure, clicking on that point will create a new selection in the explorer, and
# set all of the interactive constraints 
# to the bounds given by that particular point.

# %%
box1.tradeoff_selector()

# %%
box1.select(20)

# %%
viz.status()

# %%
viz.lever_selectors()

# %% [markdown]
# We can also use PRIM to explore solutions based only on manipulating the
# policy levers, and not the combination of all inputs (levers & uncertainties).

# %%
prim_levers = viz.prim('levers', target="Profitable")

# %%
prim_levers.tradeoff_selector()

# %%
viz.parcoords()
