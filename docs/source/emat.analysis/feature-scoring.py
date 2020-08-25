# -*- coding: utf-8 -*-
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

# %% [raw] raw_mimetype="text/restructuredtext"
# .. py:currentmodule:: emat.analysis.feature_scoring

# %%
import emat
emat.versions()

# %% [markdown]
# # Feature Scoring
#
# Feature scoring is a methodology for identifying what model inputs (in machine 
# learning terminology, “features”) have the greatest relationship to the outputs.  
# The relationship is not necessarily linear, but rather can be any arbitrary 
# linear or non-linear relationship.  For example, consider the function:

# %%
import numpy

def demo(A=0,B=0,C=0):
    """
    Y = A/2 + sin(6πB) + ε
    """
    return {'Y':A/2 + numpy.sin(6 * numpy.pi * B) + 0.1 * numpy.random.random()}


# %% [markdown]
# We can readily tell from the functional form that the *B* term is the
# most significant when all parameter vary in the unit interval, as the 
# amplitude of the sine wave attached to *B* is 1 (although the relationship 
# is clearly non-linear) while the maximum change
# in the linear component attached to *A* is only one half, and the output
# is totally unresponsive to *C*.
#
# To demonstrate the feature scoring, we can define a scope to explore this 
# demo model:

# %%
demo_scope = emat.Scope(scope_file='', scope_def="""---
scope:
    name: demo
inputs:
    A:
        ptype: exogenous uncertainty
        dtype: float
        min: 0
        max: 1
    B:
        ptype: exogenous uncertainty
        dtype: float
        min: 0
        max: 1
    C:
        ptype: exogenous uncertainty
        dtype: float
        min: 0
        max: 1
outputs:
    Y:  
        kind: info
""")

# %% [markdown]
# And then we will design and run some experiments to generate data used for
# feature scoring.

# %%
from emat import PythonCoreModel
demo_model = PythonCoreModel(demo, scope=demo_scope)
experiments = demo_model.design_experiments(n_samples=5000)
experiment_results = demo_model.run_experiments(experiments)

# %% [markdown]
# The :func:`feature_scores` method from the `emat.analysis` subpackage allows for
# feature scoring based on the implementation found in the EMA Workbench.

# %%
from emat.analysis import feature_scores
fs = feature_scores(demo_scope, experiment_results)
fs

# %% [raw] raw_mimetype="text/restructuredtext"
# Note that the :func:`feature_scores` depend on the *scope* (to identify what are input features
# and what are outputs) and the *experiment_results*, but not on the model itself.  
#
# We can plot each of these input parameters using the `display_experiments` method,
# which can help visualize the underlying data and exactly how *B* is the most important
# feature for this example.

# %%
from emat.analysis import display_experiments
fig = display_experiments(demo_scope, experiment_results, render=False, return_figures=True)['Y']
fig.update_layout(
    xaxis_title_text =f"A (Feature Score = {fs.data.loc['Y','A']:.3f})",
    xaxis2_title_text=f"B (Feature Score = {fs.data.loc['Y','B']:.3f})",
    xaxis3_title_text=f"C (Feature Score = {fs.data.loc['Y','C']:.3f})",
)
from emat.util.rendering import render_plotly
render_plotly(fig, '.png')

# %% [markdown]
# One important thing to consider is that changing the range of the input parameters 
# in the scope can significantly impact the feature scores, even if the underlying 
# model itself is not changed.  For example, consider what happens to the features
# scores when we expand the range of the uncertainties:

# %%
demo_model.scope = emat.Scope(scope_file='', scope_def="""---
scope:
    name: demo
inputs:
    A:
        ptype: exogenous uncertainty
        dtype: float
        min: 0
        max: 5
    B:
        ptype: exogenous uncertainty
        dtype: float
        min: 0
        max: 5
    C:
        ptype: exogenous uncertainty
        dtype: float
        min: 0
        max: 5
outputs:
    Y:  
        kind: info
""")

# %%
wider_experiments = demo_model.design_experiments(n_samples=5000)
wider_results = demo_model.run_experiments(wider_experiments)

# %%
from emat.analysis import feature_scores
wider_fs = feature_scores(demo_model.scope, wider_results)
wider_fs

# %%
fig = display_experiments(demo_model.scope, wider_results, render=False, return_figures=True)['Y']
fig.update_layout(
    xaxis_title_text =f"A (Feature Score = {wider_fs.data.loc['Y','A']:.3f})",
    xaxis2_title_text=f"B (Feature Score = {wider_fs.data.loc['Y','B']:.3f})",
    xaxis3_title_text=f"C (Feature Score = {wider_fs.data.loc['Y','C']:.3f})",
)
render_plotly(fig, '.png')

# %% [markdown]
# The pattern has shifted, with the sine wave in *B* looking much more like the random noise,
# while the linear trend in *A* is now much more important in predicting the output, and
# the feature scores also shift to reflect this change.

# %% [markdown]
# ## Road Test Feature Scores
#
# We can apply the feature scoring methodology to the Road Test example 
# in a similar fashion.  To demonstrate scoring, we'll first load and run
# a sample set of experients.

# %%
from emat.model.core_python import Road_Capacity_Investment
road_scope = emat.Scope(emat.package_file('model','tests','road_test.yaml'))
road_test = PythonCoreModel(Road_Capacity_Investment, scope=road_scope)
road_test_design = road_test.design_experiments(sampler='lhs')
road_test_results = road_test.run_experiments(design=road_test_design)

# %% [raw] raw_mimetype="text/restructuredtext"
# With the experimental results in hand, we can use the same :func:`feature_scores`
# function to compute the scores.

# %%
feature_scores(road_scope, road_test_results)

# %% [raw] raw_mimetype="text/restructuredtext"
# By default, the :func:`feature_scores`
# function returns a stylized DataFrame, but we can also 
# use the `return_type` argument to get a plain 'dataframe'
# or a rendered svg 'figure'.

# %%
feature_scores(road_scope, road_test_results, return_type='dataframe')

# %%
feature_scores(road_scope, road_test_results, return_type='figure')

# %% [markdown]
# The colors on the returned DataFrame highlight the most important input features
# for each performance measure (i.e., in each row).  The yellow highlighted cell 
# indicates the most important input feature for each output feature, and the 
# other cells are colored from yellow through green to blue, showing high-to-low
# importance.  These colors are from matplotlib's default "viridis" colormap. 
# A different colormap can be used by giving a named colormap in the `cmap`
# argument.

# %%
feature_scores(road_scope, road_test_results, cmap='copper')

# %% [markdown]
# You may also notice small changes in the numbers given in the two tables above. This
# occurs because the underlying algorithm for scoring uses a random trees algorithm. If
# you need to have stable (replicable) results, you can provide an integer in the 
# `random_state` argument.

# %%
feature_scores(road_scope, road_test_results, random_state=1, cmap='bone')

# %% [markdown]
# Then if we call the function again with the same `random_state`, we get the same numerical result.

# %%
feature_scores(road_scope, road_test_results, random_state=1, cmap='YlOrRd')

# %% [markdown]
# ## Interpreting Feature Scores
#
# The correct interpretation of feature scores is obviously important.  As noted above,
# the feature scores can reveal both linear and non-linear relationships. But the scores
# themselves give no information about which is which. 
#
# In addition, while the default feature scoring algorithm generates scores that total 
# to 1.0, it does not necessarily map to dividing up the explained variance. Factors that
# have little to no effect on the output still are given non-zero feature score values.
# You can see an example of this in the "demo" function above; that simple example 
# literally ignores the "C" input, but it has a non-zero score assigned.  If there are
# a large number of superfluous inputs, they will appear to reduce the scores attached
# to the meaningful inputs.
#
# It is also important to remember that these scores do not fully reflect 
# any asymmetric relationships in the data. A feature may be very important for some portion
# of the range of a performance measure, and less important in other parts of the range.
# For example, in the Road Test model, the "expand_capacity" lever has a highly asymmetric
# impact on the "net_benefits" measure: it is very important in
# determining negative values (when congestion isn't going to be bad due to low "input_flow" volumes, 
# the net loss is limited by how much we spend), but not so important for positive values 
# (nearly any amount of expansion has a big payoff if congestion is going to be bad due 
# to high "input_flow" volume).  If we plot this three-way relationship specifically, we can 
# observe it on one figure:

# %%
from matplotlib import pyplot as plt
fig, ax = plt.subplots()
ax = road_test_results.plot.scatter(
    c='net_benefits', 
    y='expand_capacity', 
    x='input_flow', 
    cmap='coolwarm',
    ax=ax,
)

# %% [markdown]
# Looking at the figure above, we can see the darker red clustered to the right,
# and the darker blue clustered in the top left.
# However, if we are not aware of this particular three-way relationship
# *a priori*, it may be difficult to discover it by looking through various
# combinations of three-way relationships.  To uncover this kind of relationship,
# threshold scoring may be useful.

# %% [markdown]
# ## Threshold Scoring

# %% [raw] raw_mimetype="text/restructuredtext"
# Threshold scoring provides a set of feature scores that don't relate to the
# overall magnitude of a performance measure, but rather whether that performance
# measure is above or below some threshold level.  The :func:`threshold_feature_scores`
# function computes such scores for a variety of different thresholds, to develop
# a picture of the relationship across the range of outputs.  

# %%
from emat.analysis.feature_scoring import threshold_feature_scores

threshold_feature_scores(road_scope, 'net_benefits', road_test_results)

# %% [raw] raw_mimetype="text/restructuredtext"
# This table of data may be overwhelming even for an analyst to interpret, but
# the :func:`threshold_feature_scores` function also offers a violin or ridgeline style figure
# that displays similar information in a more digestible format.

# %%
threshold_feature_scores(road_scope, 'net_benefits', road_test_results, return_type='figure.png')

# %%
threshold_feature_scores(road_scope, 'net_benefits', road_test_results, return_type='ridge figure.svg')

# %% [markdown]
# In these figures, we can see that "expand_capacity" is important for negative outcomes,
# but for positive outcomes we should focus more on "input_flow", and to a lesser but
# still meaningful extent also "value_of_time".

# %% [markdown]
# ## Feature Scoring API 

# %% [raw] raw_mimetype="text/restructuredtext"
#
# .. autofunction:: feature_scores
# .. autofunction:: threshold_feature_scores
