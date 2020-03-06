# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: EMAT
#     language: python
#     name: emat
# ---

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
# The `feature_scores` method from the `emat.analysis` subpackage allows for
# feature scoring based on the implementation found in the EMA Workbench.

# %%
from emat.analysis import feature_scores
fs = feature_scores(demo_scope, experiment_results, return_type='dataframe')
fs

# %% [markdown]
# Note that the `feature_scores` depend on the *scope* (to identify what are input features
# and what are outputs) and the *experiment_results*, but not on the model itself.  
#
# We can plot each of these input parameters using the `display_experiments` method,
# which can help visualize the underlying data and exactly how *B* is the most important
# feature for this example.

# %%
from emat.analysis import display_experiments
fig = display_experiments(demo_scope, experiment_results, render=False, return_figures=True)['Y']
fig.update_layout(
    xaxis_title_text =f"A (Feature Score = {fs.loc['Y','A']:.3f})",
    xaxis2_title_text=f"B (Feature Score = {fs.loc['Y','B']:.3f})",
    xaxis3_title_text=f"C (Feature Score = {fs.loc['Y','C']:.3f})",
)

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
wider_fs = feature_scores(demo_model.scope, wider_results, return_type='dataframe')
wider_fs

# %%
fig = display_experiments(demo_model.scope, wider_results, render=False, return_figures=True)['Y']
fig.update_layout(
    xaxis_title_text =f"A (Feature Score = {wider_fs.loc['Y','A']:.3f})",
    xaxis2_title_text=f"B (Feature Score = {wider_fs.loc['Y','B']:.3f})",
    xaxis3_title_text=f"C (Feature Score = {wider_fs.loc['Y','C']:.3f})",
)

# %% [markdown]
# The pattern has shifted, with the sine wave in *B* looking much more like the random noise,
# while the linear trend in *A* is now much more important in predicting the output, and
# the feature scores also shift to reflect this change.

# %% [markdown]
# ## Road Test Feature Scores
#
# We can apply the feature scoring methodology to the Road Test example 
# in a similar fashion.

# %%
from emat.model.core_python import Road_Capacity_Investment

road_scope = emat.Scope(emat.package_file('model','tests','road_test.yaml'))
road_test = PythonCoreModel(Road_Capacity_Investment, scope=road_scope)
road_test_design = road_test.design_experiments(n_samples=5000, sampler='lhs')
road_test_results = road_test.run_experiments(design=road_test_design)
feature_scores(road_scope, road_test_results)

# %% [markdown]
# The colors on the returned DataFrame highlight the most important input features
# for each performance measure.
