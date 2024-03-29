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

# %% [raw] {"raw_mimetype": "text/restructuredtext"}
# .. py:currentmodule:: emat

# %%
import emat
from emat.util.loggers import timing_log
emat.versions()

# %% [raw] {"raw_mimetype": "text/restructuredtext"}
# .. _methodology-metamodel-creation:

# %% [markdown]
# # Meta-Model Creation

# %% [markdown]
# To demostrate the creation of a meta-model, we will use the Road Test example model
# included with TMIP-EMAT.  We will first create and run a design of experiments, to
# have some experimental data to define the meta-model.

# %%
import emat.examples
scope, db, model = emat.examples.road_test()
design = model.design_experiments(design_name='lhs')
results = model.run_experiments(design)

# %% [markdown]
# We can then create a meta-model automatically from these experiments.

# %%
from emat.model import create_metamodel

with timing_log("create metamodel"):
    mm = create_metamodel(scope, results, suppress_converge_warnings=True)

# %% [markdown]
# If you are using the default meta-model regressor, as we are doing here, 
# you can directly access a cross-validation method that uses the experimental
# data to evaluate the quality of the regression model.  The `cross_val_scores`
# provides a measure of how well the meta-model predicts the experimental 
# outcomes, similar to an $R^2$ measure on a linear regression model.

# %%
with timing_log("crossvalidate metamodel"):
    display(mm.cross_val_scores())

# %% [markdown]
# We can apply the meta-model directly on a new design of experiments, and 
# use the `contrast_experiments` visualization tool to review how well the
# meta-model is replicating the underlying model's results.

# %%
design2 = mm.design_experiments(design_name='lhs_meta', n_samples=10_000)

# %%
with timing_log("apply metamodel"):
    results2 = mm.run_experiments(design2)

# %%
results2.info()

# %%
from emat.analysis import contrast_experiments
contrast_experiments(mm.scope, results2, results)

# %% [markdown]
# ## Design Sizing
#
# An important advantage of using a Latin hypercube design of experiments, as is the default in TMIP-EMAT, is that the required number of experimental runs is not directly dependent on the dimensionality of the input variables and, importantly, does not grow exponentially with the number of dimensions. With a factorial or grid-based design, the number of experiments required does expand exponentially with the number of input dimensions. 

# %% [raw] {"raw_mimetype": "text/restructuredtext"}
# Practical experience across multiple domains has led to a “rule of thumb” that good results for prediction (i.e., for the creation of a meta-model) can usually be obtained from 10 experimental data points per input variable dimension :ref:`(Loeppky et al., 2009) <Loeppky2009>`. This can still lead to a sizable number of experiments that need to be undertaken using a core model, if there are a large number of uncertainties and policy levers to consider. When defining an exploratory scope, which defines the number and definition of uncertainties and policy levers, analysts will want to carefully consider the tradeoffs between the number of experiments to be run, the resources (i.e. runtime) needed for each core model experiment, and the detail needed to adequately describe and model both uncertainties and policy levers in the context of the overall policy goals being considered.

# %% [markdown] {"raw_mimetype": "text/restructuredtext"}
# If the underlying core model is generally well behaved, and has reasonable and proportional performance measure responses to changes in input values, then it may be possible to successfully develop a good-fitting meta-model with fewer than 10 experiments per input dimension.  Unfortunately, for all but the most basic models it is quite difficult to ascertain *a priori* whether the core model will in fact be well behaved, and general "rules of thumb" are not available for this situation.  Ultimately, if the dimensionality of the inputs cannot be reduced but a project cannot afford to undertake 10 model experiments per dimension, the best course is to specifically develop a design of experiments sized to the available resource budget.  This can be done using the `n_samples` arguments of the `design_experiments` function, which allows for creating a design of any particular size.  It is preferable to intentionally create this smaller design than to simply run a subset of a larger Latin Hypercube design, as smaller design will provide better coverage across the uncertainty space.

# %% [markdown]
# Whether a design of experiments is sized with 10 experiments per input variable, or fewer, it remains important to conduct a (cross-)validation for all meta-modeled performance measures, to ensure that meta-models are performing well. The 10x approach is a "rule of thumb" and not a guarantee that enough experimental data will be generated to develop a high quality meta-model in any given circumstance. Indeed, the guidance from Loeppky et al. is not simply that 10x is enough, but rather that either (a) 10x will be sufficient or (b) the number of experiments that will actually be sufficient is at least an order of magnitude larger.
#
# Performance measures that are poorly behaved or depend on complex interactions across multiple inputs can give poor results even with a larger number of experiments. For example, in the Road Test meta-model shown above, the "net benefits" performance measure exhibits a heteroskedastic response to "expand amount" and also rare high-value outliers that are dependent on at least 3 different inputs aligning (input flow, value of time, and expand amount). In combination, these conditions make it difficult to properly meta-model this performance measure, at least when using the automatic meta-modeling tools provided by TMIP-EMAT.  The low cross-validation score reported for this performance measure reflects this, and serves as a warning to analysts that the meta-model may be unreliable for this performance measure.

# %% [markdown]
# ## Partial Metamodels

# %% [markdown]
# It may be desirable in some cases to construct a *partial* metamodel, covering only a subset of the performance measures.  This is likely to be particularly desirable if some of the performance measures are poorly fit by the meta-model as noted above, or if a large number of performance measures are included in the scope, but only a few are of particular interest for a given analysis. The time required for generating and using meta-models is linear in the number of performance measures included, so if you have 100 performance measures but you are only presently interested in 5, your meta-model can be created much faster if you only include the 5 performance measures.  It will also run much faster, but the run time for metamodels is so small anyhow, it's likely you won't notice.
#
# To create a partial meta-model for a curated set of performance measures, you can use the `include_measures` argument of the `create_metamodel` command.

# %%
with timing_log("create limited metamodel"):
    mm2 = create_metamodel(
        scope, results, 
        include_measures=['time_savings', 'present_cost_expansion'],
        suppress_converge_warnings=True,
    )

with timing_log("crossvalidate limited metamodel"):
    display(mm2.cross_val_scores())

with timing_log("apply limited metamodel"):
    results2_limited = mm2.run_experiments(design2)
    
results2_limited.info()

# %% [markdown]
# There is also an `exclude_measures` argument for the `create_metamodel` command, which will retain all of the scoped performance measures *except* the enumerated list.  This can be handy for dropping a few measures that are not working well, either because the data is bad in some way or if the measure isn't well fitted using the metamodel.

# %%
with timing_log("create limited metamodel"):
    mm3 = create_metamodel(
        scope, results, 
        exclude_measures=['net_benefits'],
        suppress_converge_warnings=True,
    )

with timing_log("crossvalidate limited metamodel"):
    display(mm3.cross_val_scores())

with timing_log("apply limited metamodel"):
    results3_limited = mm3.run_experiments(design2)
    
results3_limited.info()
