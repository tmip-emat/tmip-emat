# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
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

# %% [raw] {"raw_mimetype": "text/restructuredtext"}
# .. py:currentmodule:: emat

# %%
import emat
from emat.util.loggers import timing_log
emat.versions()

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
# ## Partial Metamodels

# %% [markdown]
# It may be desirable in some cases to construct a *partial* metamodel, covering only a subset of the performance measures.  This is likely to be particularly desirable if a large number of performance measures are included in the scope, but only a few are of particular interest for a given analysis. The time required for generating and using meta-models is linear in the number of performance measures included, so if you have 100 performance measures but you are only presently interested in 5, your meta-model can be created much faster if you only include the 5 performance measures.  It will also run much faster, but the run time for metamodels is so small anyhow, it's likely you won't notice.
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
