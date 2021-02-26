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
mm = model.create_metamodel_from_design('lhs', suppress_converge_warnings=True)
mm

# %% [markdown]
# If you are using the default meta-model regressor, as we are doing here, 
# you can directly access a cross-validation method that uses the experimental
# data to evaluate the quality of the regression model.  The `cross_val_scores`
# provides a measure of how well the meta-model predicts the experimental 
# outcomes, similar to an $R^2$ measure on a linear regression model.

# %%
mm.cross_val_scores()

# %% [markdown]
# We can apply the meta-model directly on a new design of experiments, and 
# use the `contrast_experiments` visualization tool to review how well the
# meta-model is replicating the underlying model's results.

# %%
design2 = mm.design_experiments(design_name='lhs_meta', n_samples=5000)

# %%
results2 = mm.run_experiments(design2)

# %%
from emat.analysis import contrast_experiments
contrast_experiments(mm.scope, results2, results)
