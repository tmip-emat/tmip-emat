
# CHANGELOG

All notable changes to this project will be documented in this file.

## v0.1.4 -- July 2019

### New Features

- Additional documentation and features for PRIM, including
  the ability to convert EMA Workbench's PrimBox to the EMAT Box
  format.
- A version check to ensure a compatible version of the EMA Workbench
  is installed. 
- `AbstractCoreModel.create_metamodel_from_designs` allows the creation
  of a meta-model from a multi-stage experimental design, instead of only
  from a single design.
- Add truncated uniform latin hypercube samplers, named 'ulhs99', 'ulhs98',
  and 'ulhs95'.  These samplers ignore the defined underlying distribution
  after truncating the tails of the distribution.
- Additional tests to support non-uniform distributions.

### Changes / Removals

- `AbstractCoreModel.run_experiments` allows the `db` argument to be
  set to *False*, which will prevent the model runs from being saved
  to a database, even if a default database is defined for the model.
- `FilesCoreModel` nor includes a configurable `rel_output_path` 
  attribute, defined in the configuration file, to set the location
  of output files within a model run. Previously, this value was
  hard-coded as the subdirectory './Outputs', and that remains the
  default value if `rel_output_path` is not set.


## v0.1.3 -- May 2019

### New Features

- Correlated samplers have been added for 'lhs' (Latin Hypercube) and 
  'mc' (Monte Carlo) sampling.  These samplers use the `corr` attributes
  already defined for parameters and in the scope file.
- Add `robust_evaluate` method to `AbstractCoreModel`.
- Allow initialization of `Scope` objects with no 'desc' key at the top
  level, which implies setting the description to an empty string.
- Add a `DistributionTypeError` class, and use it to improve the 
  infer-dtype capability for making parameters.  When a continuous
  distribution is given for what otherwise has been inferred as an
  integer parameter, that parameter is promoted to 'float'.  As always,
  if there is any chance of confusion it is better to define the 
  parameter dtype explicitly.
  
### Changes / Removals

- `AbstractCoreModel.run_experiments_from_design` has been removed in 
  favor of `AbstractCoreModel.run_experiments`, which provides a more 
  unified interface.
- Performance measure documentation and examples have been modified
  to favor `metamodeltype` over `transform`. The `transform` attribute
  of `Measure` remains in place, as a suitable string-based hook for
  using the `function` attribute offered by `ema_workbench`.
- Back-end code has been updated to be compatible with version 2.1 of
  the `ema_workbench` package.

## v.0.1.2 -- April 2019

### New Features

- Add the ability to design experiments that sample jointly from 
  uncertainties and levers (the default for EMAT) or to sample 
  independently from uncertainties and levers, and then combine these
  two sets of samples in a full-factorial manner (the default for 
  EMA Workbench).
