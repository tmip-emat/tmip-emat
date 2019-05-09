
# CHANGELOG

All notable changes to this project will be documented in this file.

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
