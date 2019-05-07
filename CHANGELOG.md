
# CHANGELOG

All notable changes to this project will be documented in this file.

## Pending for v0.1.2 -- May 2019

- `AbstractCoreModel.run_experiments_from_design` has been removed in 
  favor of `AbstractCoreModel.run_experiments`, which provides a more 
  unified interface.
- Performance measure documentation and examples have been modified
  to favor `metamodeltype` over `transform`. The `transform` attribute
  of `Measure` remains in place, as a suitable string-based hook for
  using the `function` attribute offered by `ema_workbench`.
- Add `robust_evaluate` method to `AbstractCoreModel`.
- Back-end code has been updated to be compatible with version 2.1 of
  the `ema_workbench` package.