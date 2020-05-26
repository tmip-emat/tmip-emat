
from ..workbench.em_framework.evaluators import BaseEvaluator, SequentialEvaluator


def prepare_evaluator(evaluator, model):
	"""
	Prepare an evaluator for use.

	This utility function initializes a SequentialEvaluator by default,
	or if a dask.distributed Client is given as an evaluator, then
	a DistributedEvaluator.

	"""

	if evaluator is None:
		evaluator = SequentialEvaluator(model)

	if not isinstance(evaluator, BaseEvaluator):
		from dask.distributed import Client
		if isinstance(evaluator, Client):
			from ..workbench.em_framework.ema_distributed import DistributedEvaluator
			evaluator = DistributedEvaluator(model, client=evaluator)

	return evaluator
