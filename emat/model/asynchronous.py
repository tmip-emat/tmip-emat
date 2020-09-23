
import asyncio
from ..workbench.em_framework.ema_distributed import AsyncDistributedEvaluator

from ..workbench.util import get_module_logger
_logger = get_module_logger(__name__)


class AsyncExperimentalDesign:
	def __init__(self, model, design):
		self.model = model
		self.results = design.copy()
		self.params = design.columns

	async def run(
			self,
			evaluator,
			max_n_workers=None,
			stagger_start=None,
	):
		if evaluator is None:
			evaluator = await AsyncDistributedEvaluator(
				self.model,
				max_n_workers=max_n_workers,
				stagger_start=stagger_start,
			)
		self._evaluator = evaluator
		self._client = self.evaluator.client
		self.model.run_experiments(
			design=self.results[self.params],
			evaluator=evaluator,
		)
		# TODO: write results as available?
		return asyncio.gather(*evaluator.futures)

	@property
	def client(self):
		try:
			return self._client
		except AttributeError:
			return

	@property
	def evaluator(self):
		try:
			return self._evaluator
		except AttributeError:
			return


def asynchronous_experiments(
		model,
		design,
		evaluator=None,
		max_n_workers=None,
		stagger_start=None,
):
	_logger.info(f"asynchronous_experiments(max_n_workers={max_n_workers})")
	t = AsyncExperimentalDesign(
		model,
		design,
	)
	t.task = asyncio.create_task(
		t.run(
			evaluator,
			max_n_workers,
			stagger_start=stagger_start,
		)
	)
	return t
