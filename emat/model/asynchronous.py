import numpy as np
import pandas as pd
import asyncio
from ..workbench.em_framework.ema_distributed import AsyncDistributedEvaluator

from ..workbench.util import get_module_logger
_logger = get_module_logger(__name__)


class AsyncExperimentalDesign:
	def __init__(self, model, design):
		self.model = model
		self.params = design.columns
		self._storage = design.reindex(
			columns=model.scope.get_all_names(),
			copy=True,
		)

	async def run(
			self,
			evaluator,
			max_n_workers=None,
			stagger_start=0,
			batch_size=None,
	):
		if evaluator is None:
			evaluator = await AsyncDistributedEvaluator(
				self.model,
				max_n_workers=max_n_workers,
				stagger_start=stagger_start,
				batch_size=batch_size,
			)
		self._evaluator = evaluator
		self._client = self.evaluator.client
		self.model.run_experiments(
			design=self._storage[self.params],
			evaluator=evaluator,
		)
		tasks = []
		for fut in evaluator.futures:
			t = asyncio.create_task(fut)
			t.add_done_callback(self._update_storage)
			tasks.append(t)
			await asyncio.sleep(stagger_start)
		return tasks

	def _update_storage(self, fut):
		for i in fut.result():
			y = pd.DataFrame(i[1], index=[self._storage.index[i[0]]])
			self._storage.update(y)

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

	def results(self):
		return self._storage.copy()


def asynchronous_experiments(
		model,
		design,
		evaluator=None,
		max_n_workers=None,
		stagger_start=0,
		batch_size=None,
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
			batch_size=batch_size,
		)
	)
	return t
