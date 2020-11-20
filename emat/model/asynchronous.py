import numpy as np
import pandas as pd
import asyncio
from ..workbench.em_framework.ema_distributed import AsyncDistributedEvaluator

from ..workbench.util import get_module_logger
_logger = get_module_logger(__name__)


class AsyncExperimentalDesign:
	def __init__(self, model, design, stagger_start=0):
		self.model = model
		self.params = design.columns
		self._storage = design.reindex(
			columns=model.scope.get_all_names(),
			copy=True,
		)
		self.stagger_start = stagger_start
		self.task = None
		self._status = pd.Series(
			data='pending',
			index=self._storage.index,
		)

	def __repr__(self):
		return f"<emat.AsyncExperimentalDesign with {self.progress()}>"

	async def run(
			self,
			evaluator,
			max_n_workers=None,
			stagger_start=None,
			batch_size=None,
	):
		if stagger_start is not None:
			self.stagger_start = stagger_start
		if evaluator is None:
			evaluator = await AsyncDistributedEvaluator(
				self.model,
				max_n_workers=max_n_workers,
				batch_size=batch_size,
			)
		self._evaluator = evaluator
		self._client = self.evaluator.client
		self.model.run_experiments(
			design=self._storage[self.params],
			evaluator=evaluator,
		)
		self._tasks = []
		for fut, ilocs in zip(evaluator.futures,evaluator.futures_ilocs):
			t = asyncio.create_task(fut)
			t.add_done_callback(self._update_storage)
			self._tasks.append(t)
			self._status.iloc[ilocs] = 'queued'
			hold = 0
			while hold < self.stagger_start:
				await asyncio.sleep(1)
				hold += 1
		return self._tasks

	def _update_storage(self, fut):
		for i in fut.result():
			y = pd.DataFrame(i[1], index=[self._storage.index[i[0]]])
			self._storage.update(y)
			self._status.iloc[i[0]] = i[2] or 'done'

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

	def status(self):
		return self._status.copy()

	def progress(self, raw=False):
		n_done = (self._status == 'done').sum()
		n_queued = (self._status == 'queued').sum()
		n_pending = (self._status == 'pending').sum()
		n_total = len(self._status)
		n_failed = n_total - n_done - n_queued - n_pending
		if raw:
			summary = {}
			if n_done: summary['done'] = n_done
			if n_queued: summary['queued'] = n_queued
			if n_pending: summary['pending'] = n_pending
			if n_failed: summary['failed'] = n_failed
			return summary
		message_part = []
		if n_done:
			message_part.append(f"{n_done} done")
		if n_pending:
			message_part.append(f"{n_pending} pending")
		if n_queued:
			message_part.append(f"{n_queued} queued")
		if n_failed:
			message_part.append(f"{n_failed} failed")
		return f"{n_total} runs: " + ", ".join(message_part)

	async def gather(self):
		try:
			_tasks = self._tasks
		except AttributeError:
			return
		result = await asyncio.gather(*_tasks)
		return result

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
