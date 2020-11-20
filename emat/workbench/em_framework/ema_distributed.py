
import os
import sys
import traceback
import math
from itertools import zip_longest
import time

from .evaluators import BaseEvaluator
from ..util import ema_logging
from .parameters import experiment_generator, Case
from ..util.ema_exceptions import EMAError, CaseError

from ..util import get_module_logger
_logger = get_module_logger(__name__)

from dask.distributed import Client, as_completed, get_worker, WorkerPlugin, Worker

def store_model_on_worker(name, model):
	worker = get_worker()
	if not hasattr(worker, '_ema_models'):
		worker._ema_models = {}
	worker._ema_models[name] = model


def run_experiment_on_worker(experiment):
	'''Run a single experiment on a dask worker.

	This code makes sure that model is initialized correctly.

	Parameters
	----------
	experiment : Case

	Returns
	-------
	experiment_id: int
	result : dict

	Raises
	------
	EMAError
		if the model instance raises an EMA error, these are reraised.
	Exception
		Catch all for all other exceptions being raised by the model.
		These are reraised.

	'''
	worker = get_worker()

	model_name = experiment.model_name
	model = worker._ema_models[model_name]
	policy = experiment.policy.copy()

	scenario = experiment.scenario
	try:
		model.run_model(scenario, policy)
	except CaseError as e:
		_logger.warning(str(e))
	except Exception as e:
		_logger.exception(str(e))
		try:
			model.cleanup()
		except Exception:
			raise e

		exception = traceback.print_exc()
		if exception:
			sys.stderr.write(exception)
			sys.stderr.write("\n")

		errortype = type(e).__name__
		raise EMAError(("exception in run_model"
						"\nCaused by: {}: {}".format(errortype, str(e))))

	outcomes = model.outcomes_output
	model.reset_model()

	return experiment.experiment_id, outcomes.copy(), getattr(model, 'comment_on_run', None)

def run_experiments_on_worker(experiments):
	"""
	Run multiple experiments in a batch on one worker.

	Sending a batch of experiments cuts down on the number of communications
	required between scheduler and worker processes.

	Parameters
	----------
	experiments : Iterable of Case

	Returns
	-------
	tuple
		The results from `run_experiment_on_worker`
	"""
	return tuple(run_experiment_on_worker(experiment) for experiment in experiments if experiment is not None)


def grouper(iterable, n, fillvalue=None):
	"Collect data into fixed-length chunks or blocks"
	# grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
	args = [iter(iterable)] * n
	return zip_longest(*args, fillvalue=fillvalue)

class ModelPlugin(WorkerPlugin):
	def __init__(self, models):
		self._msis = models
	def setup(self, worker: Worker):
		if not hasattr(worker, '_ema_models'):
			worker._ema_models = {}
		for msi in self._msis:
			worker._ema_models[msi.name] = msi


class DistributedEvaluator(BaseEvaluator):
	"""Evaluator using dask.distributed

	Parameters
	----------
	msis : collection of models
	client : distributed.Client (optional)
		A client can be provided. If one is not provided, a default Client
		will be created.
	batch_size : int (optional)
		The number of experiment to batch together when pushing tasks to distributed workers.
		If not given, the first call to evaluate_experiments will make a reasonable guess that
		will allocate batches so that there are about 10 tasks per worker.  This may or may not
		be efficient.
	max_n_workers : int (default 32)
		The maximum number of workers that will be created for a default Client.  If the number
		of cores available is smaller than this number, fewer workers will be spawned.

	"""

	_default_client = None

	def __init__(
			self,
			msis,
			*,
			client=None,
			batch_size=None,
			max_n_workers=32,
			asynchronous=False,
	):
		super().__init__(msis, )

		# Initialize a default dask.distributed client if one is not given
		if client is None:
			if type(self)._default_client is None:
				import multiprocessing
				n_workers = min(multiprocessing.cpu_count(), max_n_workers)
				type(self)._default_client = Client(
					n_workers=n_workers,
					threads_per_worker=1,
					asynchronous=asynchronous,
				)
			client = type(self)._default_client

		self.client = client
		self.batch_size = batch_size
		self.asynchronous = asynchronous

		# The worker plugin ensures that all models are copied
		# to workers before model runs are conducted, even if a
		# worker crashes and needs to be restarted.
		self.plugin = ModelPlugin(self._msis)

		if self.client and not asynchronous:
			self.client.register_worker_plugin(self.plugin)

	def initialize(self):
		pass

	def finalize(self):
		pass

	def broadcast_models_to_workers(self):
		for msi in self._msis:
			self.client.run(store_model_on_worker, msi.name, msi)

	def evaluate_experiments(self, scenarios, policies, callback, zip_over=None):
		_logger.debug("evaluating experiments asynchronously")

		ex_gen = experiment_generator(scenarios, self._msis, policies, zip_over)

		cwd = os.getcwd()

		log_message = ('storing scenario %s for policy %s on model %s')

		experiments = {
			experiment.experiment_id: experiment
			for experiment in ex_gen
		}

		if self.batch_size is None:
			# make a guess at a good batch size if one was not given
			n_workers = len(self.client.scheduler_info()['workers'])
			n_experiments = len(experiments)
			self.batch_size = math.ceil(n_experiments / n_workers / 10 )

		# Experiments are sent to workers in batches, as the task-scheduler overhead is high for quick-running models.
		batches = grouper(experiments.values(), self.batch_size)

		if self.asynchronous:

			self.futures = []
			self.futures_ilocs = []

			async def f(_b):
				future = self.client.submit(run_experiments_on_worker, _b)
				result_batch = await self.client.gather(future, asynchronous=True)
				for (experiment_id, outcome, comment_on_run) in result_batch:
					experiment = experiments[experiment_id]
					_logger.debug(
						log_message,
						experiment.scenario.name,
						experiment.policy.name,
						experiment.model_name,
					)
					callback(experiment, outcome)
					if comment_on_run:
						_logger.warning(comment_on_run)
				return result_batch

			for b in batches:
				self.futures.append(f(b))
				ilocs = []
				for i in b:
					if hasattr(i,'experiment_id'):
						ilocs.append(i.experiment_id)
				self.futures_ilocs.append(ilocs)

		else:
			# Dask no longer supports mapping over Iterators or Queues.
			# Using a normal for loop and Client.submit
			outcomes = [self.client.submit(run_experiments_on_worker, b) for b in batches]
			_logger.debug("waiting to receive experiment results")

			for future, result_batch in as_completed(outcomes, with_results=True):
				for (experiment_id, outcome, comment_on_run) in result_batch:
					experiment = experiments[experiment_id]
					_logger.debug(
						log_message,
						experiment.scenario.name,
						experiment.policy.name,
						experiment.model_name,
					)
					callback(experiment, outcome)
					if comment_on_run:
						_logger.warning(comment_on_run)

			os.chdir(cwd)

			_logger.debug("completed evaluate_experiments")


async def AsyncDistributedEvaluator(
		msis,
		*,
		client=None,
		batch_size=None,
		max_n_workers=None,
):
	# Initialize a default dask.distributed client if one is not given
	if client is None:
		if DistributedEvaluator._default_client is None:
			_logger.info("initializing default DistributedEvaluator.client")
			import multiprocessing
			if max_n_workers:
				n_workers = min(multiprocessing.cpu_count(), max_n_workers)
				_logger.info(f"  max_n_workers={max_n_workers}, actual n_workers={n_workers}")
			else:
				n_workers = multiprocessing.cpu_count()
			_logger.info(f"  n_workers={n_workers}")
			DistributedEvaluator._default_client = await Client(
				n_workers=n_workers,
				threads_per_worker=1,
				asynchronous=True,
			)
			_logger.info("completed initializing default DistributedEvaluator.client")
		client = DistributedEvaluator._default_client

	self = DistributedEvaluator(
		msis,
		client=client,
		batch_size=batch_size,
		max_n_workers=max_n_workers,
		asynchronous=True,
	)

	await self.client.register_worker_plugin(self.plugin)
	return self
