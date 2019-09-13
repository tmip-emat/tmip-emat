


from dask.distributed import Client, LocalCluster

_cluster = None
_client = None

def get_client(n_workers=4, threads_per_worker=1):
	"""A helper function to start and use one distributed client."""
	global _client,  _cluster
	if _client is None:
		_cluster = LocalCluster(
			n_workers=n_workers,
			threads_per_worker=threads_per_worker,
		)
		_client = Client(_cluster)
	return _client

