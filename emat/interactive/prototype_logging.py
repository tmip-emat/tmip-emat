import ipywidgets as widgets
import logging

class OutputWidgetHandler(logging.Handler):
	""" Custom logging handler sending logs to an output widget """

	def __init__(self, *args, **kwargs):
		super(OutputWidgetHandler, self).__init__(*args, **kwargs)
		layout = {
			'border': '1px solid red',
		}
		self.out = widgets.Output(layout=layout)

	def emit(self, record):
		""" Overload of logging.Handler method """
		formatted_record = self.format(record)
		with self.out:
			print(formatted_record)

	def show_logs(self):
		""" Show the logs """
		display(self.out)

	def clear_logs(self):
		""" Clear the current logs """
		self.out.clear_output()


logger = logging.getLogger(__name__)
handler = OutputWidgetHandler()
handler.setFormatter(logging.Formatter('%(asctime)s  - [%(levelname)s] %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

import inspect, os

def execution_tracer(log, data=None):
	fi = inspect.stack()[1][1]
	fi = os.path.basename(fi)
	log.debug(f"%% {inspect.stack()[1][3]} [{fi}:{inspect.stack()[1][2]}]")
	if data is not None:
		log.debug(f"%~ {data}")