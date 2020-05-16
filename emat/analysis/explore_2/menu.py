
from ipywidgets import Dropdown

class Menu(Dropdown):

	def __init__(self, label, commands):
		self._label = label
		self._commands = commands
		options = [label]
		for (cmd, _) in commands.items():
			options.append(cmd)
		super().__init__(options=options, value=label)
		self.observe(self._change_value, names='value')

	def _change_value(self, payload):
		if payload['new'] == self._label: return
		if payload['new'] in self._commands:
			cmd = self._commands[payload['new']]
			if cmd is not None:
				cmd()
		self.value = self._label

