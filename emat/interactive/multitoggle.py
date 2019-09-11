
from ipywidgets import Box, widget_selection, ToggleButton, Layout, Label, Widget
import traitlets


class MultiToggleButtons(Box):
	description = traitlets.Unicode()
	value = traitlets.Tuple()
	options = traitlets.Union([traitlets.List(), traitlets.Dict()])
	style = traitlets.Dict()

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._selection_obj = widget_selection._MultipleSelection()
		traitlets.link((self, 'options'), (self._selection_obj, 'options'))
		traitlets.link((self, 'value'), (self._selection_obj, 'value'))

		@observer(self, 'options')
		def _(*_):
			self.buttons = [ToggleButton(description=label,
											layout=Layout(
												margin='1',
												width='auto'
											))
							for label in self._selection_obj._options_labels]
			if self.description:
				self.label = Label(self.description, layout=Layout(width=self.style.get('description_width', '100px')))
			else:
				self.label = Label(self.description, layout=Layout(width=self.style.get('description_width', '0px')))
			self.children = [self.label]+self.buttons

			@observer(self.buttons, 'value')
			def _(*_):
				self.value = tuple(value
								   for btn, value in zip(self.buttons, self._selection_obj._options_values)
								   if btn.value)

		self.add_class('btn-group')

	def reset(self):
		opts = self.options
		self.options = []
		self.options = opts

	def set_value(self, x):
		for b, opt in zip(self.buttons, self.options):
			b.value = (opt in x)

	def set_all_on(self):
		for b, opt in zip(self.buttons, self.options):
			b.value = True

	def set_all_off(self):
		for b, opt in zip(self.buttons, self.options):
			b.value = False


def observer(widgets, trait_name):
	def wrapper(func):
		if isinstance(widgets, Widget):
			widgets.observe(func, trait_name)
		else:
			for w in widgets:
				w.observe(func, trait_name)
		func()

	return wrapper

