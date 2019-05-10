
from collections import namedtuple
from collections.abc import MutableMapping, Mapping
import itertools
from typing import Collection
import pandas

from .scope import Scope, ScopeError

Bounds = namedtuple('Bounds', ['lowerbound', 'upperbound'])

Bounds.__doc__ = """
A lower and upper bound as a 2-tuple.

Args:
	lowerbound (numeric or None): 
		The lower bound to set, or None 
		if there is no lower bound.
	upperbound (numeric or None): 
		The upper bound to set, or None 
		if there is no upper bound.
"""

class GenericBoxMixin:
	# Generic methods applicable to both Box and ChainedBox

	def inside(self, df):
		"""
		For each row of a DataFrame, identify if it is inside the box.

		Args:
			df (pandas.DataFrame): Must include a column matching every
				thresholded feature.

		Returns:
			pandas.Series
				With dtype bool.
		"""
		within = pandas.Series(True, index=df.index)
		for label, bounds in self.thresholds.items():
			if bounds.lowerbound is not None:
				within &= (df[label] >= bounds.lowerbound)
			if bounds.upperbound is not None:
				within &= (df[label] <= bounds.upperbound)
		return within

class Box(Mapping, GenericBoxMixin):
	"""
	A Box defines a set of restricted dimensions for a Scope.

	Args:
		name (str): The name for this Box.
		parent (str, optional):
			The name of the parent for this Box.  When extracted
			as a :class:`ChainedBox` from a collection of :class:`Boxes`,
			the thresholds will also include any thresholds inherited
			from this box's ancestor(s).
		scope (Scope, optional):
			A scope to associate with this box.
		upper_bounds (Mapping[str, numeric], optional):
			If given, a mapping with keys giving feature names
			and values giving an upper bound for each feature.
		lower_bounds (Mapping[str, numeric], optional):
			If given, a mapping with keys giving feature names
			and values giving a lower bound for each feature.
		bounds (Mapping[str, Bounds], optional):
			If given, a mapping with keys giving feature names
			and values giving :class:`Bounds` for each feature.
		allowed (Mapping[str, Set], optional):
			If given, a mapping with keys giving feature names
			and values giving the available :class:`Set` for
			each feature.
		relevant (Iterable, optional):
			If given, a set of names of relevant features.

	Attributes:
		thresholds (Dict[str,Union[Bounds,Set]]):
			The restricted dimensions in this Box, with feature names as
			keys and :class:`Bounds` or a :class:`Set` of available discrete
			values as the dictionary values.
		relevant_features (Set[str]):
			A :class:`Set` of features that are relevant for this Box.
			These are features, which are not themselves constrained,
			but should be considered in any analytical report developed
			based on this Box.


	"""
	def __init__(
			self,
			name,
			parent=None,
			scope=None,
			upper_bounds=None,
			lower_bounds=None,
			bounds=None,
			allowed=None,
			relevant=None,
	):
		self.thresholds = {}

		if relevant is None:
			self.relevant_features = set()
		else:
			self.relevant_features = set(relevant)

		self.parent_box_name = parent
		self.scope = scope
		self.name = name

		if upper_bounds:
			for k,v in upper_bounds.items():
				self.set_upper_bound(k,v)

		if lower_bounds:
			for k,v in lower_bounds.items():
				self.set_lower_bound(k,v)

		if bounds:
			for k,v in bounds.items():
				self.set_bounds(k,v)

		if allowed:
			for k,v in allowed.items():
				self.replace_allowed_set(k,v)

	@property
	def scope(self):
		"""Scope: A scope associated with this Box."""
		return self._scope

	@scope.setter
	def scope(self, x):
		if x is None or isinstance(x, Scope):
			self._scope = x
		else:
			raise TypeError('scope must be Scope or None')

	@property
	def measure_thresholds(self):
		"""
		Dict[str,Union[Bounds,Set]]:
			The thresholds in this Box associated with performance measures.

			A Scope must be associated with this Box to access this property.
		"""
		if self.scope is None:
			raise ValueError("need scope")
		names = self.scope.get_measure_names()
		return {k:v for k,v in self.thresholds.items() if k in names}

	@property
	def uncertainty_thresholds(self):
		"""
		Dict[str,Union[Bounds,Set]]:
			The thresholds in this Box associated with exogenous uncertainties.

			A Scope must be associated with this Box to access this property.
		"""
		if self.scope is None:
			raise ValueError("need scope")
		names = self.scope.get_uncertainty_names()
		return {k:v for k,v in self.thresholds.items() if k in names}

	@property
	def lever_thresholds(self):
		"""
		Dict[str,Union[Bounds,Set]]:
			The thresholds in this Box associated with policy levers.

			A Scope must be associated with this Box to access this property.
		"""
		if self.scope is None:
			raise ValueError("need scope")
		names = self.scope.get_lever_names()
		return {k:v for k,v in self.thresholds.items() if k in names}

	def __getitem__(self, key):
		return self.thresholds[key]

	def set_lower_bound(self, key, value):
		"""
		Set a lower bound, retaining existing upper bound.

		Args:
			key (str):
				The feature name to which this lower bound
				will be attached.
			value (numeric or None):
				The lower bound. Set explicitly to 'None' to
				leave unbounded from below.

		Raises:
			ScopeError:
				If a scope is attached to this box but the `key` cannot
				be found in the scope.
		"""
		current = self.thresholds.get(key, Bounds(None,None))
		if isinstance(current, set):
			raise ValueError("cannot set lowerbound on a set")
		if self.scope is not None:
			if key in self.scope.get_all_names():
				self.thresholds[key] = Bounds(value, current.upperbound)
			else:
				raise ScopeError(f"cannot set threshold on '{key}'")
		else:
			self.thresholds[key] = Bounds(value, current.upperbound)

	def set_upper_bound(self, key, value):
		"""
		Set an upper bound, retaining existing lower bound.

		Args:
			key (str):
				The feature name to which this upper bound
				will be attached.
			value (numeric or None):
				The upper bound. Set explicitly to 'None' to
				leave unbounded from above.

		Raises:
			ScopeError:
				If a scope is attached to this box but the `key` cannot
				be found in the scope.
		"""
		current = self.thresholds.get(key, Bounds(None,None))
		if isinstance(current, set):
			raise ValueError("cannot set upperbound on a set")
		if self.scope is not None:
			if key in self.scope.get_all_names():
				self.thresholds[key] = Bounds(current.lowerbound, value)
			else:
				raise ScopeError(f"cannot set threshold on '{key}'")
		else:
			self.thresholds[key] = Bounds(current.lowerbound, value)

	def set_bounds(self, key, lowerbound, upperbound=None):
		"""
		Set both lower and upper bounds.

		Args:
			key (str):
				The feature name to which these bounds
				will be attached.
			lowerbound (numeric, None, or Bounds):
				The lower bound, or a Bounds object that gives
				upper and lower bounds (in which case the `upperbound`
				argument is ignored).  Set explicitly to 'None' to
				leave unbounded from below.
			upperbound (numeric or None, default None):
				The upper bound. Set to 'None' to
				leave unbounded from above.

		Raises:
			ScopeError:
				If a scope is attached to this box but the `key` cannot
				be found in the scope.

		"""

		if isinstance(lowerbound, Bounds):
			b = lowerbound
			lowerbound, upperbound = b.lowerbound, b.upperbound

		if self.scope is not None:
			if key in self.scope.get_all_names():
				self.thresholds[key] = Bounds(lowerbound, upperbound)
			else:
				raise ScopeError(f"cannot set threshold on '{key}'")
		else:
			self.thresholds[key] = Bounds(lowerbound, upperbound)

	def add_to_allowed_set(self, key, value):
		"""
		Add a value to the allowed set

		Args:
			key (str):
				The feature name to which these bounds
				will be attached.
			value (Any):
				A value to add to the allowed set.

		Raises:
			ValueError:
				If there is already a directional Bounds set for `key`.
			ScopeError:
				If a scope is attached to this box but the `key` cannot
				be found in the scope.
		"""
		current = self.thresholds.get(key, set())
		if isinstance(current, Bounds):
			raise ValueError("cannot add to Bounds")
		if self.scope is not None:
			if key in self.scope.get_all_names():
				current.add(value)
				self.thresholds[key] = current
			else:
				raise ScopeError(f"cannot set threshold on '{key}'")
		else:
			current.add(value)
			self.thresholds[key] = current

	def remove_from_allowed_set(self, key, value):
		"""
		Remove a value from the allowed set

		Args:
			key (str):
				The feature name to which these bounds
				will be attached.
			value (Any):
				A value to remove from the allowed set.

		Raises:
			ValueError:
				If the threshold set for `key` is a directional Bounds
				instead of a set.
			ScopeError:
				If a scope is attached to this box but the `key` cannot
				be found in the scope.
		"""
		current = self.thresholds.get(key, set())
		if isinstance(current, Bounds):
			raise ValueError("cannot remove from Bounds")
		if self.scope is not None:
			if key in self.scope.get_all_names():
				current.pop(value, None)
				self.thresholds[key] = current
			else:
				raise ScopeError(f"cannot set threshold on '{key}'")
		else:
			current.pop(value, None)
			self.thresholds[key] = current

	def replace_allowed_set(self, key, values):
		"""
		Replace the allowed set.

		Args:
			key (str):
				The feature name to which these bounds
				will be attached.
			values (set):
				A set of values to use as the allowed set.
		"""
		if self.scope is not None:
			if key in self.scope.get_all_names():
				self.thresholds[key] = set(values)
			else:
				raise ScopeError(f"cannot set threshold on '{key}'")
		else:
			self.thresholds[key] = set(values)



	def __setitem__(self, key, value):
		if not isinstance(value, (Bounds, set)):
			raise TypeError('thresholds must be Bounds or a set')
		if self.scope:
			if key in self.scope.get_all_names():
				self.thresholds[key] = value
			else:
				raise ScopeError("cannot set threshold on '{key}'")
		else:
			self.thresholds[key] = value

	def __iter__(self):
		return itertools.chain(
			iter(self.thresholds),
		)

	def __len__(self):
		return (
			len(self.thresholds)
		)

	def __repr__(self):
		if self.keys() or self.relevant_features:
			demands = list(self.keys()) or [" "]
			relevent = list(self.relevant_features) or [" "]
			m = max(
				max(map(len, demands)) + 1,
				max(map(len, relevent)) + 1
			)
			members = []
			for k, v in self.items():
				if isinstance(v, Bounds):
					if v.lowerbound is None:
						if v.upperbound is None:
							v_ = ' Unbounded'
						else:
							v_ = f' <= {v.upperbound}'
					else:
						if v.upperbound is None:
							v_ = f' >= {v.lowerbound}'
						else:
							v_ = f': {v.lowerbound} to {v.upperbound}'
				else:
					v_ = ': ' + repr(v)
				members.append("● "+k.rjust(m) + v_)
			for k in self.relevant_features:
				members.append("◌ "+k.rjust(m))

			head = f"{self.__class__.__name__}: {self.name}"
			return head+"\n   " + '\n   '.join(members)
		else:
			return "<empty "+ self.__class__.__name__ + ">"


class ChainedBox(Mapping, GenericBoxMixin):
	"""
	A Box defines a set of restricted dimensions for a Scope.

	Args:
		boxes (Boxes):
			A collection of Boxes from which to assemble a chain.
		name (str):
			The name for this ChainedBox.  This must be the name of
			a Box in `boxes`, which serves as the seed for the chain.
			Ancestors are added recursively by finding the parent box
			of each box in the chain, until a box is found with no parent.

	"""

	def __init__(self, boxes, name):
		"""

		Parameters
		----------
		boxes : Mapping
			Dictionary of {str:Box} pairs
		name : str
			Name of this chained box
		"""
		c = boxes[name]
		self.chain = [c]
		self.names = [name]
		while c.parent_box_name is not None:
			self.names.insert(0, c.parent_box_name)
			c = boxes[c.parent_box_name]
			self.chain.insert(0, c)

	def __getitem__(self, key):
		return self.thresholds[key]

	def __iter__(self):
		return itertools.chain(
			iter(self.thresholds),
		)

	def __len__(self):
		return len(self.thresholds)

	@property
	def name(self):
		"""str: The name of the last (defining) Box in this chain."""
		return self.names[-1]

	@property
	def thresholds(self):
		"""
		Dict[str,Union[Bounds,Set]]:
			The restricted dimensions in this ChainedBox, with feature names as
			keys and the Bounds or available set as the values.
		"""
		t = {}
		for single in self.chain:
			t.update(single.thresholds)
		return t

	def measure_thresholds(self):
		"""
		The thresholds in this Box or its ancestor(s) associated with performance measures.

		A Scope must be associated with each Box in the chain to access this property.

		Returns:
			Dict[str,Union[Bounds,Set]]
		"""
		t = {}
		for single in self.chain:
			t.update(single.measure_thresholds)
		return t

	def uncertainty_thresholds(self):
		"""
		The thresholds in this Box or its ancestor(s) associated with exogenous uncertainties.

		A Scope must be associated with each Box in the chain to access this property.

		Returns:
			Dict[str,Union[Bounds,Set]]
		"""
		t = {}
		for single in self.chain:
			t.update(single.uncertainty_thresholds)
		return t

	def lever_thresholds(self):
		"""
		The thresholds in this Box or its ancestor(s) associated with policy levers.

		A Scope must be associated with each Box in the chain to access this property.

		Returns:
			Dict[str,Union[Bounds,Set]]
		"""
		t = {}
		for single in self.chain:
			t.update(single.lever_thresholds)
		return t

	@property
	def relevant_features(self):
		"""
		Set[str]: A set of features that are relevant at any step of the chain.
		"""
		t = set()
		for single in self.chain:
			t |= single.relevant_features
		return t

	@property
	def demanded_features(self):
		"""
		Set[str]: A set of features upon which thresholds are set at any step of the chain.
		"""
		t = set()
		for single in self.chain:
			t |= set(single.thresholds.keys())
		return t

	@property
	def relevant_and_demanded_features(self):
		"""
		Set[str]: The union of relevant and demanded features.
		"""
		return self.relevant_features | self.demanded_features


	def __repr__(self):
		if self.keys() or self.relevant_features:
			demands = list(self.keys()) or [" "]
			relevent = list(self.relevant_features) or [" "]
			m = max(
				max(map(len, demands)) + 1,
				max(map(len, relevent)) + 1
			)
			members = []
			for k, v in self.items():
				if isinstance(v, Bounds):
					if v.lowerbound is None:
						if v.upperbound is None:
							v_ = ' Unbounded'
						else:
							v_ = f' <= {v.upperbound}'
					else:
						if v.upperbound is None:
							v_ = f' >= {v.lowerbound}'
						else:
							v_ = f': {v.lowerbound} to {v.upperbound}'
				else:
					v_ = ': ' + repr(v)
				members.append("● "+k.rjust(m) + v_)
			for k in self.relevant_features:
				members.append("◌ "+k.rjust(m))

			head = f"{self.__class__.__name__}: {self.name}"
			return head+"\n   " + '\n   '.join(members)
		else:
			return "<empty "+ self.__class__.__name__ + ">"


	def chain_repr(self):
		return "\n".join(f"{repr(c)}" for n,c in zip(self.names,self.chain))



def find_all_boxes_with_parent(universe:dict, parent=None):
	result = []
	for name, clusterdef in universe.items():
		if clusterdef.parent_box_name == parent:
			result.append(name)
	return result

def pseudoname_boxes(boxes, root=None):
	if root is None:
		try:
			fancy = [f"Scope: {boxes.scope.name}"]
		except AttributeError:
			fancy = [f"Boxes Universe"]
		plain = [None]
	else:
		fancy = []
		plain = []
	tops = sorted(find_all_boxes_with_parent(boxes, parent=root))
	for t in tops:
		fancy.append("▷ "+t if t[0] in "▶▷" else "▶ "+t)
		plain.append(t)
		f_, p_ = pseudoname_boxes(boxes, root=t)
		for f1, p1 in zip(f_, p_):
			fancy.append("▷ "+f1 if f1[0] in "▶▷" else "▶ "+f1)
			plain.append(p1)
	return fancy, plain

class Boxes(MutableMapping):

	def __init__(self, *args, scope=None, **kw):
		self._storage = dict()
		self._scope = scope
		if len(args) == 1 and isinstance(args[0], (list, tuple, set)):
			args = args[0]
		for a in args:
			self.add(a)
		for k,v in kw.items():
			self[k] = v
		if scope is not None:
			for i in self._storage:
				self._storage[i].scope = scope
	@property
	def scope(self):
		return self._scope

	@scope.setter
	def scope(self, s):
		self._scope = s
		for i in self._storage:
			self._storage[i].scope = s

	def __getitem__(self, key):
		return self._storage[key]

	def __setitem__(self, key, value):
		if not isinstance(value, Box):
			raise TypeError(f"values must be Box not {type(value)}")
		if key != value.name:
			raise ValueError('key must match name, use Boxes.add(box)')
		self._storage[key] = value

	def add(self, value):
		if not isinstance(value, Box):
			raise TypeError(f"values must be Box not {type(value)}")
		self[value.name] = value

	def __delitem__(self, key):
		del self._storage[key]

	def __iter__(self):
		return iter(self._storage)

	def __len__(self):
		return len(self._storage)

	def plain_names(self):
		return pseudoname_boxes(self, root=None)[1]

	def fancy_names(self):
		return pseudoname_boxes(self, root=None)[0]

	def both_names(self):
		return pseudoname_boxes(self, root=None)

	def get_chain(self, name):
		return ChainedBox(self, name)

