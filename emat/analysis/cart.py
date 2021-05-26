import numpy as np
import pandas as pd
from ..scope.box import Box
from ..workbench.analysis import cart
from ..workbench.analysis import scenario_discovery_util as sdutil

class CART(cart.CART):
    """
    Classification and Regression Tree Algorithm

    CART can be used in a manner similar to PRIM. It provides access
    to the underlying tree, but it can also show the boxes described by the
    tree in a table or graph form similar to prim.

    Args:
        x (DataFrame):
            The independent variables, generally the experimental
            design inputs.
        y (array-like, 1 dimension):
            The dependent variable of interest.
        mass_min (float, default 0.05):
            A value between 0 and 1 indicating the minimum fraction
            of data points in a terminal leaf.
        mode ({BINARY, CLASSIFICATION, REGRESSION}):
            Indicates the mode in which CART is used. Binary indicates
            binary classification, classification is multiclass, and regression
            is regression.
        scope (Scope):
            The EMAT exploratory scope, used primarily to facilitate
            visualization.

    """

    def __init__(self, x, y, mass_min=0.05, mode=sdutil.RuleInductionType.BINARY, scope=None, explorer=None):
        super().__init__(x, y, mass_min, mode=mode)
        self._target_name = getattr(y, 'name', None)
        self.scope = scope
        if explorer is not None:
            self._explorer = explorer
        elif explorer is False:
            self._explorer = None
        else:
            self._explorer = None
            if hasattr(x, 'scope') and scope is None:
                scope = x.scope
            if scope is not None:
                from .explore_2.explore_visualizer import Visualizer
                self._explorer = Visualizer(x, scope=scope)

    def select(self, i):
        """
        Select a leaf from the CART tree.

        This will update the CART box to this selected box, as
        well as update the `explorer`, if one is attached to this CART.

        Args:
            i (int): The index of the box to select.
        """
        out = CartBox(self, i)

        explorer = getattr(self, '_explorer', None)
        if explorer is not None:
            from .explore_2.explore_base import DataFrameExplorerBase
            if isinstance(explorer, DataFrameExplorerBase):
                name_t = f"CART Box {i} Target [{self._target_name}]"
                name_s = f"CART Box {i} Solution [{self._target_name}]"
                explorer.new_selection(
                    out,
                    name=name_t,
                    activate=False,
                )
                explorer.new_selection(
                    out.to_emat_box(),
                    name=name_s,
                    activate=False,
                )
                if explorer.active_selection_name() not in (name_t, name_s):
                    explorer.set_active_selection_name(name_t)
                else:
                    explorer.set_active_selection_name(
                        explorer.active_selection_name(),
                        force_update=True,
                    )
            else:
                # for old explorer interface
                explorer.set_box(out.to_emat_box())

        return out

class _CartEntry(object):
    '''a descriptor for the current leaf'''

    # def __init__(self, name):
    #     self.name = name

    def __set_name__(self, cls, name):
        self.name = name

    def __get__(self, instance, _):
        try:
            return instance.cart_alg.stats[instance._cur_box][self.name]
        except KeyError:
            return instance.cart_alg.stats[instance._cur_box][self.name.replace("_"," ")]

    def __set__(self, instance, value):
        raise ValueError("this property cannot be assigned to")


class CartBox():
    """
    Information for a specific CART box, corresponding to a leaf of the tree.
    """

    coverage = _CartEntry()
    density = _CartEntry()
    res_dim = _CartEntry()
    mass = _CartEntry()

    def __init__(self, cart_alg, n=0):
        self.cart_alg = cart_alg
        self._cur_box = n

    def to_emat_box(self, i=None, name=None, src_name=None):
        if i is None:
            i = self._cur_box
        if name is None:
            name = f'CART Box {i}'
            if src_name is not None:
                name = name + f" [{src_name}]"
        limits = self.cart_alg.boxes[i]
        b = Box(name)
        for col in limits.columns:
            if isinstance(self.cart_alg.x.dtypes[col], pd.CategoricalDtype):
                if set(self.cart_alg.x[col].cat.categories) != limits[col].iloc[0]:
                    b.replace_allowed_set(col, limits[col].iloc[0])
            else:
                if limits[col].iloc[0] != self.cart_alg.x[col].min():
                    b.set_lower_bound(col, limits[col].iloc[0])
                if limits[col].iloc[1] != self.cart_alg.x[col].max():
                    b.set_upper_bound(col, limits[col].iloc[1])
        b.coverage = self.cart_alg.stats[i]['coverage']
        b.density = self.cart_alg.stats[i]['density']
        b.mass = self.cart_alg.stats[i]['mass']
        return b

    def __repr__(self):
        i = self._cur_box
        head = f"<{self.__class__.__name__} leaf {i} of {len(self.cart_alg.boxes)}>\n"
        return head + repr(self.to_emat_box()).split("\n", 1)[1]

    def to_json(self):
        state = {}
        for i in range(len(self.cart_alg.boxes)):
            state[i] = self.to_emat_box(i, name=str(i)).to_json()
        import json
        return json.dumps(state)

    @property
    def _explorer(self):
        return self.cart_alg._explorer

    @_explorer.setter
    def _explorer(self, x):
        self.cart_alg._explorer = x

    def explore(self, scope=None, data=None):
        if getattr(self, '_explorer', None) is None:
            from .explore_2.explore_visualizer import Visualizer
            if data is None:
                data = self.cart_alg.x
            if scope is None:
                scope = getattr(data, 'scope', None)
            if scope is None:
                scope = getattr(self.cart_alg, 'scope', None)
            if scope is None:
                raise ValueError("failed to initialize visualizer, cannot find scope")
            self._explorer = Visualizer(scope=scope, data=data)
            self._explorer["CART Target"] = self.to_emat_box()
        return self._explorer

    def splom(self, rows=None, cols=None):
        """
        Generate a scatter plot matrix showing this CartBox.

        Args:
            rows, cols (list-like, optional):
                The dimensions to display as rows and columns of the
                resulting scatter plot matrix.  If not provided, each
                defaults to the set of restricted dimensions on the
                current CartBox.

        Returns:
            plotly.FigureWidget
        """
        if rows is None:
            rows = sorted(self.to_emat_box().demanded_features)
        if cols is None:
            cols = sorted(self.to_emat_box().demanded_features)
        fig = self.explore().splom(
            f"{rows}|{cols}",
            rows=rows,
            cols=cols,
        )
        return fig

    def hmm(self, rows=None, cols=None):
        """
        Generate a heatmap matrix showing this CartBox.

        Args:
            rows, cols (list-like, optional):
                The dimensions to display as rows and columns of the
                resulting heatmap matrix.  If not provided, each
                defaults to the set of restricted dimensions on the
                current CartBox.

        Returns:
            plotly.FigureWidget
        """
        if rows is None:
            rows = sorted(self.to_emat_box().demanded_features)
        if cols is None:
            cols = sorted(self.to_emat_box().demanded_features)
        fig = self.explore().hmm(
            f"{rows}|{cols}",
            rows=rows,
            cols=cols,
        )
        return fig

