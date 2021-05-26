
import numpy
import pandas
import operator

from ..workbench.analysis import prim
from ..workbench.analysis.scenario_discovery_util import RuleInductionType
from ..workbench.analysis.prim_util import PrimException
from ..scope.box import Box, Bounds, Boxes
from ..scope.scope import Scope
from .discovery import ScenarioDiscoveryMixin


from plotly import graph_objects as go

class Prim(prim.Prim, ScenarioDiscoveryMixin):
    """
    Patient rule induction method

    This implementation of Prim is derived from
    the EMA workbench, and is enhanced to work
    interactively within the TMIP-EMAT framework.

    Args:
        x (pandas.DataFrame):
            The independent variables, generally the
            scoped input parameters (uncertainties
            and/or policy levers) from a design of
            experiments.
        y (array-like):
            The dependent variable, generally a
            pandas.Series or other one dimensional
            array with size equal to the number of
            rows in `x`.
        threshold (float):
            The density threshold that a box must meet.
        obj_function ({LENIENT1, LENIENT2, ORIGINAL}):
            The objective function used by PRIM. This
            defaults to a lenient objective function
            based on the gain of mean divided by the
            loss of mass.
        peel_alpha (float, default 0.05):
            The parameter controlling the peeling stage.
        paste_alpha (float, default 0.05):
            The parameter controlling the pasting stage.
        mass_min (float, default 0.05):
            The minimum mass of a box.
        threshold_type ({ABOVE, BELOW}):
            Whether to look above or below the threshold
            value.
        mode (RuleInductionType):
            This PRIM implementation defaults to binary
            classification, but this argument can be used
            to switch to regression mode.  In binary mode,
            the `y` array should only have binary values,
            while in regression mode it can contain other
            floating point values.
        update_function ({'default', 'guivarch'}):
            The default behavior for this algorithm
            after finding the first box is to
            remove all points in the first box,
            so that subsequent boxes do not include
            them.  An alternate procedure suggested by
            Guivarch et al (2016)
            doi:10.1016/j.envsoft.2016.03.006 is to simply
            set all points in the first box to be no longer
            of interest. This alternate procedure is only
            valid in binary mode.


    """

    def find_box(self):
        """
        Execute one iteration of the PRIM algorithm.

        This method will find one box, starting from the
        current state of Prim (i.e., over all observations
        that are not already within a previously selected
        PrimBox).  All existing boxes will be "frozen" by
        this command, so that those prior boxes can no
        longer be modified by their `select` methods or
        from within their `tradeoff_selector`.

        Returns:
            emat.analysis.PrimBox
        """
        primbox = super().find_box()
        if primbox is None:
            return None
        primbox.__class__ = PrimBox
        prim_explorer = getattr(self, '_explorer', None)
        if prim_explorer is not None:
            from .explore_2.explore_visualizer import Visualizer
            subdata = prim_explorer.data.iloc[primbox.yi_initial]
            if len(subdata) != len(prim_explorer.data):
                # Generally, second and later boxes get new subvisualizers
                primbox._explorer = Visualizer(
                    subdata.copy(),
                    scope=prim_explorer.scope,
                )
            else:
                primbox._explorer = prim_explorer
        primbox._target_name = getattr(self, '_target_name', None)
        primbox.select(primbox._cur_box)
        return primbox

    def tradeoff_selector(self, n=-1, colorscale='viridis', figure_class=None):
        """
        Visualize the trade off between coverage and density.

        This visualization plots all of the points along
        the peeling trajectory for the selected `PrimBox`,
        plotting coverage along the x axis, density along the
        y axis, and showing the number of restricted dimensions
        by color.

        Coverage is percentage of the cases of interest that
        are in the box (i.e., number of cases of interest in
        the box divided by total number of cases of interest).
        The starting point of the PRIM algorithm is the
        unrestricted full set of cases, which includes all
        outcomes of interest, and therefore, the coverage starts
        at 1.0 and drops as the algorithm progresses.

        Density is the share of cases in the box that are case
        of interest (i.e., number of cases of interest in the
        box divided by the total number of cases in the box).
        As the box is reduced, the density will increase (as
        that is the objective of the PRIM algorithm).

        Args:
            n (int, optional):
                The index number of the PrimBox to use.  If not
                given, the last found box is used.  If no boxes
                have been found yet, an initial box is found
                using the `find_box` method, and in this case
                giving any value other than -1 will raise an error.
            colorscale (str, default 'viridis'):
                A valid color scale name, as compatible with the
                color_palette method in seaborn.

        Returns:
            plotly.FigureWidget
        """
        try:
            box = self._boxes[n]
        except IndexError:
            if n == -1:
                box = self.find_box()
            else:
                raise
        return box.tradeoff_selector(colorscale=colorscale, figure_class=figure_class)

    def to_json(self, n=-1):
        try:
            box = self._boxes[n]
        except IndexError:
            if n == -1:
                box = self.find_box()
            else:
                raise
        return box.to_json()

    def __init__(self, x, y, threshold=0.05, *args, scope=None, explorer=None, **kwargs):
        super().__init__(x, y, threshold, *args, **kwargs)
        self._target_name = getattr(y, 'name', None)
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

def _discrete_color_scale(name='viridis', n=8):
    import seaborn as sns
    colors = sns.color_palette(name, n)
    colorlist = []
    for i in range(n):
        c = colors[i]
        thiscolor_s = f"rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})"
        colorlist.append([i/n, thiscolor_s])
        colorlist.append([(i+1)/n, thiscolor_s])
    return colorlist


class PrimBox(prim.PrimBox):
    """
    Information for a specific Prim box.

    By default, the currently selected box
    is the last box on the peeling trajectory,
    unless this is changed via :meth:`PrimBox.select`.
    """

    def box_number(self):
        """
        The number of PRIM iterations up to and including this PrimBox.

        Returns:
            int
        """
        for n, b in enumerate(self.prim._boxes, start=1):
            if b is self:
                return n
        return -1

    def to_emat_box(self, i=None, name=None, src_name=None):
        """
        Convert the current PrimBox to a standard emat.Box.

        Returns:
            emat.Box
        """
        if i is None:
            i = self._cur_box

        if name is None:
            name = f'PRIM Box {self.box_number()} Peel {i}'
            if src_name is not None:
                name = name + f" [{src_name}]"

        limits = self.box_lims[i]

        b = Box(name)

        for col in limits.columns:
            if isinstance(self.prim.x.dtypes[col], pandas.CategoricalDtype):
                if set(self.prim.x[col].cat.categories) != limits[col].iloc[0]:
                    b.replace_allowed_set(col, limits[col].iloc[0])
            else:
                if limits[col].iloc[0] != self.prim.x[col].min():
                    b.set_lower_bound(col, limits[col].iloc[0])
                if limits[col].iloc[1] != self.prim.x[col].max():
                    b.set_upper_bound(col, limits[col].iloc[1])
        b.coverage = self.peeling_trajectory['coverage'][i]
        b.density = self.peeling_trajectory['density'][i]
        b.mass = self.peeling_trajectory['mass'][i]
        return b

    def __repr__(self):
        i = self._cur_box
        head = f"<{self.__class__.__name__} peel {i+1} of {len(self.peeling_trajectory)}>"

        # make the box definition
        qp_values = self.qp[i]
        uncs = [(key, value) for key, value in qp_values.items()]
        uncs.sort(key=operator.itemgetter(1))
        uncs = [uncs[0] for uncs in uncs]
        box_lim = pandas.DataFrame( index=uncs, columns=['min','max'])
        for unc in uncs:
            values = self.box_lims[i][unc]
            box_lim.loc[unc] = [values[0], values[1]]
        head += f'\n   coverage: {self.coverage:.5f}'
        head += f'\n   density:  {self.density:.5f}'
        head += f'\n   mean: {self.mean:.5f}'
        head += f'\n   mass: {self.mass:.5f}'
        head += f'\n   restricted dims: {self.res_dim}'
        if not box_lim.empty:
            head += "\n     "+str(box_lim).replace("\n", "\n     ")
        return head

    def to_json(self):
        """
        Write out the current PrimBox in json format.

        Returns:
            str
        """
        state = {}
        for i in range(len(self.peeling_trajectory)):
            state[i] = self.to_emat_box(i, name=str(i)).to_json()
        import json
        return json.dumps(state)

    def _make_tradeoff_selector(self, colorscale='cividis', figure_class=None):
        """
        Visualize the trade off between coverage and density. Color
        is used to denote the number of restricted dimensions.

        Args:
            colorscale (str, default 'cividis'):
                A valid seaborn color scale name.

        Returns:
            plotly.FigureWidget
        """

        peeling_trajectory = self.peeling_trajectory

        hovertext = pandas.Series('', index=peeling_trajectory.index)

        if figure_class is None:
            figure_class = go.FigureWidget
        fig = figure_class()

        for i in range(len(peeling_trajectory)):
            t = str(self.to_emat_box(i, name=str(i))).replace("\n","<br>")
            hovertext.iloc[i] = f'<span style="font-family:Consolas,monospace">{t}</span>'

        n_colors = max(peeling_trajectory['res_dim'])+1
        color_scale_ = _discrete_color_scale(colorscale, n_colors)
        colortickvals = numpy.arange(0.5, n_colors, 1) * (n_colors-1)/n_colors
        colorticktext = [str(i) for i in range(n_colors)]

        symbols = numpy.zeros(len(peeling_trajectory), dtype=int)
        symbols[self._cur_box] = 4 # cross
        sizes = numpy.full(len(peeling_trajectory), 6, dtype=int)
        sizes[self._cur_box] = 9

        scatter = fig.add_scatter(
            x=peeling_trajectory['coverage'],
            y=peeling_trajectory['density'],
            mode='markers',
            marker=dict(
                color=peeling_trajectory['res_dim'],
                colorscale=color_scale_,
                showscale=True,
                colorbar=dict(
                    title="Number of Restricted Dimensions",
                    titleside="right",
                    tickmode="array",
                    tickvals=colortickvals,
                    ticktext=colorticktext,
                    ticks="outside",
                ),
                symbol=symbols,
                size=sizes,
            ),
            text=hovertext,
            hoverinfo="text",
        ).data[-1]

        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            width=600,
            height=400,
            xaxis_title_text='Coverage',
            yaxis_title_text='Density',
        )

        # create callback function
        def select_point(trace, points, selector):
            for i in points.point_inds:
                self.select(i)

        scatter.on_click(select_point)

        return fig

    def tradeoff_selector(self, colorscale='viridis', figure_class=None):
        '''
        Visualize the trade off between coverage and density.

        This visualization plots all of the points along
        the peeling trajectory for this `PrimBox`,
        plotting coverage along the x axis, density along the
        y axis, and showing the number of restricted dimensions
        by color.

        Coverage is percentage of the cases of interest that
        are in the box (i.e., number of cases of interest in
        the box divided by total number of cases of interest).
        The starting point of the PRIM algorithm is the
        unrestricted full set of cases, which includes all
        outcomes of interest, and therefore, the coverage starts
        at 1.0 and drops as the algorithm progresses.

        Density is the share of cases in the box that are case
        of interest (i.e., number of cases of interest in the
        box divided by the total number of cases in the box).
        As the box is reduced, the density will increase (as
        that is the objective of the PRIM algorithm).

        Parameters
        ----------
        colorscale : str, default 'viridis'
            A valid color scale name, as compatible with the
            color_palette method in seaborn.

        Returns
        -------
        FigureWidget
        '''
        if getattr(self, '_tradeoff_widget', None) is None:
            self._tradeoff_widget = self._make_tradeoff_selector(
                colorscale=colorscale,
                figure_class=figure_class,
            )
        return self._tradeoff_widget

    def select(self, i):
        """
        Select an entry from the peeling and pasting trajectory.

        This will update the PRIM box to this selected box, as
        well as update the `tradeoff_selector` and `explorer`,
        if either is attached to this PrimBox.

        Args:
            i (int):
                The index of the box to select.
        """
        try:
            super(PrimBox, self).select(i)
        except PrimException:
            pass
        else:
            widget = getattr(self, '_tradeoff_widget', None)
            if widget is not None:
                symbols = numpy.zeros(len(self.peeling_trajectory), dtype=int)
                symbols[i] = 4  # cross
                sizes = numpy.full(len(self.peeling_trajectory), 6, dtype=int)
                sizes[i] = 9
                widget['data'][0]['marker']['symbol'] = symbols
                widget['data'][0]['marker']['size'] = sizes
            explorer = getattr(self, '_explorer', None)
            if explorer is not None:
                from .explore_2.explore_base import DataFrameExplorerBase
                if isinstance(explorer, DataFrameExplorerBase):
                    name_t = f"PRIM Box {self.box_number()} Target [{self._target_name}]"
                    name_s = f"PRIM Box {self.box_number()} Solution [{self._target_name}]"
                    explorer.new_selection(
                        self,
                        name=name_t,
                        activate=False,
                    )
                    explorer.new_selection(
                        self.to_emat_box(),
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
                    explorer.set_box(self.to_emat_box())

    def explore(self, scope=None, data=None):
        """
        Create or access the existing visualizer.

        Args:
            scope (Scope, optional):
                Override the exploratory scope for this Box.
            data (DataFrame, optional):
                Override the data for this Box.

        Returns:
            emat.Visualizer
        """
        if getattr(self, '_explorer', None) is None:
            from .explore_2.explore_visualizer import Visualizer
            if data is None:
                data = self.prim.x
            if scope is None:
                scope = getattr(data, 'scope', None)
            if scope is None:
                raise ValueError("failed to initialize visualizer, cannot find scope")
            self._explorer = Visualizer(scope=scope, data=data)
            self._explorer["PRIM Target"] = self.to_emat_box()
        return self._explorer

    def splom(self, rows=None, cols=None):
        """
        Generate a scatter plot matrix showing this PrimBox.

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
