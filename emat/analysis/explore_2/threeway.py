
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ...viz import colors
from .components.three_dim_figure import three_dim_figure

def flex_scale(x):
    try:
        nan_min = np.nanmin(x)
        nan_range = np.nanmax(x) - nan_min
        x_ = (x - nan_min) / nan_range
        candidates ={
            'raw': x_,
            'square': x_**2,
            'sqrt': np.sqrt(x_),
        }
        candidate_scores = {
            k:np.abs(np.nanmean(v)-0.5)
            for k,v in candidates.items()
        }
        best_flex = min(candidate_scores, key=candidate_scores.get)
        return candidates[best_flex]
    except:
        return x





from ipywidgets import Dropdown, HBox, VBox, Label
from .twoway import BaseTwoWayFigure

CONST_MARKER_SIZE = "< Constant >"

class ThreeWayFigure(HBox, BaseTwoWayFigure):

    def __init__(
            self,
            viz,
            x=None,
            y=None,
            z=None,
            s=None,
    ):
        BaseTwoWayFigure.__init__(self, viz)
        axis_choices = self._get_sectional_names(self._dfv.data.columns)

        n_cols = len(self._dfv.data.columns)
        default_x = x or self._dfv.data.columns[0]
        default_y = y or self._dfv.data.columns[min(1,n_cols-1)]
        default_z = z or self._dfv.data.columns[min(2,n_cols-1)]
        default_s = s or CONST_MARKER_SIZE

        self.x_axis_choose = Dropdown(
            options=axis_choices,
            value=default_x,
        )

        self.y_axis_choose = Dropdown(
            options=axis_choices,
            value=default_y,
        )

        self.z_axis_choose = Dropdown(
            options=axis_choices,
            value=default_z,
        )

        self.s_axis_choose = Dropdown(
            options=[CONST_MARKER_SIZE]+axis_choices,
            value=default_s,
        )

        self.x_axis_choose.observe(self._observe_change_column_x, names='value')
        self.y_axis_choose.observe(self._observe_change_column_y, names='value')
        self.z_axis_choose.observe(self._observe_change_column_z, names='value')
        self.s_axis_choose.observe(self._observe_change_column_s, names='value')

        self.axis_choose = VBox(
            [
                Label("X Axis"),
                self.x_axis_choose,
                Label("Y Axis"),
                self.y_axis_choose,
                Label("Z Axis"),
                self.z_axis_choose,
                Label("Marker Size"),
                self.s_axis_choose,
                Label("Selection"),
                self._dfv._active_selection_chooser,
            ],
            layout=dict(
                overflow='hidden',
            )
        )
        if self._dfv.data.index.name:
            hover_name = self._dfv.data.index.name
        else:
            hover_name = "Experiment"
        self.hover_name = hover_name

        self.graph3d = go.FigureWidget(
            three_dim_figure(
                data=self._dfv.data,
                x=self.x_axis_choose.value,
                y=self.y_axis_choose.value,
                z=self.z_axis_choose.value,
                selection=self._dfv.active_selection(),
                scope=self.scope,
                hover_name=hover_name,
            )
        )
        self.set_xyz(
            x=self.x_axis_choose.value,
            y=self.y_axis_choose.value,
            z=self.z_axis_choose.value,
            s=self.s_axis_choose.value,
        )
        self.change_selection_color(self._dfv.active_selection_color())
        from ...util.naming import multiindex_to_strings

        self.graph3d.data[0]['meta'] = multiindex_to_strings(self._dfv.data.index)

        super().__init__(
            [
                self.graph3d,
                self.axis_choose,
            ],
            layout=dict(
                align_items='center',
            )
        )


    def _observe_change_column_x(self, payload):
        if payload['new'][:3] == '-- ' and payload['new'][-3:] == ' --':
            # Just a heading, not a real option
            payload['owner'].value = payload['old']
            return
        self.set_xyz(payload['new'], None, None)


    def _observe_change_column_y(self, payload):
        if payload['new'][:3] == '-- ' and payload['new'][-3:] == ' --':
            # Just a heading, not a real option
            payload['owner'].value = payload['old']
            return
        self.set_xyz(None, payload['new'], None)


    def _observe_change_column_z(self, payload):
        if payload['new'][:3] == '-- ' and payload['new'][-3:] == ' --':
            # Just a heading, not a real option
            payload['owner'].value = payload['old']
            return
        self.set_xyz(None, None, payload['new'])

    def _observe_change_column_s(self, payload):
        if payload['new'][:3] == '-- ' and payload['new'][-3:] == ' --':
            # Just a heading, not a real option
            payload['owner'].value = payload['old']
            return
        if payload['new'] == CONST_MARKER_SIZE:
            self.set_xyz(None, None, None, False)
        else:
            self.set_xyz(None, None, None, payload['new'])


    def set_xyz(self, x, y, z, s=None, drawbox=True):
        """
        Set the new X,Y,Z axis data and marker size S.

        Args:
            x,y,z,s (str or array-like):
                The name of the new columns in `df`, or a
                computed array or pandas.Series of values.
        """
        template_tags = dict(
            x=self.graph3d.layout['scene']['xaxis']['title']['text'],
            y=self.graph3d.layout['scene']['yaxis']['title']['text'],
            z=self.graph3d.layout['scene']['zaxis']['title']['text'],
        )
        try:
            with self.graph3d.batch_update():

                for _values, _ax in zip([x,y,z],['x','y','z']):
                    if _values is None:
                        continue
                    if isinstance(_values, str):
                        _label = self._get_shortname(_values)
                        _values = self._dfv.data[_values]
                    else:
                        try:
                            _label = self._get_shortname(_values.name)
                        except:
                            _label = _ax
                    _values, _ticktext, _tickvals, _scales = self._manage_categorical(_values)
                    setattr(self, f"_{_ax}", _values)
                    setattr(self, f"_{_ax}_ticktext", _ticktext or [])
                    setattr(self, f"_{_ax}_tickvals", _tickvals or [])
                    if _label is not None:
                        self.graph3d.layout['scene'][f'{_ax}axis']['title'] = {'text':_label}
                        template_tags[_ax] = _label
                    else:
                        self.graph3d.layout['scene'][f'{_ax}axis']['title'] = {}
                    self.graph3d.data[0][_ax] = _values
                    _data_range = [_values.min(), _values.max()]
                    setattr(self, f"_{_ax}_data_range", _data_range)
                    if _ticktext is not None:
                        self.graph3d.layout['scene'][f'{_ax}axis']['range'] = (
                            _data_range[0] - 0.3,
                            _data_range[1] + 0.3,
                        )
                        # self.graph3d.layout['scene'][f'{_ax}axis']['tickmode'] = 'array'
                        self.graph3d.layout['scene'][f'{_ax}axis']['ticktext'] = _ticktext
                        self.graph3d.layout['scene'][f'{_ax}axis']['tickvals'] = _tickvals
                    else:
                        _data_width = _data_range[1] - _data_range[0]
                        if _data_width <= 0:
                            _data_width = 1
                        self.graph3d.layout['scene'][f'{_ax}axis']['range'] = (
                            _data_range[0] - _data_width * 0.07,
                            _data_range[1] + _data_width * 0.07,
                        )
                        # self.graph3d.layout['scene'][f'{_ax}axis']['tickmode'] = None
                        self.graph3d.layout['scene'][f'{_ax}axis']['ticktext'] = None
                        self.graph3d.layout['scene'][f'{_ax}axis']['tickvals'] = None

                if s is False or s == CONST_MARKER_SIZE:
                    self.graph3d.data[0]['marker']['size'] = 8
                elif s is not None:
                    _values = s
                    if isinstance(_values, str):
                        _label = self._get_shortname(_values)
                        _values = self._dfv.data[_values]
                    else:
                        try:
                            _label = self._get_shortname(s.name)
                        except:
                            _label = _ax
                    _values = flex_scale(_values)
                    _values, _ticktext, _tickvals, _scales = self._manage_categorical(_values, perturb=False)
                    if _ticktext is not None:
                        _values = _values.map({
                            0: "x",
                            1: "circle",
                            2: "square",
                            3: "diamond",
                            4: "cross",
                            5: "diamond-open",
                            6: "square-open",
                            7: "circle-open",
                        })
                        self.graph3d.data[0]['marker']['symbol'] = _values
                        self.graph3d.data[0]['marker']['size'] = 8
                    else:
                        self.graph3d.data[0]['marker']['size'] = (_values+1).fillna(0) * 8
                        self.graph3d.data[0]['marker']['symbol'] = "circle"

                hovertemplate = f"{template_tags['x']}: %{{x}}<br>" \
                                f"{template_tags['y']}: %{{y}}<br>" \
                                f"{template_tags['z']}: %{{z}}" \
                                f"<extra>{self.hover_name} %{{meta}}</extra>"
                self.graph3d.data[0]['hovertemplate'] = hovertemplate

                if drawbox:
                    self.draw_box()
        except:
            #_logger.exception('ERROR IN DataFrameViewer.set_x')
            raise


    def change_selection(self, new_selection, new_color=None):
        if new_selection is None:
            self.selection = None
            with self.graph3d.batch_update():
                self.graph3d.data[0]['marker']['color'] = np.zeros(len(self._dfv.data), np.int8)
                #self.draw_box()
            return

        if new_selection.size != len(self._dfv.data):
            raise ValueError(f"new selection size ({new_selection.size}) "
                             f"does not match length of data ({len(self._dfv.data)})")
        # self.selection = new_selection
        with self.graph3d.batch_update():
            selection_as_int = self.selection.astype(int)
            self.graph3d.data[0]['marker']['color'] = selection_as_int
            if new_color is not None:
                self.change_selection_color(new_color)
            self.draw_box()

    def change_selection_color(self, new_color=None):
        if new_color is None:
            new_color = colors.DEFAULT_HIGHLIGHT_COLOR
        with self.graph3d.batch_update():
            self.graph3d.data[0]['marker']['colorscale'] = [[0, colors.DEFAULT_BASE_COLOR], [1, new_color]]

    def draw_box(self):
        pass