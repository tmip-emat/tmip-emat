
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ....viz import colors

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

def three_dim_figure(
        data,
        x,
        y,
        z,
        selection=None,
        scope=None,
        hover_name=None,
        mass=300,
        maximum_marker_opacity=0.9,
        minimum_marker_opacity=0.01,
):
    """

    Args:
        data:
        x:
        y:
        z:
        selection (pd.Series[Bool]):
            The active selection.
        scope (emat.Scope, optional):
            Used to label axes with shortnames where
            defined instead of identifier labels.
    Returns:

    """
    unselected_color = colors.DEFAULT_BASE_COLOR
    selected_color = colors.DEFAULT_HIGHLIGHT_COLOR

    if scope is None:
        from ....scope.scope import Scope
        scope = Scope("")

    marker_color = pd.Series(data=unselected_color, index=data.index)
    if selection is not None:
        marker_color[selection] = selected_color

    marker_opacity = np.clip(
        mass/len(data),
        minimum_marker_opacity,
        maximum_marker_opacity,
    )

    scene_bgcolor = 'rgb(255,255,255,0)'
    scene_gridcolor = '#E5ECF6'

    hovertemplate = f"{scope.shortname(x)}: %{{x}}<br>" \
                    f"{scope.shortname(y)}: %{{y}}<br>" \
                    f"{scope.shortname(z)}: %{{z}}" \
                    f"<extra>{hover_name} %{{meta}}</extra>"

    fig = go.Figure(
        go.Scatter3d(
            x=data[x],
            y=data[y],
            z=data[z],
            mode='markers',
            marker=dict(
                size=8,
                color=marker_color,
                opacity=marker_opacity,
            ),
            hovertemplate=hovertemplate,
        ),
        layout=dict(
            margin=dict(
                l=10, r=10, t=0, b=0,
            ),
            scene={
                'xaxis': {
                    'backgroundcolor': scene_bgcolor,
                    'gridcolor': scene_gridcolor,
                    'gridwidth': 2,
                    'linecolor': scene_gridcolor,
                    'showbackground': False,
                    'ticks': '',
                    'zerolinecolor': scene_gridcolor,
                    'title': {'text': scope.shortname(x)},
                },
                'yaxis': {
                    'backgroundcolor': scene_bgcolor,
                    'gridcolor': scene_gridcolor,
                    'gridwidth': 2,
                    'linecolor': scene_gridcolor,
                    'showbackground': False,
                    'ticks': '',
                    'zerolinecolor': scene_gridcolor,
                    'title': {'text': scope.shortname(y)},
                },
                'zaxis': {
                    'backgroundcolor': scene_bgcolor,
                    'gridcolor': scene_gridcolor,
                    'gridwidth': 2,
                    'linecolor': scene_gridcolor,
                    'showbackground': False,
                    'ticks': '',
                    'zerolinecolor': scene_gridcolor,
                    'title': {'text': scope.shortname(z)},
                },
                'aspectmode': 'cube',
            },
        ),

    )
    return fig


