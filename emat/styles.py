
try:
    import ipywidgets as widgets
except ImportError:
    slider_layout = None
    togglebuttons_layout = None
else:
    slider_layout = widgets.Layout(
        width='250px',
    )
    togglebuttons_layout = widgets.Layout(

    )

slider_style = {
    # 'description_width': '150px',
    'min_width': '250px',
}

figure_dims = dict(
    width=300,
    height=175,
)

figure_margins = dict(
    l=10, r=10, t=40, b=10,
)

widget_frame = dict(
    align_items = 'center',
    border='1px solid #AAA',
    width='270px',
    margin='2px',
)