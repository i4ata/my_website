"""This script launches an interactive environment for the Mandelbro Set"""

from dash import dcc, html, Input, Output, State, ctx, callback, register_page
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from pages.mandelbrot.mandelbrot import mandelbrot_set
from utils import slider, updatemenu

with open('pages/mandelbrot/text.md') as f:
    text = f.read()

register_page(__name__, path='/mandelbrot')

# LAYOUT OF THE APP
layout = html.Div([

    # The text
    dcc.Markdown(text, mathjax=True),
    
    # Parameters
    html.Div([
        html.Label('Height:'), 
        dcc.Input(id='height', type='number', value=500, step=10, min=100)
    ]),
    html.Div([
        html.Label('Width: '),
        dcc.Input(id='width', type='number', value=500, step=10, min=100)
    ]),
    html.Div([
        html.Label('Maximum Iterations: '),
        dcc.Input(id='iterations', type='number', value=50, step=10, min=10, max=100)
    ]),
    
    # The Mandelbrot Set image as a heatmap
    dcc.Loading(id='loading', children=dcc.Graph(id='heatmap')),
    
    # Buttons for generating and resetting the Mandelbrot Set heatmap
    html.Button('Generate', id='generate'),
    html.Button('Reset', id='reset'),

    # The graph for the evolution of a specific number c
    html.Div(id='graph_evolution')
])

# TODO: This callback is called twice when loading the page for the first time
@callback(
    Output('heatmap', 'figure'),
    State('height', 'value'),
    State('width', 'value'),
    State('iterations', 'value'),
    Input('heatmap', 'relayoutData'),
    Input('generate', 'n_clicks'),
    Input('reset', 'n_clicks')
)
def generate(height, width, iterations, data, generate, reset):
    """
    Call to generate the Mandelbrot Set heatmap
    """

    # Just a check
    if height is None or width is None or iterations is None:
        return None

    # Define the range of the set
    x_min, x_max, y_min, y_max = (

        # Default. If the graph is reset or on the first call
        (-2., 1., -1.5, 1.5)
        
        if data is None or 'autosize' in data or ctx.triggered_id == 'reset' else 
        
        # New range when zooming in
        (data['xaxis.range[0]'], data['xaxis.range[1]'], data['yaxis.range[1]'], data['yaxis.range[0]'])
    )
    
    # Generate the set. Save all iterations as frames
    mandelbrot = mandelbrot_set(
        range_real=(x_min, x_max),
        range_imag=(y_min, y_max),
        dims=(height, width),
        max_iter=iterations,
        save_all=True
    )

    # Draw the heatmap
    fig = px.imshow(
        mandelbrot,
        animation_frame=0,
        x=np.linspace(x_min, x_max, width),
        y=np.linspace(y_min, y_max, height),
        aspect='equal',
        origin='lower',
        color_continuous_scale='Hot',
        labels={'x': 'Re(c)', 'y': 'Im(c)', 'color': 'Iterations'},
        title='Mandelbrot Set Visualization',
        range_color=[0, iterations],
        contrast_rescaling='minmax'
    )
    fig.update_layout(coloraxis_showscale=False, autosize=True, height=1000, width=1000)
    
    fig.layout.sliders[0]['currentvalue']['prefix'] = 'Iteration: '
    fig.layout.updatemenus[0]['buttons'][0]['label'] = 'Play'
    fig.layout.updatemenus[0]['buttons'][1]['label'] = 'Pause'
    return fig

@callback(
    Output('graph_evolution', 'children'),
    Input('heatmap', 'clickData'),
    Input('iterations', 'value')
)
def click_on_image(clickData, iterations):
    """Call to observe the evolution of a single number c"""

    if clickData is None or iterations is None:
        return None
    
    # Generate the trajectory for the selected number
    # TODO: Technically redundant to do since all trajectories are already calculated
    c = complex(clickData['points'][0]['x'], clickData['points'][0]['y'])
    z, z_real, z_imag, z_magnitudes = 0, [0], [0], [0]
    for i in range(iterations):
        z = z ** 2 + c
        z_magnitude = abs(z)

        z_real.append(z.real)
        z_imag.append(z.imag)
        z_magnitudes.append(z_magnitude)
        if z_magnitude > 2.: break

    # Plot the 3 subplots, [Re(z), Im(z), |z|]
    data = pd.DataFrame({
        'Re(z)': z_real, 
        'Im(z)': z_imag, 
        '|z|': z_magnitudes, 
        'Iteration': range(len(z_real))
    })
    fig = _plot(c, data, iterations)
    return dcc.Graph(id=f'graph_{c}', figure=fig)

def _plot(c: complex, data: pd.DataFrame, iterations: int) -> go.Figure:
    
    fig = make_subplots(cols=3)
    
    # DRAW LINES + CIRCLE
    hovertemplates = (
        'Iteration: %{x}<br>Re(z): %{y}<extra></extra>',
        'Iteration: %{x}<br>Im(z): %{y}<extra></extra>',
        'Iteration: %{pointNumber}<br>Re(z): %{x}<br>Im(z): %{y}<extra></extra>'
    )
    line = dict(color='blue')
    fig.add_traces(
        data=[
            go.Scatter(
                x=data['Iteration'], y=data['Re(z)'], 
                mode='lines', line=line, name='Re(z)', 
                hovertemplate=hovertemplates[0]
            ),
            go.Scatter(
                x=data['Iteration'], y=data['Im(z)'], 
                mode='lines', line=line, name='Im(z)',
                hovertemplate=hovertemplates[1]
            ),
            go.Scatter(
                x=data['Re(z)'], y=data['Im(z)'], 
                mode='lines', line=line, name='z',
                hovertemplate=hovertemplates[2]
            )
        ],
        rows=1,
        cols=[1,2,3]
    )
    fig.add_shape(type='circle', x0=-2, y0=-2, x1=2, y1=2, row=1, col=3)

    # SET RANGES
    fig.update_xaxes(range=(0, iterations), title='Iteration', row=1, col=1)
    fig.update_xaxes(range=(0, iterations), title='Iteration', row=1, col=2)

    x_range = (min(-2, data['Re(z)'].min()), max(2, data['Re(z)'].max()))
    margin = max(abs(x_range[0]), abs(x_range[1])) * .1
    fig.update_xaxes(range=(x_range[0] - margin, x_range[1] + margin), constrain='domain', title='Re(z)', row=1, col=3)
    
    y_range_1_2 = (data[['Re(z)', 'Im(z)']].min().min(), data[['Re(z)', 'Im(z)']].max().max())
    margin = max(abs(y_range_1_2[0]), abs(y_range_1_2[1])) * .1
    fig.update_yaxes(range=(y_range_1_2[0] - margin, y_range_1_2[1] + margin), title='Re(z)', row=1, col=1)
    fig.update_yaxes(range=(y_range_1_2[0] - margin, y_range_1_2[1] + margin), title='Im(z)', row=1, col=2)

    y_range_3 = (min(-2, data['Im(z)'].min()), max(2, data['Im(z)'].max()))
    margin = max(abs(y_range_3[0]), abs(y_range_3[1])) * .1
    fig.update_yaxes(range=(y_range_3[0] - margin, y_range_3[1] + margin), scaleanchor='x3', title='Im(z)', row=1, col=3)

    # ANIMATION
    marker = dict(color='orange', size=10)
    
    fig.add_traces(
        data=[
            go.Scatter(
                x=[0], y=[0], mode='markers', marker=marker, 
                hovertemplate=hovertemplates[subplot_index]
            )
            for subplot_index in range(3)
        ],
        rows=1, cols=[1,2,3]
    )
     
    fig.frames = [
        go.Frame(
            data=[
                {'x': [data['Iteration'][i]], 'y': [data['Re(z)'][i]]},
                {'x': [data['Iteration'][i]], 'y': [data['Im(z)'][i]]},
                {'x': [data['Re(z)'][i]], 'y': [data['Im(z)'][i]]}
            ],
            traces=[3, 4, 5],
            name=str(i)
        )
        for i in range(len(data))
    ]

    fig.update_layout(
        title=f'Evolution of c={c}',
        showlegend=False,
        updatemenus=[updatemenu],
        sliders=[slider(len(data))]
    )

    return fig
