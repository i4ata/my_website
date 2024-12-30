from dash import Dash, dcc, html, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from mandelbrot import mandelbrot_set

app = Dash(__name__)
server = app.server
app.layout = html.Div([
    dcc.Input(id='height', type='number', value=500, step=10, min=100),
    dcc.Input(id='width', type='number', value=500, step=10, min=100),
    dcc.Input(id='iterations', type='number', value=50, step=10, min=10, max=100),
    dcc.Loading(id='loading', children=dcc.Graph(id='heatmap')),
    html.Button('Generate', id='generate'),
    html.Button('Reset', id='reset'),
    html.Div(id='graph')
])

@app.callback(
    Output('heatmap', 'figure'),
    State('height', 'value'),
    State('width', 'value'),
    State('iterations', 'value'),
    Input('heatmap', 'relayoutData'),
    Input('generate', 'n_clicks'),
    Input('reset', 'n_clicks')
)
def generate(height, width, iterations, data, generate, reset):
    
    if height is None or width is None or iterations is None:
        return None

    x_min, x_max, y_min, y_max = (
        (-2., 1., -1.5, 1.5)
        if data is None or 'autosize' in data or ctx.triggered_id == 'reset' else 
        (data['xaxis.range[0]'], data['xaxis.range[1]'], data['yaxis.range[0]'], data['yaxis.range[1]'])
    )
    
    mandelbrot = mandelbrot_set(
        range_real=(x_min, x_max),
        range_im=(y_min, y_max),
        dims=(height, width),
        max_iter=iterations,
        save_all=True
    )

    fig = px.imshow(
        mandelbrot,
        animation_frame=0,
        x=np.linspace(x_min, x_max, width),
        y=np.linspace(y_min, y_max, height),
        aspect='equal',
        color_continuous_scale='Hot',
        labels={'x': 'Re(c)', 'y': 'Im(c)', 'color': 'Iterations'},
        title='Mandelbrot Set Visualization',
        range_color=[0, iterations],
        contrast_rescaling='minmax'
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    
    fig.layout.sliders[0]['currentvalue']['prefix'] = 'Iteration: '
    fig.layout.updatemenus[0]['buttons'][0]['label'] = 'Play'
    fig.layout.updatemenus[0]['buttons'][1]['label'] = 'Pause'
    return fig

@app.callback(
    Output('graph', 'children'),
    Input('heatmap', 'clickData'),
    Input('iterations', 'value')
)
def display_click_data(clickData, iterations):
    if clickData is None or iterations is None:
        return None
    c = complex(clickData['points'][0]['x'], clickData['points'][0]['y'])
    z, z_real, z_imag, z_magnitudes = 0, [0], [0], [0]
    for i in range(iterations):
        z = z ** 2 + c
        z_real.append(z.real)
        z_imag.append(z.imag)
        z_magnitude = abs(z)
        z_magnitudes.append(z_magnitude)
        if z_magnitude > 2.: break
    data = pd.DataFrame(
        {'Re(z)': z_real, 'Im(z)': z_imag, '|z|': z_magnitudes, 'Iteration': range(len(z_real))}
    )
    fig = _plot(c, data, iterations)
    return dcc.Graph(id=f'graph_{c}', figure=fig)

def _plot(c: complex, data: pd.DataFrame, iterations: int) -> go.Figure:
    fig = make_subplots(cols=3)
    
    # DRAW LINES + CIRCLE
    hovertemplates = (
        'Iteration: %{x}<br>Re(z): %{y}<extra></extra>',
        'Iteration: %{x}<br>Im(z): %{y}<extra></extra>',
        'Re(z): %{x}<br>Im(z): %{y}<extra></extra>'
    )
    line = dict(color='blue')
    fig.append_trace(
        go.Scatter(
            x=data['Iteration'], y=data['Re(z)'], 
            mode='lines', line=line, name='Re(z)', 
            hovertemplate=hovertemplates[0]
        ), 
        row=1, col=1
    )
    fig.append_trace(
        go.Scatter(
            x=data['Iteration'], y=data['Im(z)'], 
            mode='lines', line=line, name='Im(z)',
            hovertemplate=hovertemplates[1]
        ),
        row=1, col=2
    )
    fig.append_trace(
        go.Scatter(
            x=data['Re(z)'], y=data['Im(z)'], 
            mode='lines', line=line, name='z',
            hovertemplate=hovertemplates[2]
        ),
        row=1, col=3
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
    fig.append_trace(
        go.Scatter(x=[0], y=[0], mode='markers', marker=marker, hovertemplate=hovertemplates[0]), 
        row=1, col=1
    )
    fig.append_trace(
        go.Scatter(x=[0], y=[0], mode='markers', marker=marker, hovertemplate=hovertemplates[1]), 
        row=1, col=2
    )
    fig.append_trace(
        go.Scatter(x=[0], y=[0], mode='markers', marker=marker, hovertemplate=hovertemplates[2]), 
        row=1, col=3
    )
    
    fig.frames = [
        go.Frame(
            data=[
                go.Scatter(x=[data['Iteration'][i]], y=[data['Re(z)'][i]]),
                go.Scatter(x=[data['Iteration'][i]], y=[data['Im(z)'][i]]),
                go.Scatter(x=[data['Re(z)'][i]], y=[data['Im(z)'][i]])
            ],
            traces=[3, 4, 5],
            name=str(i)
        )
        for i in range(len(data))
    ]

    fig.update_layout(
        title=f'Evolution of c={c}',
        showlegend=False,
        updatemenus=[_updatemenu],
        sliders=[_slider(len(data))]
    )

    return fig
    
# BELOW THIS IT'S ALL DEFAULT

def _slider(n: int) -> dict:
    return {
        'active': 0,
        'currentvalue': {'prefix': 'Iteration: '},
        'len': 0.9,
        'pad': {'b': 10, 't': 60},
        'steps': [
            {
                'args': [
                    [str(i)],  
                    {
                        'frame': {'duration': 0, 'redraw': True}, 
                        'mode': 'immediate',
                        'fromcurrent': True,
                        'transition': {'duration': 0, 'easing': 'linear'}
                    }
                ],
                'label': str(i),
                'method': 'animate',
            }
            for i in range(n)
        ],
        'active': 0,
        'x': 0.1,
        'xanchor': 'left',
        'y': 0,
        'yanchor': 'top'
    }

_play_button = {
    'label': 'Play', 
    'method': 'animate',
    'args': [
        None, 
        {
            'frame': {'duration': 500, 'redraw': True},
            'mode': 'immediate', 
            'fromcurrent': True, 
            'transition': {'duration': 500, 'easing': 'linear'}
        }
    ]
}

_pause_button = {
    'label': 'Pause',
    'method': 'animate',
    'args': [
        [None], 
        {
            'frame': {'duration': 0, 'redraw': True},
            'mode': 'immediate', 
            'fromcurrent': True, 
            'transition': {'duration': 0, 'easing': 'linear'}
        }
    ]
}

_updatemenu = {
    'buttons': [_play_button, _pause_button],
    'direction': 'left',
    'pad': {'r': 10, 't': 70},
    'showactive': False,
    'type': 'buttons',
    'x': 0.1,
    'xanchor': 'right',
    'y': 0,
    'yanchor': 'top'            
}

if __name__ == '__main__':
    app.run_server(debug=True)
    # app.run_server()

