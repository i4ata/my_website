# THIS VALUES ARE ALL DEFAULT

def slider(n: int) -> dict:
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

updatemenu = {
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
