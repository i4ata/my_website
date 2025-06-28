from dash import callback, Input, Output, html, State, no_update
from pathlib import Path
import os
from typing import List, Any, Optional

from pages.thesis.scripts.pages.home.layout import PREFIX
import pages.thesis.scripts.pages.utils as utils
from pages.thesis.scripts.forest import Forest, RegressionForest, SurvivalForest

def format_filename(f: str):
    return (
        f
        .replace('pages/thesis/resources/models', '')
        .replace('/survival', 'Survival | ')
        .replace('/regression', 'Regression | ')
        .replace('/binary', ' (Binary)')
        .replace('/continuous', ' (Continuous)')
        .replace('/causal_forest.pkl', '')
        .replace('_', ' ')
        .replace('/', ' ')
        .title()
    )

# Called when the page is loaded
@callback(
    Output(PREFIX+'models', 'options'),
    Input('url', 'pathname'),
    State(PREFIX+'models', 'options')
)
def load_page(pathname: Optional[str], current_options):
    if pathname is None: return []
    if current_options is not None: return no_update
    if pathname == '/thesis':
        return [
            {'label': format_filename(file), 'value': file}
            for file in map(str, Path('pages', 'thesis', 'resources', 'models').rglob('*causal*.pkl'))
        ]
    return []

def get_summary(forest: Forest, forest_path: str) -> List[Any]:
    return [
        html.P('✅ Model loaded successfully!'),
        html.Ul([
            html.Li('To interact with the individual trees, go to Forest'),
            html.Li('To interact with the individual variables, go to Variables'),
            html.Li('To interact with the most common tree, go to Most Common Tree'),
        ]),
        html.P('Summary:'),
        html.Ul([
            html.Li(f'Task: {"Survival" if isinstance(forest, SurvivalForest) else "Regression"}'),
            html.Li(f'Model size on disk: {os.path.getsize(forest_path) / 1024 ** 2 :.3f} MB'),
            html.Li(f'Total number of samples: {len(forest.df)}'),
            html.Li(f'Total number of features: {len(forest.df.columns) - 1}'),
            html.Li(f'Number of trees: {len(forest) - 1}'),
            html.Li(['Training performance', html.Ul([html.Li(f'{metric}: {value}') for metric, value in forest.metrics.items()])]),
            html.Li(['Hyperparameters', html.Ul([html.Li(f'{param}: {value}') for param, value in forest.hparams.items()])]),
            html.Li([f'Frequency of the most common tree: {forest.highest_frequency}']),
            html.Li([f'Is the tree trained on the original (non-bootstrapped) also the most common tree: {"Yes" if forest.popularity_counts[-1] == forest.highest_frequency else "No"}'])
        ])
    ]

@callback(
    Output(PREFIX+'info', 'children'),
    Input(PREFIX+'models', 'value')
)
def select_model(forest_path: Optional[str]):
    if forest_path is None: return None
    assert os.path.exists(forest_path)

    # Load the model again only if needed
    if utils.helper is None or forest_path != utils.helper.forest_path:
        utils.helper = utils.create_helper(forest_path=forest_path)
    
    return get_summary(forest=utils.helper.forest, forest_path=forest_path)
