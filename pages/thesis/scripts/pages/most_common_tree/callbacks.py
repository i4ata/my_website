from dash import Output, Input, State, clientside_callback, callback, no_update
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Optional

from pages.thesis.scripts.pages.most_common_tree.layout import PREFIX
import pages.thesis.scripts.pages.utils as utils
from pages.thesis.scripts.most_common_tree import MostCommonNode

# Called when the page is loaded
@callback(
    Output(PREFIX+'txt-info', 'children'),
    Output(PREFIX+'cyto-tree', 'elements'),
    Output(PREFIX+'txt-warning', 'children'),
    Input('url', 'pathname')
)
def load_page(pathname: Optional[str]):
    if utils.helper is None: return None, [], 'Go back to Thesis and choose a forest first!'
    if pathname == '/thesis/most_common_tree': return (
        f'The most common tree occurs {utils.helper.forest.highest_frequency}/{len(utils.helper.forest)} times. Click on a node to visualize the distribution of threshold',
        utils.helper.graph_data_most_common,
        no_update
    )
    return None, [], None

# Called when a node is selected
@callback(
    Output(PREFIX+'graph-threshold-distribution', 'figure'),
    Output(PREFIX+'graph-threshold-distribution', 'style'),
    Input(PREFIX+'cyto-tree', 'tapNodeData')
)
def plot_threshold_dist(node_data):
    if node_data is None: return {}, {'display': 'none'}
    node: MostCommonNode = utils.helper.nodes_most_common[node_data['id']]
    if node.terminal: return {}, {'display': 'none'}
    title = '<br>'.join([
        f'Distribution for the threshold of {node.feature} in node {node.node_id}',
        ' | '.join([f'{k}: {v:.4f}' for k, v in pd.Series(node.thresholds).describe().to_dict().items()]),
        f'Context: {node.context.replace("`", "").replace(">", " > ").replace("<=", " <= ")}'
    ])
    fig = px.histogram(x=node.thresholds, title=title)
    fig.add_vline(np.median(node.thresholds), line_dash='dash', line_width=3, line_color='orange')
    return fig, {'display': 'block'}

# Use to display the order of the leaves in the tree correctly
clientside_callback(
    """
    function (id, layout) {
        layout.depthSort = (a, b) => a.data('id') - b.data('id');
        cy.layout(layout).run();
        return layout;
    }
    """,
    Output(PREFIX+'cyto-tree', 'layout'),  # update the (dash) cytoscape component's layout
    Input(PREFIX+'cyto-tree', 'id'),       # trigger the function when the Cytoscape component loads (1)
    State(PREFIX+'cyto-tree', 'layout'),   # grab the layout so we can update it in the function
    prevent_initial_call=False # ensure (1) (needed if True at the app level)
)
