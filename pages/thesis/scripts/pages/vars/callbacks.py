from dash import Output, Input, no_update, callback
import pages.thesis.scripts.pages.utils as utils
PREFIX = 'vars-'

@callback(
    Output(PREFIX+'radio-variable', 'options'),
    Input('url', 'pathname'),
)
def load_page(pathname: str):
    if pathname != '/thesis/variables': return no_update 
    return utils.helper.variables
    
@callback(
    Output(PREFIX+'graph', 'figure'),
    Output(PREFIX+'graph', 'style'),
    Input(PREFIX+'radio-variable', 'value')
)
def choose_variable(var: str):
    if var is None: return {}, {'display': 'none'}
    summary_full, selected_in_n_trees = utils.helper.get_summary_for_var_full(var, utils.helper.nodes_with_variable[var])
    title = f'You selected {var} | It has been selected in {selected_in_n_trees}/{len(utils.helper.forest)} trees'
    fig = utils.helper.get_summary_graph(summary_full)
    fig.update_layout(title=title)
    return fig, {'display': 'block'}
