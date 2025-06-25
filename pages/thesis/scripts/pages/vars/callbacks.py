from dash import Output, Input, State, ctx, no_update, callback, html
from typing import List, Dict, Any

from pages.thesis.scripts.causal_node import CausalNode
import pages.thesis.scripts.pages.utils as utils
from pages.thesis.scripts.pages.vars.layout import var_table_styles

PREFIX = 'vars-'

@callback(
    Output(PREFIX+'radio-variable', 'options'),
    Output(PREFIX+'txt-warning', 'children'),
    Output(PREFIX+'intro', 'children'),
    Input('url', 'pathname'),
)
def load_page(pathname: str):
    if utils.helper is None: return [], 'Go back to Thesis and choose a forest first!', None
    intro = 'Here you can interact with individual variables. Click on one of the available ones to visualize how many trees selected it and in the context of which other variables'
    if pathname == '/thesis/variables': return utils.helper.variables, no_update, intro
    return [], None, None

@callback(
    Output(PREFIX+'variable-visualization-div', 'hidden'),
    Output(PREFIX+'graph', 'figure'),
    Input(PREFIX+'radio-variable', 'value')
)
def choose_variable(var: str):
    if var is None: return True, {}
    summary_full, selected_in_n_trees = utils.helper.get_summary_for_var_full(var, utils.helper.nodes_with_variable[var])
    title = f'You selected {var} | It has been selected in {selected_in_n_trees}/{len(utils.helper.forest)} trees'
    fig = utils.helper.get_summary_graph(summary_full)
    fig.update_layout(title=title)
    return False, fig

# @callback(
#     Output(PREFIX+'table', 'data'),
#     Output(PREFIX+'table', 'columns'),
#     Output(PREFIX+'table', 'selected_rows'),
#     Output(PREFIX+'table', 'selected_columns'),
#     Output(PREFIX+'table-header-div', 'children'),
#     Input(PREFIX+'radio-variable', 'value'),
#     Input(PREFIX+'button-show-table', 'n_clicks')
# )
# def see_table_for_var(var: str, clicked: int):
#     if var is None or ctx.triggered_id == PREFIX+'radio-variable': return None, None, [], [], None
#     nodes_with_var = utils.helper.nodes_with_variable[var]
    
#     # The chosen variable is never significant
#     if nodes_with_var == []: return None, None, [], [], None
#     data = utils.helper.get_table(var=var, nodes_with_var=nodes_with_var)
#     columns = [{'name': col, 'id': col, 'selectable': True} for col in data[0].keys()]
#     return data, columns, [], [], [
#         html.P(f'The following table shows all nodes in which {var} is available.'),
#         html.Ul([
#             html.Li('Rows colored in blue represent nodes in which the variable is chosen as a split.'),
#             html.Li('Splits with a * indicate that they are statistically significant.'),
#             html.Li('Individual columns can be selected to highlight them'),
#             html.Li('Individual rows can be selected to visualize the split at the node')      
#         ])
#     ]

# @callback(
#     Output(PREFIX+'table', 'style_data_conditional'),
#     Input(PREFIX+'table', 'selected_rows'),
#     Input(PREFIX+'table', 'selected_columns'),
#     State(PREFIX+'table', 'style_data_conditional')
# )
# def highlight_row_or_column(selected_rows: List[int], selected_columns: List[int], current_styles: List[Dict[str, Any]]):
#     styles = var_table_styles.copy() if current_styles else []
#     border = '3px solid black'
#     if selected_columns: styles.append({'if': {'column_id': selected_columns[0]}, 'border-left': border, 'border-right': border})
#     if selected_rows: styles.append({'if': {'row_index': selected_rows[0]}, 'border-bottom': border, 'border-top': border})
#     return styles

# @callback(
#     Output(PREFIX+'graph-at-node', 'figure'),
#     Output(PREFIX+'graph-at-stratum', 'figure'),
#     Output(PREFIX+'graphs-div', 'hidden'),
#     Input(PREFIX+'radio-variable', 'value'),
#     Input(PREFIX+'table', 'selected_rows'),
#     State(PREFIX+'table', 'data'),
# )
# def select_in_table(var: str, selected_rows: List[int], table_data: List[Dict[str, Any]]):
#     if table_data is None or not selected_rows: return {}, {}, True
#     r = table_data[selected_rows[0]]
#     node: CausalNode = utils.helper.nodes_with_variable[var][r['tree id']][r['node id']] 
#     df_at_node = utils.helper.forest.df.query(node.context) if node.context else utils.helper.forest.df.copy()
#     fig_at_node = utils.helper.show_graph(df_at_node, var, r['threshold'])
#     fig_at_strata = utils.helper.show_graph_strata(df_at_node, node.strata[var], var, r['threshold'])
#     return fig_at_node, fig_at_strata, False
