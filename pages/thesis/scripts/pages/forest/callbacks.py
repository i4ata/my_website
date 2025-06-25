from dash import Output, Input, State, ctx, no_update, html, callback, clientside_callback
from dash.exceptions import PreventUpdate
import traceback
import numpy as np
import pandas as pd
from typing import Literal, Optional, Dict, List, Any

from pages.thesis.scripts.causal_node import CausalNode
from pages.thesis.scripts.causal_node import ReasonForTerminal
import pages.thesis.scripts.pages.utils as utils
from pages.thesis.scripts.pages.forest.layout import table_styles

PREFIX='forest-'

@callback(
    Output(PREFIX+'radio-tree', 'options'),
    Output(PREFIX+'txt-warning', 'children'),
    Output(PREFIX+'txt-tree-selection-header', 'children'),
    Input('url', 'pathname')
)
def load_page(pathname: str):
    if utils.helper is None: return [], 'Go back to Thesis and choose a forest first!', None
    if pathname != '/thesis/forest': return [], no_update, None
    
    options = [
        {
            'label': html.Span(
                str(i), 
                style=(
                    ({'color': 'green'} if i == len(utils.helper.forest) - 1 else {}) 
                    | 
                    ({'font-weight': 'bold'} if utils.helper.forest.popularity_counts[i] == utils.helper.forest.highest_frequency else {})
                )
            ),
            'value': i
        }
        for i in range(len(utils.helper.forest))
    ]
    return options, no_update, 'Select a tree from the forest to interact!'
    
@callback(
    Output(PREFIX+'cyto-tree', 'elements'),
    Output(PREFIX+'cyto-tree', 'tapNodeData'),
    Output(PREFIX+'cyto-tree', 'selectedNodeData'),
    Output(PREFIX+'txt-tree', 'children'),
    Output(PREFIX+'save-tree-div', 'hidden'),
    Input(PREFIX+'radio-tree', 'value'),
)
def select_tree(tree_idx: Optional[int]):
    if tree_idx is None: return [], None, None, None, True
    tree_info = (
        [f'You selected tree {tree_idx}, It appears {utils.helper.forest.popularity_counts[tree_idx]} times in the forest'] + 
        ([html.Br(), 'This tree is trained on the original (non-bootstrapped) data'] if tree_idx == len(utils.helper.forest)-1 else [])
    )
    return utils.helper.graph_data[tree_idx], None, None, tree_info, True # Set to false to enable making pdf graphs from the tree 

@callback(
    Output(PREFIX+'save-tree-txt-output', 'children'),
    Input(PREFIX+'save-tree-button-save', 'n_clicks'),
    State(PREFIX+'radio-tree', 'value'),
    State(PREFIX+'save-tree-name', 'value'),
    State(PREFIX+'save-tree-radio-leaves', 'value')
)
def save_tree(n_clicks: Optional[int], tree_idx: Optional[int], name: str, plot_leaves: bool):
    if n_clicks is None or tree_idx is None or utils.helper is None: return no_update
    utils.helper.forest.trees[tree_idx].show_tree(df=utils.helper.forest.df, name=name, plot_leaf=plot_leaves)
    return f'Tree #{tree_idx} successfully saved at resources/graphviz/{name}.pdf'

@callback(
    Output(PREFIX+'txt-click-tree-node', 'children'),
    Output(PREFIX+'graph-at-node', 'figure'),
    Output(PREFIX+'graph-at-node', 'style'),
    Output(PREFIX+'radio-variable', 'options'),
    Output(PREFIX+'radio-variable', 'value'),
    Output(PREFIX+'variable-selection-header', 'children'),
    Input(PREFIX+'radio-tree', 'value'),
    Input(PREFIX+'cyto-tree', 'tapNodeData')
)
def display_click_data(tree_idx: Optional[int], node_data: Optional[Dict[Literal['id', 'label'], str]]):
    """
    Upon tapping a node in the tree at hand, print a summary for that node,
    draw the associated graph, and enable variable selection
    """

    # Upon first call, do nothing
    if node_data is None:
        return 'Click on a node to see details' if tree_idx is not None else None, {}, {'display': 'none'}, [], None, None
    
    # Get the node object with the corresponding id in the selected tree
    node: CausalNode = utils.helper.nodes[tree_idx][node_data['id']]

    # Generate a message to be printed about this node
    click_message = [
        f'You clicked on node {node_data["label"]} (ID: {node_data["id"]})', html.Br(), 
        f'Context: {node.context.replace("`", "")}', html.Br(), 
        f'Samples: {node.samples}'
    ]
    if node.terminal: click_message.extend([
            html.Br(),
            f'Reason for being terminal: {node.reason_for_terminal.value}',
            html.Br(),
            f'Risk score: {node.prediction:.4f}'   
        ])

    # Get the df at the node at hand by querying the original data
    df_at_node = utils.helper.forest.df.query(node.context) if node.context else utils.helper.forest.df.copy()
    
    # Generate the graph to plot
    fig = (
        utils.helper.show_graph(df=df_at_node, col=node.feature, threshold=node.threshold) 
        if not node.terminal else
        utils.helper.show_graph(df=df_at_node)
    )

    if (
        node.terminal and 
        node.reason_for_terminal in (
            ReasonForTerminal.min_samples_per_split,
            ReasonForTerminal.no_more_possible_splits
        )
    ):
        return click_message, fig, {'display': 'block'}, [], None, None

    node_vars = sorted(df_at_node.columns[(df_at_node.columns != utils.helper.y_col) & (df_at_node.nunique() > 1)])
    return click_message, fig, {'display': 'block'}, ['ALL'] + node_vars, 'ALL', 'Available variables'

@callback(
    Output(PREFIX+'table', 'data'),
    Output(PREFIX+'table', 'columns'),
    Output(PREFIX+'table', 'selected_rows'),
    Output(PREFIX+'table', 'selected_columns'),
    Output(PREFIX+'table-stratified', 'data'),
    Output(PREFIX+'table-stratified', 'columns'),
    Output(PREFIX+'table-stratified', 'selected_rows'),
    Output(PREFIX+'table-stratified', 'selected_columns'),
    Output(PREFIX+'div-tables', 'hidden'),
    Input(PREFIX+'radio-tree', 'value'),
    Input(PREFIX+'cyto-tree', 'tapNodeData'),
    Input(PREFIX+'radio-variable', 'value')
)
def get_tables(tree_idx: int, node_data: Optional[Dict[Literal['id', 'label'], str]], variable: Optional[str]):
    if node_data is None or variable is None: return None, None, [], [], None, None, [], [], True
    node: CausalNode = utils.helper.nodes[tree_idx][node_data['id']]
 
    tree_df = (
        utils.helper.forest.df.sample(frac=1., replace=True, ignore_index=True, random_state=utils.helper.forest.bootstrap_random_states[tree_idx])
        if tree_idx < len(utils.helper.forest.bootstrap_random_states) else
        utils.helper.forest.df.copy()
    )
    all_splits, strata, stratified_scores = node.fit(df=tree_df.query(node.context) if node.context else tree_df.copy(), y_col=node.y_col, **utils.helper.forest.hparams)

    ###################################################
    # INDEPENDENT DF
    # If the selected variable is ALL, show the best split for each variable
    if variable == 'ALL':
        df_ind: pd.DataFrame = (
            all_splits
            .loc[list(zip(node.best_splits_per_var_independent_all.index, node.best_splits_per_var_independent_all['threshold'].values))]
            .reset_index()
        )
    # Otherwise, show all splits for the selected variable
    else:
        df_ind = all_splits.loc[variable].reset_index(names='threshold')
    df_ind = df_ind.round(3)

    data_ind = df_ind.to_dict('records')
    columns_ind = [{'name': i, 'id': i, 'selectable': True} for i in df_ind.columns]

    if ctx.triggered_id == 'radio-variable':
        return data_ind, columns_ind, [], [], no_update, no_update, no_update, no_update, False
    ###################################################

    ###################################################
    # STRATIFIED DF
    if node.reason_for_terminal in (
        ReasonForTerminal.no_significant_splits,
        ReasonForTerminal.min_samples_per_split,
        ReasonForTerminal.no_more_possible_splits
    ):
        return data_ind, columns_ind, [], [], None, None, [], [], False

    df_strat = stratified_scores.reset_index(names='variable').round(3)

    # Move the threshold to the second position
    threshold = df_strat.pop('threshold')
    df_strat.insert(1, 'threshold', threshold)

    data_strat = df_strat.to_dict('records')
    columns_strat = [{'name': i, 'id': i, 'selectable': True} for i in df_strat.columns]
    ###################################################
    
    return data_ind, columns_ind, [], [], data_strat, columns_strat, [], [], False

@callback(
    Output(PREFIX+'table', 'style_data_conditional'),
    Output(PREFIX+'graph-at-split', 'figure'),
    Output(PREFIX+'graph-at-split', 'style'),
    Input(PREFIX+'radio-tree', 'value'),
    Input(PREFIX+'cyto-tree', 'tapNodeData'),
    Input(PREFIX+'radio-variable', 'value'),
    Input(PREFIX+'table', 'selected_rows'),
    Input(PREFIX+'table', 'selected_columns'),
    State(PREFIX+'table', 'data'),
    State(PREFIX+'table', 'style_data_conditional')
)
def select_in_table(
    tree_idx: int, 
    node_data: Optional[Dict[Literal['id', 'label'], str]], 
    variable: Optional[str], 
    selected_rows: List[int], 
    selected_columns: List[int],
    table_data: List[Dict[str, Any]], 
    current_styles: List[Dict[str, Any]]
):
    """Upon selecting a raw or a column in the table, highlight them, and generate a graph for the threshold at the selected row"""

    # If the table has not been generated yet, do nothing
    if not table_data: return current_styles, {}, {'display': 'none'}
    
    # Reset the styles to the default
    styles = table_styles.copy() if current_styles else []
    
    # Highlight the row with the largest significant absolute z test statistic
    z = np.array([row['z'] for row in table_data], dtype=float)
    if not np.isnan(z).all():
        index = 'threshold' if variable != 'ALL' else 'variable'
        best_index = table_data[np.nanargmax(np.abs(z))][index]
        styles.append({
            'if': {
                'filter_query': f'{{{index}}} = `{best_index}` && {{p_corrected}} <= 0.05'
            },
            'backgroundColor': 'sandybrown'
        })

    border = '3px solid black'

    # Highlight the selected column
    if selected_columns:
        c = selected_columns[0]
        styles.append({'if': {'column_id': c}, 'border-left': border, 'border-right': border})
    
    # If a row has been selected, highlight it and draw the graph
    if selected_rows:
        r = selected_rows[0]
        styles.append({'if': {'row_index': r}, 'border-bottom': border, 'border-top': border})

        # Case where the column selection triggered the callback 
        # but there is already a selected row. In that case, don't redraw the graph
        if (
            len(ctx.triggered_prop_ids) == 1 and 
            PREFIX+'table.selected_columns' in ctx.triggered_prop_ids
        ):
            return styles, no_update, no_update
        
        # Redraw the graph
        node: CausalNode = utils.helper.nodes[tree_idx][node_data['id']]
        if table_data[r]['p_corrected'] is None:
            return styles, {}, {'display': 'none'}
    
        df_at_node = utils.helper.forest.df.query(node.context) if node.context else utils.helper.forest.df.copy()
        curves_at_node = utils.helper.show_graph(
            df_at_node, 
            variable if variable != 'ALL' else table_data[r]['variable'], 
            table_data[r]['threshold']
        )
        return styles, curves_at_node, {'display': 'block'}
    return styles, {}, {'display': 'none'}

@callback(
    Output(PREFIX+'table-stratified', 'style_data_conditional'),
    Output(PREFIX+'graph-at-stratum', 'figure'),
    Output(PREFIX+'graph-at-stratum', 'style'),
    Output(PREFIX+'txt-strata', 'children'),
    Input(PREFIX+'radio-tree', 'value'),
    Input(PREFIX+'cyto-tree', 'tapNodeData'),
    Input(PREFIX+'table-stratified', 'selected_rows'),
    Input(PREFIX+'table-stratified', 'selected_columns'),
    State(PREFIX+'table-stratified', 'data'),
    State(PREFIX+'table-stratified', 'style_data_conditional'),
)
def select_in_stratified_table(
    tree_idx: int, 
    node_data: Optional[Dict[Literal['id', 'label'], str]], 
    selected_rows: List[int], 
    selected_columns: List[int], 
    table_data: List[Dict[str, Any]], 
    current_styles: List[Dict[str, Any]]
):
    """Upon selecting a raw or a column in the table, highlight them, and enable the user to select a specific stratum"""
    try:
        # If we haven't generated the table everything is default
        if not table_data: return current_styles, {}, {'display': 'none'}, None
        node: CausalNode = utils.helper.nodes[tree_idx][node_data['id']]

        styles = table_styles.copy() if current_styles else []
        
        z = np.array([row['z'] for row in table_data], dtype=float)
        if not np.isnan(z).all():
            best_index = table_data[np.nanargmax(np.abs(z))]['variable']
            styles.extend([
                {
                    'if': {
                        'filter_query': f'{{variable}} = `{best_index}` && {{p_corrected}} <= 0.05'
                    },
                    'backgroundColor': 'sandybrown'
                },
                {
                    'if': {
                        'filter_query': f'{{variable}} = `{node.feature}`'
                    },
                    'backgroundColor': 'turquoise'
                }
            ])
        
        border = '3px solid black'

        strata_info = None
        if selected_columns:
            c = selected_columns[0]
            styles.append({'if': {'column_id': c}, 'border-left': border, 'border-right': border})
        
        if selected_rows:
            r = selected_rows[0]
            styles.append({'if': {'row_index': r}, 'border-bottom': border, 'border-top': border})
            var = table_data[r]['variable']
            strata_info = utils.helper.get_stratification_summary(node, var)
        else:
            return styles, {}, {'display': 'none'}, None
        
        # Case where the column selection triggered the callback but there is already a selected row. In that case, don't redraw the graph
        if len(ctx.triggered_prop_ids) == 1 and PREFIX+'table-stratified.selected_columns' in ctx.triggered_prop_ids:
            return styles, no_update, no_update, no_update
        
        df_at_node = utils.helper.forest.df.query(node.context) if node.context else utils.helper.forest.df.copy()
        var, threshold = table_data[r]['variable'], table_data[r]['threshold']
        fig = utils.helper.show_graph_strata(
            df_at_node, node.strata[var], var, threshold
        )
        return styles, fig, {'display': 'block'}, strata_info
    except Exception as e:
        print(traceback.format_exc())
        raise PreventUpdate

# Copied from https://stackoverflow.com/questions/78017670/is-it-possible-to-use-the-depthsort-argument-in-breadthfirst-layout-in-dash-cyto
# Used to make sure that the left child is to the left of the parent and the right child is to the right
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
