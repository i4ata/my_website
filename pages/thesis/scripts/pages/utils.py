import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
from typing import Dict, Optional, Tuple, List, Any
from tqdm.auto import tqdm
from dash import html
import re
import math
from abc import ABC, abstractmethod

from pages.thesis.scripts.base import Node
from pages.thesis.scripts.causal_node import CausalNode
from pages.thesis.scripts.scikit_node import ScikitNode
# from pages.thesis.scripts.forest import Forest, SurvivalForest, RegressionForest

# This is just a way to load the pickle files that were previously dumped in the original thesis repo
import sys
sys.path.append('pages/thesis')
from scripts.forest import Forest, SurvivalForest, RegressionForest


import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class VisualizationHelper(ABC):

    def __init__(self, forest: Forest, forest_path: str) -> None:
        
        self.forest_path = forest_path
        self.forest = forest
        self.graph_data, self.nodes = zip(*[
            self.get_nodes_and_edges(tree.root) 
            for tree in self.forest.trees
        ])
        self.y_col = self.forest.y_col
        
        cols = self.forest.df.columns
        self.variables = cols[cols != self.y_col].to_list()
        self.nodes_with_variable = {
            var: self.get_nodes_with_variable(var, self.forest) 
            for var in self.variables
        }
        self.graph_data_most_common, self.nodes_most_common = self.get_nodes_and_edges(self.forest.most_common_tree.root)

    def get_stratification_summary(self, node: CausalNode, variable: str) -> tuple:
        stratifying_variables = (
            [var.split('`')[1] for var in node.strata[variable][0].split(' & ')]
            if node.strata[variable][0] else []
        )
        prefix = variable.split('_', maxsplit=1)[0]
        strata_info = (
            'Stratified by:', 
            html.Ul([html.Li(v, style={'color': 'darkgreen'}) for v in stratifying_variables]),
            'Omitting:',
            html.Ul([
                html.Li(var, style={'color': 'sandybrown' if var in node.strata else 'tomato'}) 
                for var in node.all_vars if (
                    var != variable and
                    var not in stratifying_variables and
                    var.split('_', maxsplit=1)[0] != prefix
                )
            ])
        )
        return strata_info

    def get_summary_graph(self, summary: Dict[str, Dict[bool, int]]) -> go.Figure:

        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.T.reset_index(names='Context')
        summary_df['Context'] = summary_df['Context'].replace('', 'No Context')
        summary_df_long = summary_df.melt(id_vars='Context', var_name='Category', value_name='Count')
        summary_df_long = (
            summary_df_long
            .assign(l=summary_df_long['Context'].apply(lambda x: -1 if x == 'No Context' else x.count(',')))
            .sort_values(by=['l', 'Context'], ignore_index=True)
            .drop('l', axis='columns')
        )
        summary_df_long['Category'] = summary_df_long['Category'].map({False: 'Not selected', True: 'Selected'})
        fig = px.bar(
            summary_df_long, y='Context', x='Count', color='Category', orientation='h', 
            category_orders={'Context': summary_df_long['Context'].unique().tolist()}
        )
        return fig

    @abstractmethod
    def _show_graph_groups(self, df: pd.DataFrame, col: str, threshold: float) -> go.Figure:
        pass

    @abstractmethod
    def _show_graph_single(self, df: pd.DataFrame) -> go.Figure:
        pass
        
    def show_graph(self, df: pd.DataFrame, col: Optional[str] = None, threshold: Optional[float] = None) -> go.Figure:
        return (
            self._show_graph_groups(df=df, col=col, threshold=threshold)
            if col is not None and threshold is not None else
            self._show_graph_single(df=df)
        )

    @abstractmethod
    def show_graph_strata(self, df: pd.DataFrame, strata: List[str], col: str, threshold: float) -> go.Figure:
        pass

    def get_nodes_and_edges(self, root: Node) -> Tuple[List, Node]:
        data, nodes = [], {}
        def traverse(node: Node) -> None:
            node_id = str(node.node_id)

            if node.terminal: label = 'Leaf'
            else: label = f'{node.feature} > {node.threshold}'
            
            data.append({'data': {'id': node_id, 'label': label}})
            nodes[node_id] = node
            if not node.terminal:
                data.append({'data': {'source': node_id, 'target': str(node.left.node_id), 'label': 'T'}})
                traverse(node.left)
                data.append({'data': {'source': node_id, 'target': str(node.right.node_id), 'label': 'F'}})
                traverse(node.right)
        traverse(root)
        return data, nodes

    def get_summary_for_var_full(self, var: str, nodes: List[Dict[int, CausalNode]]) -> Tuple[Dict[str, Dict[bool, int]], int]:

        summary: Dict[str, Dict[bool, int]] = defaultdict(lambda: {False: 0, True: 0})
        selected_in_n_trees = 0
        for tree in nodes:
            selected_here = False
            for node in tree.values():
                if node.feature == var: selected_here = True
                summary[', '.join(re.findall(r'`(.*?)`', node.context))][selected_here] += 1
            selected_in_n_trees += selected_here
        return dict(summary), selected_in_n_trees

    def get_nodes_with_variable(self, var: str, forest: Forest) -> List[Dict[int, Node]]:

        def traverse(root: ScikitNode) -> List[ScikitNode]:
            # Either a terminal node OR the variable does not vary anymore in the remainder of the tree
            if root.terminal or var not in root.all_vars: return []
            return [root] + traverse(root.left) + traverse(root.right)

        return [{node.node_id: node for node in traverse(tree.root)} for tree in forest.trees]

    def get_table(self, var: str, nodes_with_var: List[Dict[int, CausalNode]]) -> List[Dict[str, Any]]:
        prefix = var.split('_', maxsplit=1)[0]
        data = []
        for i, tree in enumerate(nodes_with_var):
            
            for node in tree.values():
                if var not in node.strata:
                    stratified_by = []
                    not_stratified_by = []
                else:
                    best = node.best_splits_per_var_independent_all # A bit of a shorthand
                    stratification_variables = re.findall(r'`(.*?)`', node.strata[var][0])
                    stratified_by = [
                        v + f" > {best.loc[v, 'threshold']}" + ('*' if best.loc[v, 'significant'] else '') 
                        for v in stratification_variables
                    ]
                    
                    not_stratified_by = [
                        v + f" > {best.loc[v, 'threshold']}" + ('*' if best.loc[v, 'significant'] else '')
                        for v in node.all_vars
                        if v != var and v.split('_', maxsplit=1)[0] != prefix and v not in stratification_variables
                    ]

                data.append({
                    'tree id': i,
                    'node id': node.node_id,
                    'threshold': node.best_splits_per_var_independent_all.loc[var, 'threshold'],
                    'context': node.context.replace(' & ', '\n').replace('`', '').replace('>', ' > ').replace('<', ' < '),
                    'depth': 0 if node.context == '' else node.context.count('&') + 1, # Infer the depth of the node using the context,
                    'is split': 'T' if node.feature == var else 'F',
                    'stratified by': '\n'.join(stratified_by),
                    'not stratified by': '\n'.join(not_stratified_by)
                })
        return data
    
class VisualizationHelperRegression(VisualizationHelper):
    
    def _show_graph_groups(self, df: pd.DataFrame, col: str, threshold: float) -> go.Figure:
        return px.box(
            data_frame=df.assign(split=(df[col]>threshold).map({True: f'{col} > {threshold}', False: f'{col} <= {threshold}'})), 
            x='split', y=self.y_col, 
            category_orders={'split': [f'{col} > {threshold}', f'{col} <= {threshold}']}
        )
    
    def _show_graph_single(self, df: pd.DataFrame) -> go.Figure:
        return px.histogram(x=df[self.y_col])
    
    def show_graph_strata(self, df: pd.DataFrame, strata: List[str], col: str, threshold: float) -> go.Figure:
        if not strata or strata == ['']: return self.show_graph(df, col, threshold)
        stratum_df = pd.concat([df.query(s).assign(stratum=s) for s in strata])
        stratum_df['split'] = (stratum_df[col] > threshold).map({True: f'> {threshold}', False: f'<= {threshold}'})
        category_orders = {'split': [f'> {threshold}', f'<= {threshold}'], 'stratum': sorted(strata)}
        stratum_df['stratum'] = stratum_df['stratum'].str.replace(' & ', '<br>').str.replace('`', '')
        fig = px.box(stratum_df, x='stratum', y=self.y_col, color='split', category_orders=category_orders)
        fig.update_layout(title=f'Effect of {col} in the different strata', legend_title_text=col)
        return fig

class VisualizationHelperSurvival(VisualizationHelper):

    def __init__(self, forest: Forest, forest_path: str):
        super().__init__(forest, forest_path)
        self._unique_times_all = np.sort(self.forest.df[self.y_col].abs().unique())

    def _get_curve(self, df: pd.DataFrame) -> pd.DataFrame:
        estimator = KaplanMeierFitter()
        # TODO: DEAL WITH THIS IN A BETTER WAY MYB
        suffix = '123_random_suffix_to_ensure_it_doesnt_match_123_fix_later'
        E, T = 't' + suffix, 'e' + suffix
        assert not {E, T}.issubset(df.columns)
        temp_df = (
            df
            .assign(**{E: df[self.y_col] >= 0, T: df[self.y_col].abs()})
            .drop(self.y_col, axis='columns')
        )
        estimator.fit(temp_df[T], temp_df[E])
        summary = temp_df.groupby(T).agg(
            Deaths = (E, 'sum'),
            Censored = (E, lambda x: (x==0).sum())
        )
        summary['At risk'] = len(df) - (
            (summary['Deaths'] + summary['Censored'])
            .cumsum()
            .shift(1, fill_value=0)
        )
        summary[['Deaths', 'Censored']] = summary[['Deaths', 'Censored']].cumsum()
        result = (
            pd.concat(
                (estimator.survival_function_, estimator.confidence_interval_, summary),
                axis=1
            )
            .rename(columns={
                f'KM_estimate_lower_0.95': 'Lower 95%',
                f'KM_estimate_upper_0.95': 'Upper 95%',
                f'KM_estimate': 'Estimate'
            })
            .round(4)
        )
        return result

    def _get_confint(self, df: pd.DataFrame, color: str, legendgroup: str) -> go.Scatter:
        assert {'Upper 95%', 'Lower 95%'}.issubset(df.columns)
        return go.Scatter(
            x=np.concatenate((df.index, df.index[::-1])),
            y=pd.concat((df['Upper 95%'], df['Lower 95%'][::-1])),
            fill='toself',
            fillcolor=color,
            opacity=.2,
            line={'width': 0},
            showlegend=False,
            hoverinfo='skip',
            legendgroup=legendgroup
        )

    def _show_graph_groups(self, df: pd.DataFrame, col: str, threshold: float) -> go.Figure:
        mask = df[col] > threshold
        
        left_df = df[mask]
        left_summary = self._get_curve(df=left_df)
        # THIS REINDEXING IS TO ENSURE THAT WE CAN HOVER ON EVERY POIN IN TIME (SO PROBABLY NOT THAT NECESSARY)
        # left_summary = left_summary.reindex(self._unique_times_all[self._unique_times_all <= left_summary.index[-1]], method='ffill')
        left_label = f'> {threshold} (T: {(left_df[self.y_col] >= 0).sum()}, C: {(left_df[self.y_col] < 0).sum()})'
        left_summary[col] = left_label

        right_df = df[~mask]
        right_summary = self._get_curve(df=right_df)
        # right_summary = right_summary.reindex(self._unique_times_all[self._unique_times_all <= right_summary.index[-1]], method='ffill')
        right_label = f'<= {threshold} (T: {(right_df[self.y_col] >= 0).sum()}, C: {(right_df[self.y_col] < 0).sum()})'
        right_summary[col] = right_label

        color_map = {left_label: 'red', right_label: 'blue'}
        curves = pd.concat((left_summary, right_summary)).reset_index(names='time')
        fig = px.line(
            data_frame=curves,
            x='time', 
            y='Estimate',
            title=f'Kaplan-Meier curves of both groups in this node given the context', 
            color=col, 
            color_discrete_map=color_map, 
            hover_data=['At risk', 'Deaths', 'Censored', 'Upper 95%', 'Lower 95%'],
        )
        fig.update_yaxes(range=[-0.05, 1.])
        fig.update_xaxes(range=[0, None])

        fig.add_trace(self._get_confint(
            df=left_summary, 
            color=color_map[left_label], 
            legendgroup=left_label
        ))
        fig.add_trace(self._get_confint(
            df=right_summary, 
            color=color_map[right_label], 
            legendgroup=right_label
        ))
        
        return fig
    
    def _show_graph_single(self, df: pd.DataFrame) -> go.Figure:
        summary = self._get_curve(df=df)
        summary = summary.reindex(self._unique_times_all[self._unique_times_all <= summary.index[-1]], method='ffill')
        fig = px.line(
            data_frame=summary.reset_index(names='time'),
            x='time',
            y='Estimate',
            title=f'Kaplan-Meier curve for this node', 
            hover_data=['At risk', 'Deaths', 'Censored', 'Upper 95%', 'Lower 95%'],
            color_discrete_sequence=['red']
        )
        fig.update_yaxes(range=[-0.05, 1])
        fig.update_xaxes(range=[0, None])
        fig.add_trace(self._get_confint(df=summary, color='red', legendgroup=None))
        return fig
    
    def show_graph_strata(self, df: pd.DataFrame, strata: List[str], col: str, threshold: float) -> go.Figure:
        if not strata or strata == ['']: return self.show_graph(df, col, threshold)
        n = len(strata)
        if n > 4 and n % 4 == 0:
            rows, cols = n // 4, 4
        else:
            rows, cols = 1, n
        fig = make_subplots(
            rows=rows, cols=cols, shared_yaxes=True, y_title='KM Estimate', x_title='time',
            subplot_titles=[stratum.replace(' & ', '<br>').replace('`', '') for stratum in strata]
        )
        
        for i, stratum in tqdm(enumerate(strata)):
            fig.add_traces(self.show_graph(df.query(stratum), col, threshold).data, rows=1+i//cols, cols=1+i%cols)
        fig.update_layout(showlegend=False)
        if rows > 1: fig.update_layout(height=300*rows, width=400*cols)

        for axis in fig['layout']:
            if axis.startswith('xaxis'): fig['layout'][axis]['range'] = [-0.05, df[self.y_col].max()]
            if axis.startswith('yaxis'): fig['layout'][axis]['range'] = [0, 1]
        return fig

def create_helper(forest_path: str) -> VisualizationHelper:

    forest = Forest.load(forest_path)
    if isinstance(forest, SurvivalForest):
        helper = VisualizationHelperSurvival(forest, forest_path)
    elif isinstance(forest, RegressionForest):
        helper = VisualizationHelperRegression(forest, forest_path)
    else:
        raise ValueError('Uknown forest type')
    return helper

helper: Optional[VisualizationHelper] = None
