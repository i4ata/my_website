"""This module defines the trees"""

import pandas as pd
import numpy as np

from pages.thesis.scripts.base import FittableTree, RegressionTree, SurvivalTree
from pages.thesis.scripts.scikit_node import ScikitRegressionNode, ScikitSurvivalNode
from pages.thesis.scripts.causal_node import CausalRegressionNode, CausalSurvivalNode

class ScikitRegressionTree(FittableTree, RegressionTree):
    node_type = ScikitRegressionNode

class ScikitSurvivalTree(FittableTree, SurvivalTree):
    node_type = ScikitSurvivalNode

    def __init__(self, *, tree_id: int, df: pd.DataFrame, y_col: str, **node_params):
        
        self._unique_times = np.sort(df[y_col].abs().unique()) # Observed event times (ONLY uncensored)
        super().__init__(tree_id=tree_id, df=df, y_col=y_col, **node_params)
        del self._unique_times

class CausalRegressionTree(ScikitRegressionTree):
    node_type = CausalRegressionNode

class CausalSurvivalTree(ScikitSurvivalTree):
    node_type = CausalSurvivalNode
