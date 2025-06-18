"""
The MostCommonTree is a regular binary tree.
At each node we have the variable and the distribution of the threshold
"""

import numpy as np
import pandas as pd

from pages.thesis.scripts.base import Tree, SurvivalTree, RegressionTree, Node
import logging

from typing import List, Optional

threshold_agg = np.median

class MostCommonNode(Node):
    """
    Base class for a node in a 'most common tree'

    Attributes:
        thresholds (list[float]): The threshold selected by each tree selected to infer the 'most common tree'
    """

    def __init__(
        self, *,
        node_id: int,
        context: str = '',
        feature: Optional[str] = None, 
        thresholds: Optional[List[float]] = None,
        terminal: bool = False
    ) -> None:
        """
        Create a node, simply set the attributes. The selected threshold is the median of all thresholds

        Args
            node_id (int): The node's unique identifier
            context (str): The path to the node
            feature (str): The feature selected by all trees at this node
            thresholds (list[float]): Each tree's selected threshold at this node
            terminal (bool): Whether the node is a leaf or not
        """

        super().__init__()
        self.is_fitted = True
        self.node_id = node_id
        self.context = context
        self.feature = feature
        self.terminal = terminal
        if not self.terminal:
            self.thresholds = thresholds
            self.threshold = threshold_agg(thresholds)
        
class MostCommonTree(Tree):
    """
    Base class representing a 'most common tree'
    
    Attributes
        node_type (MostCommonNode): Constant, the `Node` implementation used for this tree
    """

    def __init__(self, trees: List[Tree], df: pd.DataFrame, y_col: str) -> None:
        """
        Construct the most common tree

        Args
            trees (list[Tree]): The trees that most frequently appear in the forest
            df (pandas.DataFrame): The entire training data. The rows are the samples, the columns are the features
            y_col (str): The name of the target column
        """

        super().__init__()
        self.y_col = y_col
        self._current_node_id = -1 # First increment in _build so that it starts at 0
        self.node_type = MostCommonNode
        
        # Check for whether there are 2 different trees that occurr the maximum number of times
        flag = False
        for tree in trees:
            if flag: break
            for other in trees:
                if tree != other: 
                    flag = True
                    logging.warning('There are tied most popular trees | using only the first one')
                    break
        
        # Build the tree recursively as per usual
        roots = [trees[0].root] if flag else [tree.root for tree in trees]
        self.root = self._build(roots, df)
        del self._current_node_id

    def _build(self, roots: List[Node], df: pd.DataFrame, context: str = '') -> MostCommonNode:
        """
        Build the tree recursively as usual

        Args
            roots (list[Node]): The current nodes for all trees
            df (pandas.DataFrame): The training data available at this node
            context (str): The path to this node
        """

        # Make the new node
        self._current_node_id += 1
        thresholds = [root.threshold for root in roots]
        node = self.node_type(node_id=self._current_node_id, context=context, feature=roots[0].feature, thresholds=thresholds, terminal=roots[0].terminal)
        node.prediction = self._calculate_node_prediction(df[self.y_col])

        # We are at a leaf
        if node.terminal: return node
        
        # Recursively construct the left and right children
        if context: context += ' & '
        left, right = node.split(df)
        threshold = threshold_agg(thresholds)
        node.left = self._build([root.left for root in roots], left, context+f'`{roots[0].feature}`>{threshold}')
        node.right = self._build([root.right for root in roots], right, context+f'`{roots[0].feature}`<={threshold}')
        return node

class MostCommonTreeRegression(MostCommonTree, RegressionTree):
    """Most common tree to be used in regression tasks"""
    pass

class MostCommonTreeSurvival(MostCommonTree, SurvivalTree):
    """Most common tree to be used in survival tasks"""
    
    # Still unclear how to handle this in a better way. I just need to have access to the unique times at all points
    def __init__(self, trees: List[Tree], df: pd.DataFrame, y_col: str) -> None:
        self._unique_times = np.sort(df[y_col].abs().unique()) # Observed event times (ONLY uncensored)
        super().__init__(trees=trees, df=df, y_col=y_col)
        del self._unique_times