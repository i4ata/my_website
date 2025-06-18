"""Base classes for nodes and leaves"""

import pandas as pd
import os
import shutil
import graphviz
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from typing import Optional, Tuple
from abc import ABC, abstractmethod
from time import time
import logging

# IT WOULD WORK ONLY FOR THE CURRENT PARAMS.YAML
TEMP_MAPPING = {
    'Strong Confounded Cause': 'Z1',
    'Weak Independent Cause': 'Z2',
    'Independent Cause': 'Z1',
    'Conditioned Cause': 'Z2',
    'Confounder 1': 'C1',
    'Confounder 2': 'C2',
    'Nested Confounder': 'C1',
    'Independent Cause 1': 'Z1',
    'Independent Cause 2': 'Z2',
    'Double Conditioned Cause': 'Z3',
    'Non-linear Cause': 'Z1'
}

class Node(ABC):
    """
    Base class for a node in a binary decision tree

    Attributes:
        node_id (int): unique identifier for the node
        feature (str|None): the feature associated with that node
        threshold (float|None): The threshold associated with that node (Condition: feature > threshold)
        left (Node|None): The left child
        right (Node|None): The right child
        prediction (float): The node's prediction for the target variable
        context (str): The conjunction of the node's ancestors' conditions, i.e. the path to the node. It can be directly used with `pandas.DataFrame.query(context)`
        terminal (bool): Whether the node is a leaf or not
        is_fitted (bool): Whether the node has been fitted or not

    Methods:
        __eq__(other: Node) -> bool: 2 nodes are equal if their selected features (if not leaf), left children, and right children are the same
        split(df: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]: Use the node to split a pandas dataframe. The first element is the samples where feature > threshold
    """

    node_id: int
    feature: Optional[str] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    prediction: Optional[float] = None
    context: str
    terminal: bool
    is_fitted: bool = False

    def __eq__(self, other: 'Node') -> bool:
        """Compare two nodes"""

        if self.terminal == other.terminal == True: return True
        return (
            not self.terminal and               # Current node is not a leaf
            not other.terminal and              # Other node is not a leaf
            self.feature == other.feature and   # Compare the splitting column
            self.left == other.left and         # Recursive call on left child
            self.right == other.right           # Recursive call on right child
        )
    
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a dataframe in 2 based on the node's learned feature and threshold
        
        Args:
            df (pandas.DataFrame): Rows are samples, columns are features

        Returns:
            [left,right] (tuple[pandas.DataFrame,pandas.DataFrame]): Left is the samples where feature > threshold, right is where feature <= threshold
        """

        assert not self.terminal
        assert self.is_fitted
        mask = df[self.feature] > self.threshold
        left_df, right_df = df[mask], df[~mask]
        return left_df, right_df

class FittableNode(Node):
    """
    Base class for a node that can be fitted on a dataset. Fit entails selecting an appropriate feature and a threshold
    
    Methods:
        fit(df: pandas.DataFrame, y_col: str): Fit the node on a dataset `df` with a target column `y_col`
    """

    def __init__(self, *, node_id: int, context: str) -> None:
        """
        Args:
            node_id (int): Unique identifier for the node
            context (str): The path to the current node
        """
        
        super().__init__()
        self.node_id = node_id
        self.context = context

    @abstractmethod
    def fit(self, *, df: pd.DataFrame, y_col: str, **kwargs):
        """
        Fit the node on a given dataset

        Args:
            df (pandas.DataFrame): The data available to the node. Rows are samples, columns are features
            y_col (str): The name of the target column
            **kwargs (dict[str,Any]): The parameters of the node
        """
        pass    

class Tree(ABC):
    """
    Base class for a binary decision tree

    Attributes:
        tree_id: (int): unique identifier for the tree
        root (Node): The root node
        y_col (str): The name of the target variable
        node_type (Node): The implementation of `Node` for the tree's nodes

    Methods:
        predict_single(sample: pandas.Series) -> float: Make a prediction for a single sample. The index are the feature names. It is the constant prediction of the leaf associated with that sample
        predict(df: pandas.DataFrame) -> pandas.Series: Make a prediction for multiple samples. The rows are the samples, the columns are the features. Returns a series with the same index, the values are the associated predictions 
        __eq__(other: Tree) -> bool: Compare 2 trees. 2 trees are the same if their roots are the same (refer to `Node.__eq__`)
        show_tree(df: pandas.DataFrame, extra: str, name: str, plot_leaf: bool, view: bool): Visualize the tree as a graphviz diagram
        
        _calculate_node_prediction(y: pandas.Series) -> float: Calculate a node's prediction given the associated subset of the target
        _plot_lead(node: Node, y: pandas.Series, file_path: str) -> visualize the leaf's prediction
    """

    tree_id: int = 0
    root: Node
    y_col: str
    node_type: type[Node]
    y_label: str = 'default_y_label' # Used to show when plotting the trees

    def predict_single(self, sample: pd.Series) -> float:
        """
        Use the tree to make a prediction for a given sample

        Args:
            sample (pandas.Series): The index are the feature names
        
        Returns:
            prediction (float): The constant prediction of the leaf node associated with the sample
        """

        # Pass the sample down the tree to reach the associated leaf
        current_node = self.root
        while not current_node.terminal:
            current_node = current_node.left if sample[current_node.feature] > current_node.threshold else current_node.right
        return current_node.prediction

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Use the tree to make a prediction for multiple samples

        Args:
            df (pandas.DataFrame): The rows are the samples, the columns are the features
        
        Returns:
            prediction (pandas.Series): The same index, the values are the associated prediction
        """
        return df.apply(self.predict_single, axis='columns').rename(f'y_pred_{self.tree_id}')

    def __eq__(self, other: 'Tree') -> bool:
        """Compare two trees"""

        return self.root == other.root

    @abstractmethod
    def _calculate_node_prediction(self, y: pd.Series) -> float:
        """
        Calculate the prediction for a node using the associated subset of targets (task specific)

        Args:
            y (pandas.Series): The training data targets associated with the node of interest

        Returns:
            prediction (float): The node's prediction inferred from the input
        """
        pass

    @abstractmethod
    def _plot_leaf(self, node: Node, y: pd.Series, file_path: str) -> None:
        """
        Visualize graphically a leaf's prediction (task specific)

        Args:
            node (Node): The associated leaf
            y (pandas.Series): The training data targets associated with that leaf
            file_path (str): Destination to save the plot
        """
        pass
    
    def show_tree(self, df: pd.DataFrame, extra: str = '', name: str = 'deafult_name', plot_leaf: bool = False, view: bool = False) -> None:
        """
        Visualize the entire tree graphically using graphviz

        Args:
            df (pandas.DataFrame): The entire training data
            extra (str): Extra to add to the save directory of the resulting plot. Currently saved at `resources/graphviz`
            name (str): How to name the resulting .pdf file
            plot_leaf (bool): Whether to show plots for the data at the leaves or just the prediction
            view (bool): Whether to immediately open the plot after creation
        """

        # Save PDF of graph here
        save_dir = os.path.join('resources', 'graphviz', extra)
        
        # Save temporary leaf plots here
        temp_dir = os.path.join(save_dir, name+'_plots')
        
        os.makedirs(temp_dir, exist_ok=True)

        graph = graphviz.Digraph() # Create an empty graph and populate it as we go down the tree

        # Function to traverse the tree
        # Populate the graph
        # Generate leaf plots if required
        def traverse(node: Node, df: pd.DataFrame) -> Node:
            node_id = str(node.node_id)
            node_df = df.query(node.context) if node.context else df.copy()

            # If the node is a leaf, stop the recursion, plot the leaf, save the plot
            if node.terminal:

                # Don't plot the leaf. Just display the prediction
                if not plot_leaf:
                    graph.node(node_id, f'n: {len(node_df)}\n{self.y_label}: {node.prediction:.2f}', shape='ellipse')
                    return node
                
                # Plot the leaf
                file_path = os.path.join(temp_dir, node_id + '.svg')
                self._plot_leaf(node, node_df[self.y_col], file_path)
                graph.node(
                    name=node_id,
                    label='',
                    image=os.path.abspath(file_path), 
                    shape='none'
                )
                return node
            
            # Add the node to the graph (Add the shortening of feature names for the plots)
            graph.node(node_id, f'{TEMP_MAPPING.get(node.feature, node.feature)} > {round(node.threshold, 3)}', shape='box')
            
            # Get left child, add it to the graph
            left = traverse(node.left, node_df)
            graph.edge(node_id, str(left.node_id), label='True' if node_id == '0' else '')

            # Get the right child, add it to the graph
            right = traverse(node.right, node_df)
            graph.edge(node_id, str(right.node_id), label='False' if node_id == '0' else '')

            return node
        traverse(self.root, df)
        
        # Render the graph and save it to memory
        # Remove all other temporary files
        tree_path = os.path.join(save_dir, name)
        graph.render(filename=tree_path, view=view, cleanup=True)
        shutil.rmtree(temp_dir)
        plt.close(plt.gcf())

class FittableTree(Tree):
    """
    Base class for a fittable tree

    Attributes:
        node_type (type[FittableNode]): The associated implementation fo `FittableNode`

    Methods:
        _build(df: pandas.DataFrame, depth: int, context: str) -> Node: Recursively build the tree from training data (ConstructTree in the Algorithm)
    """

    node_type: type[FittableNode]
    
    def __init__(self, *, tree_id: int, df: pd.DataFrame, y_col: str, **node_params) -> None:
        """
        Fit a whole tree on training data (ConstructTree)

        Args:
            tree_id (int): unique identifier for the tree
            df (pandas.DataFrame): The entire training data. Rows are samples and columns are features
            y_col (str): The name of the target variable
            node_params (dict[str,any]): The hyperparameters for the tree's nodes
        """

        assert y_col in df.columns
        self.tree_id = tree_id
        self.y_col = y_col
        self._node_params = node_params
        self._current_node_id = 0 # Assign unique IDs to the nodes, delete after
        
        # Construct the tree (this is ConstructTree in the Algorithm)
        self.root = self._build(df)    
        del self._current_node_id
        del self._node_params

    def _build(self, df: pd.DataFrame, context: str = '') -> Node:
        """
        Recursively fit the tree to the data (ConstructTree)

        Args:
            df (pandas.DataFrame): The current subset of the original training dataset (D')
            depth (int): The current depth of the tree (h)
            context (str): The path to the current node

        Returns:
            root (Node): The tree's root
        """

        # Create the node
        start = time()
        node = self.node_type(node_id=self._current_node_id, context=context)
        node.fit(df=df, y_col=self.y_col, **self._node_params)
        node.prediction = self._calculate_node_prediction(df[self.y_col])
        end = time()
        logging.debug(f'Tree {self.tree_id} | Building node {self._current_node_id} | time: {end - start:.4f} | data: {df.shape}')
        self._current_node_id += 1
        
        # Base case, a leaf was just made so stop splitting
        if node.terminal:
            return node
        
        # Split the data according to the node's condition
        # Left df = df | X > tau
        # right df = df | X <= tau
        left_df, right_df = node.split(df)

        # Recursively build the left and right children
        if context: context += ' & '
        node.left = self._build(left_df, context+f'`{node.feature}`>{node.threshold}')
        node.right = self._build(right_df, context+f'`{node.feature}`<={node.threshold}')

        return node

class RegressionTree(Tree):
    """
    Base class for a regression tree

    Methods:
        _plot_leaf(node: Node, y: pandas.Series, file_path: str): Plot the histogram for the target at that leaf
        _calculate_node_prediction(y: pandas.Series): The prediction is the mean target of all training samples associated with that leaf
    """
    y_label = 'y'

    def _plot_leaf(self, node: Node, y: pd.Series, file_path: str):
        """Plot the distribution of the target at the leaf"""

        assert node.terminal
        plt.clf()
        sns.histplot(y)
        plt.title(f'Samples: {len(y)}\nMean: {y.mean():.2f}')
        plt.savefig(file_path)

    def _calculate_node_prediction(self, y: pd.Series) -> float:
        """The prediction is simply the mean of all training samples' targets associated with the leaf"""

        return y.mean()

class SurvivalTree(Tree):
    """
    Base class for a survival tree

    Methods:
        _plot_leaf(node: Node, y: pandas.Series, file_path: str): Plot the survival function associated with that leaf
        _calculate_node_prediction(y: pandas.Series): The prediction is the risk, i.e. the sum of the CHF at that leaf
    """
    y_label = 'risk'

    def _plot_leaf(self, node: Node, y: pd.Series, file_path: str):
        """Plot the survival function estimation associated with that leaf"""
        assert node.terminal
        plt.clf()
        kmf = KaplanMeierFitter()
        kmf.fit(y.abs(), y >= 0)
        kmf.plot(color='red')
        plt.gca().legend().remove()
        
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\hat{S}(t)$')
        # plt.ylim(-.05, 1.05)
        plt.xlim(0, y.max())
        plt.title(f'Samples: {len(y)}\nRisk: {node.prediction:.2f}')
        plt.savefig(file_path)

    def _calculate_node_prediction(self, y: pd.Series) -> float:
        """The prediction is the sum of the CHF at the leaf. 
        To calculate it, it is required that the tree stores the unique observed times present in the entire training dataset"""
        assert hasattr(self, '_unique_times')
        risk = (
            NelsonAalenFitter().fit(y.abs(), timeline=self._unique_times, event_observed=y>=0)
            .cumulative_hazard_['NA_estimate']
            .sum()
        )
        return risk
