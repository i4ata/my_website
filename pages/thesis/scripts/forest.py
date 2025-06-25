"""
This class implements a Causal Survival/Regression Forest
"""

import pandas as pd
from sksurv.metrics import concordance_index_censored
import multiprocessing as mp
import os
import numpy as np
import pickle
from typing import Dict, Union, List, Any
from abc import ABC, abstractmethod
import logging

from pages.thesis.scripts.base import FittableTree
from pages.thesis.scripts.most_common_tree import MostCommonTree, MostCommonTreeRegression, MostCommonTreeSurvival
from pages.thesis.scripts.tree import CausalRegressionTree, CausalSurvivalTree, ScikitRegressionTree, ScikitSurvivalTree

# Make a wrapper that maps a dictionary to keyword arguments
def _tree_constructor_wrapper(kwargs: Dict[str, Any]):
    tree_cls = kwargs.pop('tree_type')
    return tree_cls(**kwargs)

class Forest(ABC):
    """
    Causal Survival/Regression Forest
    
    Attributes:
        tree_type (type[FittableTree]): The associated implementation of `FittableTree`
        most_common_tree_type (type[MostCommonTree]): The associated implementation of `MostCommonTree`
        y_col (str): The name of the target column / dependent variable
        df (pandas.DataFrame): The entire training data, must contain the column `y_col`
        bootstrap_random_states (list[int]): The random states used to bootstrap the data for each tree
        trees (list[FittableTree]): The set of trees in the forest
        predictions (pandas.Series): The forest's prediction for each sample of the training data
        metrics (dict[int,float|int]): A summary of the forest's performance on the training data
        full_data_tree (FittableTree): A tree that is fit on the entire (non-bootstrapped) training dataset
        popularity_counts (list[int]): For each tree, how often does a similar tree occur in the forest
        highest_frequency (int): The frequency of the most common tree
        most_common_tree (MostCommonTree): The most common tree in the forest

    Methods:
        predict (df: pandas.DataFrame): Use the forest to predict the outcomes of a set of samples
        __len__(): The size of the forest
        save(extra: str, name: str): Save the forest at `resources/models/<extra>/<name>.pkl`
        load(path: str): Load the forest from a .pkl file at `path`

        _get_metrics(): Evaluate the forest's predictive performance on the training dataset (task specific)
    """

    tree_type: type[FittableTree]
    most_common_tree_type: type[MostCommonTree]

    def __init__(self, *, df: pd.DataFrame, y_col: str, n_estimators: int, **tree_hparams) -> None:
        """
        Fit a forest

        Args:
            df (pandas.DataFrame): The entire training data with samples along the rows and variables along the columns
            y_col (str): The name of the target variable
            n_estimators (int): The size of the forest
            **tree_hparams (dict[str,any]): The hyperparameters shared by the trees
        """
        
        logging.info(f'Creating a forest of {n_estimators}')
        assert y_col in df.columns
        
        self.y_col = y_col
        self.df = df
        self.bootstrap_random_states = list(range(n_estimators))
        self.hparams = tree_hparams

        # Fit all trees independently in parallel
        with mp.Pool() as pool:
            self.trees: List[FittableTree] = pool.map(
                func=_tree_constructor_wrapper,
                iterable=[
                    {
                        'tree_type': self.tree_type,

                        'tree_id': i+1,

                        # The tree's training data. Bootstrapping
                        'df': df.sample(frac=1., replace=True, ignore_index=True, random_state=self.bootstrap_random_states[i]), 
                        
                        # Column in training data containing the ourcome
                        'y_col': self.y_col, 
                        
                        # The trees' hyperparameters
                        **tree_hparams
                    }
                    for i in range(n_estimators)
                ]
            )

        # Store the predictions of the training data
        # as a pandas.Series with the same index as the training df
        # and the values are the risk scores
        predictions = self.predict(df)

        # Calculate the metrics for the training data
        # If survival: cindex, concordant, discordant, tied_risk, tied_time
        # If regression: mse, r2
        self.metrics: Dict[str, Union[float, int]] = self._get_metrics(y=df[y_col], predictions=predictions)
        
        # Fit a tree on the original dataset
        logging.info('Fitting 1 tree on the entire dataset')
        self.full_data_tree = self.tree_type(tree_id=n_estimators, df=df, y_col=self.y_col, **tree_hparams)
        self.trees.append(self.full_data_tree)

        # The value at the i-th index is the number of times the i-th tree appears in the forest
        self.popularity_counts = [sum(tree == other for other in self.trees) for tree in self.trees]

        # The frequency of the most frequently occurring tree
        self.highest_frequency = max(self.popularity_counts)

        # Obtain the most common tree
        self.most_common_tree = self.most_common_tree_type(
            trees=[
                tree
                for i, tree in enumerate(self.trees)
                if self.popularity_counts[i] == self.highest_frequency
            ],
            df=self.df,
            y_col=self.y_col
        )

    @abstractmethod
    def _get_metrics(self, y: pd.Series, predictions: pd.Series) -> Dict[str, Union[float, int]]:
        """
        Compute the metrics associated with the forest's performance on the training data (task specific)

        Args:
            y (pandas.Series): The targets
            predictions (pandas.Series): The model's predictions. Same index as `y`

        Returns:
            metrics (dict[str,float|int]): The keys are the metric names, the values are the associated values
        """
        pass
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict the outcome of a set of samples.
        See `Tree.predict` for details.

        Args:
            df (pandas.DataFrame): The rows are the samples, the columns are features

        Returns:
            pandas.Series: Same index, the values are the associated predictions
        """

        return pd.concat(objs=[tree.predict(df) for tree in self.trees], axis='columns').T.mean().rename('y_pred')

    def __len__(self) -> int:
        return len(self.trees)

    def save(self, extra: str, name: str) -> None:
        """
        Save the forest as a .pkl file to memory. Save to `resources/models/<extra>/<name>.pkl`

        Args:
            extra (str): Additional paths to the total save path
            name (str): The name to give the model
        """

        save_dir = os.path.join('resources', 'models', extra)
        os.makedirs(save_dir, exist_ok=True)
        model_file_path = os.path.join(save_dir, f'{name}.pkl')
        logging.info(f'Saving forest to {model_file_path}')
        with open(model_file_path, 'wb') as f:
            pickle.dump(self, f)
        logging.info(f'Done! To visualize it, run python -m scripts.viz')
        logging.info(f'To delete it, run rm {model_file_path}')
        logging.info(f"To check the model's space on disk, run du -sh {model_file_path}")
        
    @classmethod
    def load(self, path: str) -> 'Forest':
        """
        Load the forest from a .pkl file

        Args:
            path (str): The relative path to a .pkl file

        Return:
            Forest: The forest object saved at `path`
        """
        
        # load_dir = os.path.join('resources', 'models', extra, f'{name}.pkl')
        assert os.path.exists(path), f"No file name {path}, aborting"
        logging.info(f'Loading forest from {path}')
        with open(path, 'rb') as f:
            instance = pickle.load(f)
        return instance

class RegressionForest(Forest):
    """Base class for a random forest used in regression tasks"""
    most_common_tree_type = MostCommonTreeRegression

    def _get_metrics(self, y: pd.Series, predictions: pd.Series) -> Dict[str, Union[float, int]]:
        """
        The forest's metrics are the R^2, MSE, MAE, and the MSE of the mean

        Args:
            y (pandas.Series): The regression outcome
            predictions (pandas.Series): The model's predictions. Same index as `y`

        Returns:
            metrics (dict[str,float]): keys are {R^2, MSE, MAE, MSE of the mean}
        """

        return {
            'R2': round(1 - np.sum((predictions - y) ** 2) / np.sum(y ** 2), ndigits=4),
            'MSE': round(np.mean((predictions - y) ** 2), ndigits=4),
            'MAE': round(np.mean(np.abs(predictions - y)), ndigits=4),
            'MSE of the mean': round(np.mean((y.mean() - y) ** 2), ndigits=4)
        }
    
class SurvivalForest(Forest):
    """Base class for a random forest used in survival tasks"""
    most_common_tree_type = MostCommonTreeSurvival

    def _get_metrics(self, y: pd.Series, predictions: pd.Series) -> Dict[str, Union[float, int]]:
        """
        The forest's metric is the C-Index

        Args:
            y (pandas.Series): The observed times (negative if censored)
            predictions (pandas.Series): The model's predicted risks. Same index as `y`

        Returns:
            metrics (dict[str,int|float]): The keys are {C-Index, Concordant pairs, Discordant pairs, Pairs with tied risk, Pairs with tied event time}
        """
        
        cindex, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
            event_indicator=y>=0,
            event_time=y.abs(),
            estimate=predictions
        )
        return {
            'C-Index': round(cindex, ndigits=4),
            'Concordant pairs': concordant,
            'Discordant pairs': discordant,
            'Pairs with tied risk': tied_risk,
            'Pairs with tied event time': tied_time
        }

class CausalRegressionForest(RegressionForest):
    """CET for regression"""
    tree_type = CausalRegressionTree

class CausalSurvivalForest(SurvivalForest):
    """CET for survival"""
    tree_type = CausalSurvivalTree

class ScikitRegressionForest(RegressionForest):
    """Scikit tree for regression"""
    tree_type = ScikitRegressionTree

class ScikitSurvivalForest(SurvivalForest):
    """Scikit tree for survival"""
    tree_type = ScikitSurvivalTree
