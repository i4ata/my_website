"""This module implements descision tree nodes equivalent to scikit-learn and scikit-survival"""

from abc import abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Literal, List
from enum import Enum
import scipy.stats as stats
from scipy.sparse import csr_array

from pages.thesis.scripts.base import FittableNode

class ScikitReasonForTerminal(Enum):
    """All possible reasons that a node is terminal (useful for visualization)"""

    max_depth = f'Maximum depth reached'
    min_samples_per_split = f'Not enough samples to split the node further'
    no_more_possible_splits = f'The are no more variables by which to split the data'

class ScikitNode(FittableNode):
    """
    Base class for regular scikit decision tree node
    
    Methods:
    - _get_independent_scores(df: pandas.DataFrame): Compute the independence scores for each splits
    - _get_independent_scores_post(X: numpy.ndarray, y: numpy.ndarray): Conduct independent statistical tests after properly preprocessing the training data (task-specific)
    """
    all_vars: List[str] # The variables available to the node
    
    def fit(
        self, *,
        df: pd.DataFrame, 
        y_col: str,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        p_value_correction: Literal['global', 'within_feature', None],
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Learn the node from the training data (i.e. the feature and the threshold)

        Args:
            df (pandas.DataFrame): The training data available at the node
            y_col (str): The name of the target column

            max_depth (int): Any node at that depth is a leaf
            min_samples_split (int): Any splits that would result in either subgroup having less samples than this is discarded
            min_samples_leaf (int): Any node with less samples than this is a leaf
            p_value_correction ('global'|'within_feature'|None): The way to do p-value correction. 'Global' entails that all p-values are multiplied by all tests conducted at the current stage. 'Within_feature' entails that the p-value for each threshold for each feature is multiplied by the unique values of that feature
            
        Returns:
            all_splits (pandas.DataFrame|None): If the node is not a leaf (or a leaf due to reaching maximum depth), a pandas dataframe with 2 indices: the features and their respective unique thresholds. The columns are the summary statistics from the tests
        """
        
        assert y_col in df.columns
        self.is_fitted = True
        depth = 0 if self.context == '' else self.context.count('&') + 1 # Infer the depth of the node using the context

        # OTHER ATTRIBUTES
        self.reason_for_terminal: Optional[ScikitReasonForTerminal] = None
        self.terminal = False
        self.y_col = y_col
        self.samples: int = len(df)
        
        # Remove all columns that have only a single unique value (impossible to split by them)
        df = df.loc[:, (df.nunique() > 1) | (df.columns == y_col)]
        self.all_vars: List[str] = sorted(df.columns[df.columns != y_col])
        
        # If we don't have any more variables with which to split the data, stop here
        if set(df.columns) == {y_col}:
            self.terminal = True
            self.reason_for_terminal = ScikitReasonForTerminal.no_more_possible_splits
            return None

        # If we don't have enough samples to split the data, stop here
        if self.samples < min_samples_split:
            self.terminal = True
            self.reason_for_terminal = ScikitReasonForTerminal.min_samples_per_split
            return None
        
        # Make the node terminal if the maximum depth is reached. Though compute the stats still
        if depth >= max_depth and not self.terminal and self.reason_for_terminal is None:
            self.terminal = True
            self.reason_for_terminal = ScikitReasonForTerminal.max_depth
            self.threshold, self.col = None, None

        # Compute sigma_ind in the Algorithm
        all_splits = self._get_independent_scores(df, min_samples_leaf=min_samples_leaf)
        
        # Do correction
        to_correct = all_splits.loc[all_splits['p'].notna()]
        all_splits.loc[to_correct.index, 'p_corrected'] = to_correct['p'] * (
            to_correct.groupby('variable')['p'].transform('count') if p_value_correction == 'within feature' else
            len(to_correct) if p_value_correction == 'global' else
            1
        )
        all_splits['p_corrected'] = all_splits['p_corrected'].clip(0, 1)

        # Get the best combination of feature and threshold
        best_comb = all_splits['z'].abs().idxmax()
        if best_comb is np.nan:
            self.feature, self.threshold = None, None
            self.terminal = True
            self.reason_for_terminal = ScikitReasonForTerminal.no_more_possible_splits
        else:
            self.feature, self.threshold = best_comb
        
        return all_splits
        
    @abstractmethod
    def _get_independent_scores_post(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the test statistic and p-value for each split given alread processed training data (task specific)

        Args:
            X (numpy.ndarray): A binary (n x m) array representing all possible ways to split the training data. (ij) is True if sample i belongs to group 1 in split j
            y (numpy.ndarray): A n-dimensional vector representing the target variable for each input sample

        Returns:
            [Z,p] (numpy.ndarray,numpy.ndarray): Both arrays are n-dimensional vectors. Z is the test statistics, p is the corresponding p-values
        """
        pass

    def _get_independent_scores(self, df: pd.DataFrame, min_samples_leaf: int) -> pd.DataFrame:
        """
        Evaluate each variable independently of the others (sigma_ind)

        Args:
            df (pandas.DataFrame): The entire training data available to the node
            min_samples_leaf (int): Any node with less samples than this is a leaf

        Returns:
            results (pandas.DataFrame): A pandas dataframe with 2 indices: all features and their respective thresholds. The 2 columns are the corresponding test statistic and p-value
        """

        # {column: sorted unique values of that column}
        uniques = {
            col: np.sort(df[col].unique())[:-1] 
            for col in df.columns if col != self.y_col
        }

        # Index 1: The variables (v)
        # Index 2: All thresholds in ascending order (t)
        # Columns: 1 per sample (i)
        # Values: True if the value of sample i's variable v is larger than the threshold t, else, False
        # True represents group 1, False represents group 0
        data = pd.concat(
            objs={
                col: pd.DataFrame(
                    df[col].to_numpy()[:, np.newaxis] > uniques[col][np.newaxis], columns=uniques[col]
                ).T
                for col in uniques
            },
            names=['variable', 'threshold']
        )

        # Determine the size of group 1 across all possible splits
        counts = data.sum(axis='columns')

        # Filter out all splits where either child would contain too few samples
        group_data = data.loc[(counts >= min_samples_leaf) & (counts <= (len(df) - min_samples_leaf))]

        # Make an empty dataframe to store the results
        results = pd.DataFrame(index=data.index, dtype=float, columns=['split', 'p', 'z', 'p_corrected'])
        
        # The column 'split' is the proportion of group 1 in that split
        results['split'] = counts / len(df)

        # If all leaves are uneven, return all NANs, this almost never happens
        if group_data.empty: return results
        
        # Binary (n, m)
        # element (i,j) is True, then sample i belongs to group 1 in split j
        # If it belongs to group 0, element (i,j) is False
        group_mask = group_data.T.to_numpy(dtype=bool) # (n, m) binary, the group membership

        # Do the thing (task-specific)
        Z, p = self._get_independent_scores_post(group_mask, df[self.y_col].to_numpy())
        
        # Record the results
        results.loc[group_data.index, ['p', 'z']] = np.stack((p, Z), axis=1)
        
        return results

class ScikitRegressionNode(ScikitNode):
    """Regular scikit-learn node that can be used for regression tasks"""

    def _get_independent_scores_post(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The independent quality of a split is given by a regular independent sample t-test. This is a simple vectorized implementation of it

        Args:
            X (numpy.ndarray): A binary (n x m) array representing all possible ways to split the training data. (ij) is True if sample i belongs to group 1 in split j
            y (numpy.ndarray): A n-dimensional vector representing the regression outcome for each input sample

        Returns:
            [Z,p] (numpy.ndarray,numpy.ndarray): Both arrays are n-dimensional vectors. Z is the t test statistics, p is the corresponding p-values
        """

        y_expanded = y[:, np.newaxis]
        gr0, gr1 = ~X, X
        gr0_size, gr1_size = gr0.sum(axis=0), gr1.sum(axis=0)
        gr0_mean, gr1_mean = (y_expanded * gr0).sum(axis=0) / gr0_size, (y_expanded * gr1).sum(axis=0) / gr1_size
        gr0_var, gr1_var = ((y_expanded - gr0_mean) ** 2 * gr0).sum(axis=0) / (gr0_size - 1), ((y_expanded - gr1_mean) ** 2 * gr1).sum(axis=0) / (gr1_size - 1)
        t = (gr1_mean - gr0_mean) / np.sqrt(gr0_var / gr0_size + gr1_var / gr1_size)

        ####### Degrees of freedom
        numerator = (gr0_var / gr0_size + gr1_var / gr1_size) ** 2
        denominator = ((gr0_var / gr0_size) ** 2 / (gr0_size - 1)) + ((gr1_var / gr1_size) ** 2 / (gr1_size - 1))
        df = numerator / denominator  # Correct Welch df
        #######
        
        p_value = 2 * stats.t.sf(np.abs(t), df=df)
        return t, p_value

class ScikitSurvivalNode(ScikitNode):
    """Regular scikit-survival node that can be used for survival modeling"""

    def _get_at_risk(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the number of people at risk for each observed time

        Args:
            x (numpy.ndarray): (t x m) int array, where (ij) is the number of observed events at time i in split j

        Returns:
            at_risk (numpy.ndarray): (t x m) int array, where (ij) is the number of samples at risk at time i in split j
        """

        at_risk = np.flipud(np.cumsum(np.flipud(x), axis=0))
        return at_risk

    def _get_independent_scores_post(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The independent quality of a split is given by a regular logrank test. This is a simple vectorized implementation of it


        Args:
            X (numpy.ndarray): A binary (n x m) array representing all possible ways to split the training data. (ij) is True if sample i belongs to group 1 in split j
            y (numpy.ndarray): A n-dimensional vector representing the observed time for each input sample (it's negative if the sample is censored)

        Returns:
            [Z,p] (numpy.ndarray,numpy.ndarray): Both arrays are n-dimensional vectors. Z is the logrank test statistics, p is the corresponding p-values
        """
        
        # Binary (n, 1)
        # element i is True if sample i experiences the event
        uncensored_mask = (y >= 0)[:, np.newaxis]
        
        # Real (t)
        # All unique times in ascending order
        tau = np.unique(np.abs(y))
        
        # Natural (n)
        # element i is the observed time of sample i (no matter if censored or uncensored)
        time_indices = np.searchsorted(tau, np.abs(y))

        # Binary (n, t)
        # element (i,j) is True if sample i's observed time is t, else False
        # So there is only 1 True value per row, All others are False
        comparison = csr_array(
            (np.ones_like(time_indices, dtype=float),  # Data (all ones)
            (np.arange(len(time_indices)), time_indices)),  # Row, Col indices
            shape=(len(time_indices), len(tau))
        )

        # Natural (t, m) int
        # Element (i,j) is the number of DEATHS at time i in split j
        d_0 = comparison.T @ (~X & uncensored_mask) # Group 0
        d_1 = comparison.T @ ( X & uncensored_mask) # Group 1
        d = d_0 + d_1 # Both added

        # Natural (t, m)
        # Element (i,j) is the number of samples AT RISK at time i in split j (i.e., alive and uncensored)
        N_0 = self._get_at_risk(comparison.T @ ~X) # Group 0
        N_1 = self._get_at_risk(comparison.T @  X) # Group 1
        N = N_0 + N_1 # Both added

        # Get the stats, just the formula for the logrank
        # Each of the following quantities if of shape (m)
        O = np.sum(d_1, axis=0)
        E_over_time = N_1 * d / N
        E = np.nansum(E_over_time, axis=0)
        V = np.nansum((N_0 * E_over_time * (N - d)) / (N * (N - 1)), axis=0)
        Z = (O - E) / np.sqrt(V)
        p_value = np.minimum(stats.norm.sf(np.abs(Z)) * 2, 1)
        
        return Z, p_value
