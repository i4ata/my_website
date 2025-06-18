"""This module implements nodes of a causal decision trees"""

from abc import abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List, Literal
from statsmodels.formula.api import mixedlm, ols
from statsmodels.base.wrapper import ResultsWrapper
from scipy.sparse import csr_array
import scipy.stats as stats
from enum import Enum
from pages.thesis.scripts.scikit_node import ScikitNode, ScikitRegressionNode, ScikitSurvivalNode
 
class ReasonForTerminal(Enum):
    """All possible reasons that a node is terminal (useful for visualization)"""

    ## These are repeated. TODO: Figure out a way how to do it without the duplication (Can't extend enum)
    max_depth = f'Maximum depth reached'
    min_samples_per_split = f'Not enough samples to split the node further'
    no_more_possible_splits = f'The are no more variables by which to split the data'
    ##
    
    no_significant_splits = f'There is no way to split the data in two significantly different groups'
    no_stratified_significant_splits = f'There is no way to split the data in two significantly different groups while accounting for confounding'
    
class CausalNode(ScikitNode):
    """
    Base class for causal decision tree node. It has 3 hyperparameters additional to scikit-learn:

    Attributes:
        strata (dict[str,list[str]]): The keys are all variables available to the node, the values are the variables by which the key variable is stratified in the second stage
        best_splits_per_var_independent_all (pandas.DataFrame): the index are the variables, there are 2 columns: 'threshold' -> (float) the best threshold for that variable; 'significant' -> (bool) whether that threshold is significant
    
    Methods:
        _get_stratified_scores_single(stratified_df: pandas.DataFrame): Compute the stratified score of one split (task specific)
        _get_stratified_scores(df: pandas.DataFrame, best_splits_per_var: pandas.Series): Compute the stratified scores for the best thresholds of each variable (sigma_strat)
        _stratify(df: pandas.DataFrame, col: str, best_splits: pandas.Series): Stratify the training data based on the other features
    """

    def fit(
        self, *,
        df: pd.DataFrame, 
        y_col: str,
        min_samples_stratum: int,
        alpha_ind: float,
        alpha_strat: float,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        p_value_correction: Literal['global', 'within_feature', None],
        **kwargs
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, List[str]]], Optional[pd.DataFrame]]:
        """
        Learn the node from the training data (i.e. the feature and the threshold)

        Args:
            df (pandas.DataFrame): The training data available at the node
            y_col (str): The name of the target column
            
            alpha_ind (float): Significance threshold of the independent statistical tests
            alpha_strat (float): Significance threshold of the stratified statistical tests
            min_samples_stratum (int): Stratify until the smalles stratum would have at most this many samples
            
            max_depth (int): Any node at that depth is a leaf
            min_samples_split (int): Any splits that would result in either subgroup having less samples than this is discarded
            min_samples_leaf (int): Any node with less samples than this is a leaf
            p_value_correction ('global'|'within_feature'|None): The way to do p-value correction. 'Global' entails that all p-values are multiplied by all tests conducted at the current stage. 'Within_feature' entails that the p-value for each threshold for each feature is multiplied by the unique values of that feature
            
        Returns:
            all_splits (pandas.DataFrame|None): Return value of ScikitNode.fit
            strata (dict[str,list[str]]|None): The keys are the variables, the values are the variables used for stratification
            stratified_scores (pandas.DataFrame): The rows are the variables and best thresholds, the columns are the summary statistics
        """
        
        all_splits: Optional[pd.DataFrame] = super().fit(
            df=df, y_col=y_col, 
            max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
            p_value_correction=p_value_correction
        )
        self.strata: Dict[str, List[str]] = {}
        if all_splits is None:
            return None, None, None
        assert 'stratum' not in df.columns

        # TODO: This is repeated in scikit_node, dunno how to avoid that without saving depth as an attribute
        depth = 0 if self.context == '' else self.context.count('&') + 1

        best_splits_per_var_independent = (
            all_splits
            .query('p_corrected < @alpha_ind')                      # Get rows corresponding to significant splits
            ['z']                                                   # Use only the z statistic
            .abs()                                                  # We look at its magnitude
            .groupby('variable')                                    # For each variable
            .agg(['idxmax', 'max'])                                 # Get the best |Z| and its corresponding threshold for each variable
            .rename(columns={'idxmax': 'threshold', 'max': '|z|'})  # Rename to make a bit more sense
            .sort_values(by='|z|', ascending=False)                 # Sort the best thresholds per variable by significance in descending order
            ['threshold']                                           # Extract the threshold
            .apply(lambda x: x[1])                                  # Currently it's (var, threshold), so get only the threshold
            .rename('threshold')                                    # Rename to make sense
        )
        
        ##########################################################################################
        # THIS IS A BIT SCUFFED. IT'S USED IN THE VISUALIZATION ONLY
        def find_best_split(group: pd.DataFrame):
            best_split = (
                group.loc[(group['split'] - 0.5).abs().idxmin()]
                if group['z'].isnull().all() else
                group.loc[group['z'].abs().idxmax()]
            )
            return pd.Series({
                'significant': best_split['p_corrected'] < 0.05,
                'threshold': best_split.name[1]
            })
        self.best_splits_per_var_independent_all = all_splits.groupby('variable').apply(find_best_split)
        ##########################################################################################

        # If no significant splits exist, stop here
        if len(best_splits_per_var_independent) == 0:
            self.terminal = True
            self.reason_for_terminal = ReasonForTerminal.no_significant_splits
            return all_splits, None, None
        
        # Sigma_strat in the Algorithm
        strata, stratified_scores = self._get_stratified_scores(
            df=df,
            best_splits_per_var=best_splits_per_var_independent,
            min_samples_stratum=min_samples_stratum
        )
        stratified_scores['p_corrected'] = (stratified_scores['p'] * (len(stratified_scores) if p_value_correction == 'global' else 1)).clip(0, 1)

        self.strata = strata # TEMPORARY (I HOPE)
        stratified_scores_filtered = stratified_scores.query('p_corrected < @alpha_strat')
        
        # If again no significant splits, stop here
        if len(stratified_scores_filtered) == 0:
            self.terminal = True
            self.reason_for_terminal = ReasonForTerminal.no_stratified_significant_splits
            return all_splits, strata, stratified_scores_filtered

        self.feature = stratified_scores_filtered['z'].abs().idxmax()
        self.threshold = stratified_scores_filtered.loc[self.feature, 'threshold']

        # Make the node terminal if the maximum depth is reached TODO: This is also repeated in scikit_node
        if depth >= max_depth and not self.terminal and self.reason_for_terminal is None:
            self.terminal = True
            self.reason_for_terminal = ReasonForTerminal.max_depth
            self.threshold, self.feature = None, None
        
        return all_splits, strata, stratified_scores

    @abstractmethod
    def _get_stratified_scores_single(self, stratified_df: pd.DataFrame) -> pd.Series:
        """
        The implementation of sigma_strat in the Algorithm (task specific)

        Args:
            stratified_df (pandas.DataFrame): The indices are the samples. The columns are {survival time, event indicator, stratum, split}
        
        Returns:
            pandas.Series: The summary statistics. Must contain 'z' and 'p'
        """
        pass

    def _get_stratified_scores(self, df: pd.DataFrame, best_splits_per_var: pd.Series, min_samples_stratum: int) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
        """
        The implementation of sigma_strat in the Algorithm for each variable.
        
        Args:
            df (pandas.DataFrame): The data available to the current node
            best_splits_per_var (pandas.Series): The indices are the variables, the values are the best thresholds for each variable. They are ordered by significance in descending order
            min_samples_stratum (int): Stratify until the smalles stratum would have at most this many samples
            
        Returns:
            {var: List[str]}: all stratifying conditions for each variable
            pandas.DataFrame: The results from the test. The indices are the variables, the columns are the statistics
        """

        strata: Dict[str, List[str]] = {}
        results: Dict[str, pd.Series] = {}

        # Unfortunately it has to be a for-loop because each variable can be stratified by a different number of other variables
        for col in best_splits_per_var.index:
            stratified_df = self._stratify(df, col, best_splits_per_var, min_samples_stratum=min_samples_stratum)
            strata[col] = sorted(stratified_df['stratum'].unique())
            results[col] = self._get_stratified_scores_single(stratified_df)
            results[col]['threshold'] = best_splits_per_var[col]
        results = pd.concat(results, axis='columns').T
        return strata, results

    def _stratify(self, df: pd.DataFrame, col: str, best_splits: pd.Series, min_samples_stratum: int) -> pd.DataFrame:
        """
        The Stratify procedure in the Algorithm.
        Stratify the data for `col` based on all other columns in order as much as possible.

        Args:
            df (pandas.DataFrame): The data at the current node
            col (str): The name of the column that we want to score right now
            best_splits (pandas.Series): The index are the variables, the values are the best threshold for that variable sorted by significance in descending order
            min_samples_stratum (int): Stratify until the smalles stratum would have at most this many samples
            
        Returns:
            pandas.DataFrame: The index is the same as `df`, there are 4 columns: {event time, event indicator, stratum, split}
        """
        
        assert {self.y_col, col}.issubset(df.columns)
        assert set(best_splits.index).issubset(df.columns)
        temp_df = df[[self.y_col]].assign(split=df[col] > best_splits[col], stratum='')
        prefix = col.split('_', maxsplit=1)[0] # To know if the variable has been one-hot encoded
        flag = False # To know if we already did at least 1 iteration
        for other_col in best_splits.index:

            # If the variable was one-hot encoded, skip it, since it makes no sense to stratify
            if other_col.startswith(prefix): continue

            # Stratify the data by that column (big brain implementation)
            temp_df['stratum'] += (df[other_col] > best_splits[other_col]).map({
                True: f'`{other_col}` > {best_splits[other_col]}',
                False: f'`{other_col}` <= {best_splits[other_col]}'
            })
            # If we reached the minimum samples per stratum, go back one step
            # If this happens in the first step, there is no strata
            if np.min(pd.crosstab(temp_df['stratum'], temp_df['split'])) < min_samples_stratum:
                temp_df['stratum'] = temp_df['stratum'].str.rsplit(' & ', n=1).str[0] if flag else ''
                break # TODO: leaving this break out might actually be good. that way we stratify by as many variables as possible
            flag = True
            temp_df['stratum'] += ' & '

        # Case when we looped over everything, remove the last ' & '
        else:
            temp_df['stratum'] = temp_df['stratum'].str.rsplit(' & ', n=1).str[0]
        return temp_df

class CausalRegressionNode(CausalNode, ScikitRegressionNode):
    """Causal node to be used in regression tasks"""

    def _get_summary(self, model_fit: ResultsWrapper) -> pd.Series:
        return pd.Series({'z': model_fit.tvalues['split[T.True]'].astype(float), 'p': model_fit.pvalues['split[T.True]'].astype(float)})

    def _get_stratified_scores_single(self, stratified_df: pd.DataFrame) -> pd.Series:
        """
        The implementation of sigma_strat. This is determined by a regular linear mixed model

        Args:
            stratified_df (pandas.DataFrame): The indices are the samples. The columns are {survival time, event indicator, stratum, split}
        
        Returns:
            pandas.Series: The summary statistics
        """

        assert set(stratified_df.columns) == {'stratum', self.y_col, 'split'}

        # If the data is not stratified, do a regular t-test (TODO: The test is already done before, reuse the results somehow)
        if stratified_df['stratum'].nunique(dropna=False) == 1:
            gr1, gr2 = stratified_df[stratified_df['split']][self.y_col], stratified_df[~stratified_df['split']][self.y_col]
            test = stats.ttest_ind(gr1, gr2, equal_var=False)
            return pd.Series({'z': test.statistic, 'p': test.pvalue})
            # model_fit = ols(f'{self.y_col} ~ split', data=stratified_df).fit()
            # return self._get_summary(model_fit)

        # Temporary bullshit (maybe not temporary lol)
        try:
            model_fit = mixedlm(f'{self.y_col} ~ split', stratified_df, groups=stratified_df['stratum']).fit()
            return self._get_summary(model_fit)
        except Exception as e:
            model_fit = mixedlm(f'{self.y_col} ~ split', stratified_df, groups=stratified_df['stratum']).fit_regularized(alpha=.1, ptol=1e-4)
            return self._get_summary(model_fit)
            
class CausalSurvivalNode(CausalNode, ScikitSurvivalNode):
    """Causal node to be used in survival tasks"""

    def _get_stratified_scores_single(self, stratified_df: pd.DataFrame) -> pd.Series:
        """
        The implementation of sigma_strat. This is determined by the stratified logrank test

        Args:
            stratified_df (pandas.DataFrame): The indices are the samples. The columns are {survival time, event indicator, stratum, split}
        
        Returns:
            pandas.Series: The summary statistics
        """
        
        assert set(stratified_df.columns) == {'stratum', self.y_col, 'split'}

        # DEFINITIONS:
        # n: total number of samples
        # m: the number of strata
        # t: number of all unique survival times (censored and uncensored)

        # Real (t)
        # All unique times in ascending order
        tau = np.sort(stratified_df[self.y_col].abs().unique())
        
        # Binary [n, m] 
        # The stratum membership one-hot encoded
        # element (i, j) is true if sample i belongs to stratum j
        # All strata are mutually exclusive
        strata = csr_array(pd.get_dummies(stratified_df['stratum']), dtype=float)

        # Natural (n)
        # element i is the observed time of sample i (no matter if censored or uncensored)
        time_indices = np.searchsorted(tau, stratified_df[self.y_col].abs().to_numpy())

        # Binary (n, t)
        # element (i,j) is True if sample i's observed time is t, else False
        # So there is only 1 True value per row, All others are False
        comparison = csr_array(
            (np.ones_like(time_indices, dtype=bool),
            (np.arange(len(time_indices)), time_indices)),
            shape=(len(time_indices), len(tau))
        )

        # Binary (n)
        # Group assignment
        # element i is True if sample i belongs to group 1, else, False
        split = stratified_df['split'].to_numpy(dtype=bool)

        # Binary (n)
        # element i is True if sample i experiences the event
        event_indicator = (stratified_df[self.y_col] >= 0).to_numpy()
        
        # Natural (t, m) int
        # Element (i,j) is the number of DEATHS at time i in stratum j
        d_0 = ((comparison.T * (~split & event_indicator)) @ strata).toarray() # Group 0
        d_1 = ((comparison.T * ( split & event_indicator)) @ strata).toarray() # Group 1
        d = d_0 + d_1 # Group 0 + Group 1
        
        # Natural (t, m)
        # Element (i,j) is the number of samples AT RISK at time i in split j (i.e., alive and uncensored)
        N_0 = self._get_at_risk(((comparison.T * ~split) @ strata).toarray())
        N_1 = self._get_at_risk(((comparison.T *  split) @ strata).toarray())
        N = N_0 + N_1

        # Real (m) 
        # Compute observed, expected value, and variance of the hypergeometric distribution
        O = np.sum(d_1, axis=0)
        E_over_time = N_1 * d / N
        E = np.nansum(E_over_time, axis=0)
        V = np.nansum((N_0 * E_over_time * (N - d)) / (N * (N - 1)), axis=0)
    
        # Real (1) 
        # Get Z-statistic
        Z = np.sum(O - E) / np.sqrt(np.sum(V))
        p_value = np.minimum(stats.norm.sf(np.abs(Z)) * 2, 1)

        return pd.Series({'z': Z, 'p': p_value})
