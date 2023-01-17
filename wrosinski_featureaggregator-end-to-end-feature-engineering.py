import gc
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

%matplotlib inline
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
class FeatureAggregator(object):

    """Feature aggregator - automated feature aggregation method.
    Two ways of usage, either selected aggregations can be applied onto
    numerical and categorical columns or specific combinations of aggregates
    can be set for each column.

    # Arguments:
        df: (pandas DataFrame), DataFrame to create features from.
        aggregates_cat: (list), list containing aggregates for
            categorical features
        aggregates_num: (list), list containing aggregates for
            numerical features.

    """

    def __init__(self,
                 df,
                 aggregates_cat=['mean', 'std'],
                 aggregates_num=['mean', 'std', 'sem', 'min', 'max']):

        self.df = df.copy()
        self.aggregates_cat = aggregates_cat
        self.aggregates_num = aggregates_num

    def process_features_batch(self,
                               categorical_columns=None,
                               categorical_int_columns=None,
                               numerical_columns=None,
                               to_group=['SK_ID_CURR'], prefix='BUREAU'):
        """Process, group features in batch.

        # Arguments:
            categorical_columns: (list), list of categorical columns, which need
            to be label-encoded (factorized).
            categorical_int_columns: (list), list of categorical columns, which
            are already of integer type.
            numerical_columns: (list), list of numerical columns.
            to_group: (list), list of columns to group by.
            prefix: (string), prefix for columns names.

        # Returns:
            df_cat/df_num: (pandas DataFrame), DataFrame with aggregated columns.

        """

        assert isinstance(
            to_group, list), 'Variable to group by must be of type list.'

        if categorical_columns is not None:
            assert len(categorical_columns) > 0, 'No columns to encode.'
            self.categorical_features_factorize(categorical_columns)
            df_cat = self.create_aggregates_set(
                columns=categorical_columns,
                aggregates=self.aggregates_cat,
                to_group=to_group, prefix=prefix)
            print('\nAggregated df_cat shape: {}'.format(df_cat.shape))
            return df_cat

        if categorical_int_columns is not None:
            assert len(categorical_int_columns) > 0, 'No columns to encode.'
            df_cat = self.create_aggregates_set(
                columns=categorical_int_columns,
                aggregates=self.aggregates_cat,
                to_group=to_group, prefix=prefix)
            print('\nAggregated df_cat int shape: {}'.format(df_cat.shape))
            return df_cat

        if numerical_columns is not None:
            assert len(numerical_columns) > 0, 'No columns to encode.'
            df_num = self.create_aggregates_set(
                columns=numerical_columns,
                aggregates=self.aggregates_num,
                to_group=to_group, prefix=prefix)
            print('\nAggregated df_num shape: {}'.format(df_num.shape))
            return df_num

        return

    def process_features_selected(self,
                                  aggregations,
                                  categorical_columns,
                                  to_group=['SK_ID_CURR'], prefix='BUREAU'):
        """Process, group features for selected combinations of aggregates
        and columns.

        # Arguments:
            categorical_columns: (list), list of categorical columns, which need
            to be label-encoded (factorized).
            to_group: (list), list of columns to group by.
            prefix: (string), prefix for columns names.

        # Returns:
            df_agg: (pandas DataFrame), DataFrame with aggregated columns.

        """

        assert isinstance(
            to_group, list), 'Variable to group by must be of type list.'

        if categorical_columns:
            # Provide categorical_columns argument if some features need to be factorized.
            self.categorical_features_factorize(categorical_columns)

        df_agg = self.create_aggregates_set(
            aggregations=aggregations,
            to_group=to_group, prefix=prefix)

        print('\nAggregated df_agg shape: {}'.format(df_agg.shape))

        return df_agg

    def create_aggregates_set(self,
                              aggregations=None,
                              columns=None,
                              aggregates=None,
                              to_group=['SK_ID_CURR'],
                              prefix='BUREAU'):
        """Create selected aggregates.

        # Arguments:
            aggregations: (dict), dictionary specifying aggregates for selected columns.
            columns: (list), list of columns to group for batch aggregation.
            aggregates: (list), list of aggregates to apply on columns argument
            for batch aggregation.
            to_group: (list), list of columns to group by.
            prefix: (string), prefix for columns names.

        # Returns:
            df_agg: (pandas DataFrame), DataFrame with aggregated columns.

        """

        assert isinstance(
            to_group, list), 'Variable to group by must be of type list.'

        if aggregations is not None:
            print('Selected aggregations:\n{}\n.'.format(aggregations))
            df_agg = self.df.groupby(
                to_group).agg(aggregations)

        if columns is not None and aggregates is not None:
            print('Batch aggregations on columns:\n{}\n.'.format(columns))
            df_agg = self.df.groupby(
                to_group)[columns].agg(aggregates)

        df_agg.columns = pd.Index(['{}_{}_{}'.format(
            prefix, c[0], c[1].upper()) for c in df_agg.columns.tolist()])
        df_agg = df_agg.reset_index()

        return df_agg

    def get_column_types(self):
        """Select categorical (to be factorized), categorical integer and numerical
        columns based on their dtypes. This facilitates proper grouping and aggregates selection for
        different types of variables.
        Categorical columns needs to be factorized, if they are not of
        integer type.

        # Arguments:
            self.df: (pandas DataFrame), DataFrame to select variables from.

        # Returns:
            categorical_columns: (list), list of categorical columns which need factorization.
            categorical_columns_int: (list), list of categorical columns of integer dtype.
            numerical_columns: (list), list of numerical columns.
        """

        categorical_columns = [
            col for col in self.df.columns if self.df[col].dtype == 'object']
        categorical_columns_int = [
            col for col in self.df.columns if self.df[col].dtype == 'int']
        numerical_columns = [
            col for col in self.df.columns if self.df[col].dtype == 'float']

        categorical_columns = [
            x for x in categorical_columns if 'SK_ID' not in x]
        categorical_columns_int = [
            x for x in categorical_columns_int if 'SK_ID' not in x]

        print('DF contains:\n{} categorical object columns\n{} categorical int columns\n{} numerical columns.\n'.format(
            len(categorical_columns), len(categorical_columns_int), len(numerical_columns)))

        return categorical_columns, categorical_columns_int, numerical_columns

    def categorical_features_factorize(self, categorical_columns):
        """Factorize categorical columns, which are of non-number dtype.

        # Arguments:
            self.df: (pandas DataFrame), DataFrame to select variables from.
            Transformation is applied inplace.

        """

        print('\nCategorical features encoding: {}'.format(categorical_columns))

        for col in categorical_columns:
            self.df[col] = pd.factorize(self.df[col])[0]

        print('Categorical features encoded.\n')

        return

    def check_and_save_file(self, df, filename, dst='../input/'):
        """Utility function to check if there isn't a file with the same name already.

        # Arguments:
            df: (pandas DataFrame), DataFrame to save.
            filename: (string), filename to save DataFrame with.

        """

        filename = '{}{}.pkl'.format(dst, filename)
        if not os.path.isfile(filename):
            print('Saving: {}'.format(filename))
            df.to_pickle('{}'.format(filename))
        return


def feature_aggregator_on_df(df,
                             aggregates_cat,
                             aggregates_num,
                             to_group,
                             prefix,
                             suffix='basic',
                             save=False,
                             categorical_columns_override=None,
                             categorical_int_columns_override=None,
                             numerical_columns_override=None):
    """Wrapper for FeatureAggregator to process dataframe end-to-end using batch aggregation.
    It takes lists of aggregates for categorical and numerical features, which are created for
    selected column (to_group), by which data is grouped. In addition to that, prefix and suffix can
    be provided to facilitate column naming.
    _override arguments can be used if only selected subset of each type of columns should
    be aggregated. If those are not provided, FeatureAggregator processes all columns for each type.

        # Arguments:
            aggregates_cat: (list), list of aggregates to apply to categorical features.
            aggregates_num: (list), list of aggregates to apply to numerical features.
            to_group: (list), list of columns to group by.
            prefix: (string), prefix for column names.
            suffix: (string), suffix for filename.
            save: (boolean), whether to save processed DF.
            categorical_columns_override: (list), list of categorical columns
            to override default, inferred list.
            categorical_int_columns_override: (list), list of categorical integer
            columns to override default, inferred list.
            numerical_columns_override: (list), list of numerical columns
            to override default, inferred list.

        # Returns:
            to_return: (list of pandas DataFrames), DataFrames with aggregated columns,
            one for each type of column types. This is due to the fact that not every
            raw dataframe may contain all types of columns.

        """

    assert isinstance(aggregates_cat, list), 'Aggregates must be of type list.'
    assert isinstance(aggregates_num, list), 'Aggregates must be of type list.'

    t = time.time()
    to_return = []

    column_base = ''
    for i in to_group:
        column_base += '{}_'.format(i)

    feature_aggregator_df = FeatureAggregator(
        df=df,
        aggregates_cat=aggregates_cat,
        aggregates_num=aggregates_num)

    print('DF prefix: {}, suffix: {}'.format(prefix, suffix))
    print('Categorical aggregates - {}'.format(aggregates_cat))
    print('Numerical aggregates - {}'.format(aggregates_num))

    df_cat_cols, df_cat_int_cols, df_num_cols = feature_aggregator_df.get_column_types()

    if categorical_columns_override is not None:
        print('Overriding categorical_columns.')
        df_cat_cols = categorical_columns_override
    if categorical_columns_override is not None:
        print('Overriding categorical_int_columns.')
        df_cat_int_cols = categorical_int_columns_override
    if categorical_columns_override is not None:
        print('Overriding numerical_columns.')
        df_num_cols = numerical_columns_override

    if len(df_cat_cols) > 0:
        df_curr_cat = feature_aggregator_df.process_features_batch(
            categorical_columns=df_cat_cols,
            to_group=to_group, prefix=prefix)
        if save:
            feature_aggregator_df.check_and_save_file(
                df_curr_cat, '{}_cat_{}_{}'.format(prefix, column_base, suffix))
        to_return.append(df_curr_cat)
        del df_curr_cat
        gc.collect()

    if len(df_cat_int_cols) > 0:
        df_curr_cat_int = feature_aggregator_df.process_features_batch(
            categorical_int_columns=df_cat_int_cols,
            to_group=to_group, prefix=prefix)
        if save:
            feature_aggregator_df.check_and_save_file(
                df_curr_cat_int, '{}_cat_int_{}_{}'.format(prefix, column_base, suffix))
        to_return.append(df_curr_cat_int)
        del df_curr_cat_int
        gc.collect()

    if len(df_num_cols) > 0:
        df_curr_num = feature_aggregator_df.process_features_batch(
            numerical_columns=df_num_cols,
            to_group=to_group, prefix=prefix)
        if save:
            feature_aggregator_df.check_and_save_file(
                df_curr_num, '{}_num_{}_{}'.format(prefix, column_base, suffix))
        to_return.append(df_curr_num)
        del df_curr_num
        gc.collect()

    print('\nTime it took to create features on df: {:.3f}s'.format(
        time.time() - t))

    return to_return


def feature_aggregator_on_df_selected(df,
                                      aggregations,
                                      to_group,
                                      prefix,
                                      suffix='basic',
                                      save=False):
    """Wrapper for FeatureAggregator to process dataframe end-to-end using selected
    aggregates/columns combinations.
    It takes dictionary of aggregates/columns combination for selected features,
    which are created for selected column (to_group), by which data is grouped.
    In addition to that, prefix and suffix can be provided to facilitate column naming.

        # Arguments:
            aggregations: (dict), dictionary containing combination of columns/aggregates.
            to_group: (list), list of columns to group by.
            prefix: (string), prefix for column names.
            suffix: (string), suffix for filename.
            save: (boolean), whether to save processed DF.

        # Returns:
            to_return: (list of pandas DataFrames), DataFrames with aggregated columns,
            one for each type of column types. This is due to the fact that not every
            raw dataframe may contain all types of columns.

        """

    assert isinstance(
        to_group, list), 'Variable to group by must be of type list.'

    t = time.time()
    to_return = []

    column_base = ''
    for i in to_group:
        column_base += '{}_'.format(i)

    feature_aggregator_df = FeatureAggregator(df=df)

    print('DF prefix: {}, suffix: {}'.format(prefix, suffix))

    df_cat_cols, df_cat_int_cols, df_num_cols = feature_aggregator_df.get_column_types()

    if len(df_cat_cols) > 0:
        df_aggs = feature_aggregator_df.process_features_selected(
            aggregations=aggregations,
            categorical_columns=df_cat_cols,
            to_group=to_group,
            prefix=prefix)
    else:
        df_aggs = feature_aggregator_df.process_features_selected(
            aggregations=aggregations,
            to_group=to_group,
            prefix=prefix)

    if save:
        feature_aggregator_df.check_and_save_file(
            df_aggs, '{}_selected_{}_{}'.format(prefix, column_base, suffix))

    to_return.append(df_aggs)
    del df_aggs
    gc.collect()

    print('\nTime it took to create features on df: {:.3f}s'.format(
        time.time() - t))

    return to_return

aggs1 = ['mean', 'median', 'min', 'max', 'count', 'std', 'sem', 'sum', 'mad']
aggs2 = ['mean', 'median', 'min', 'max', 'count', 'std', 'sem', 'sum']

aggs_num_basic = ['mean', 'min', 'max', 'std', 'sem', 'sum']
aggs_cat_basic = ['mean', 'std', 'sum']

aggs_num = aggs_num_basic
aggs_cat = aggs_cat_basic
train = pd.read_csv("../input/application_train.csv") 
test = pd.read_csv("../input/application_test.csv")

bureau = pd.read_csv("../input/bureau.csv")
bureau_bal = pd.read_csv('../input/bureau_balance.csv')
%%time

bureau_bal_dfs = feature_aggregator_on_df(
    bureau_bal, aggs_cat, aggs_num, ['SK_ID_BUREAU'], 'bureau_balance', 'basic', save=False)

bureau_bal_dfs[0]
%%time

bureau_ = bureau.merge(bureau_bal_dfs[0], how='left', on='SK_ID_BUREAU', copy=False)
bureau_ = bureau_.merge(bureau_bal_dfs[1], how='left', on='SK_ID_BUREAU', copy=False)

bureau_dfs = feature_aggregator_on_df(
    bureau_, aggs_cat, aggs_num, ['SK_ID_CURR'], 'bureau', 'basic', save=False)

bureau_dfs[1]
%%time

prev = pd.read_csv("../input/previous_application.csv")
prev = prev.drop(['SK_ID_PREV'], axis=1)

prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']


num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }

prev_selected_aggs = feature_aggregator_on_df_selected(
    prev, num_aggregations, to_group=['SK_ID_CURR'], prefix='prev', suffix='basic_selected', save=False)
cred_card_bal = pd.read_csv("../input/credit_card_balance.csv")
cred_card_bal = cred_card_bal.drop(['SK_ID_PREV'], axis=1)

cred_card_bal_dfs = feature_aggregator_on_df(
    cred_card_bal, aggs_cat, aggs_num, ['SK_ID_CURR'], 'cred_card_balance', 'basic', save=False)
pos_cash_bal = pd.read_csv("../input/POS_CASH_balance.csv")
pos_cash_bal = pos_cash_bal.drop(['SK_ID_PREV'], axis=1)

pos_cash_bal_dfs = feature_aggregator_on_df(
    pos_cash_bal, aggs_cat, aggs_num, ['SK_ID_CURR'], 'pos_cash_balance', 'basic', save=False)
ins = pd.read_csv('../input/installments_payments.csv')
ins = ins.drop(['SK_ID_PREV'], axis=1)

ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

ins_dfs = feature_aggregator_on_df(
    ins, aggs_cat, aggs_num, ['SK_ID_CURR'], 'installments', 'basic', save=False)
