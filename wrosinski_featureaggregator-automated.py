import gc
import glob
import os
import time

import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold
gc.enable()
# Define whether all columns should be processed with sampled aggregates (False)
# or sample columns and sample aggregates for each of them using selected
# aggregations (True).
use_selected = True


# Set data source.
data_src = '../input/'


# 'mad' is very slow, so the second list excludes it.
aggs_all_num = ['mean', 'median', 'min', 'max', 'count', 'std', 'sem', 'sum', 'mad']

# Default lists of aggregates to sample from, _num for numerical variables.
# _cat for categorical variables.
aggs_medium_num = ['mean', 'median', 'min', 'max', 'count', 'std', 'sem', 'sum']
aggs_all_cat = ['mean', 'std', 'sum', 'median', 'count']
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
                                  categorical_columns=None,
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

        if categorical_columns is not None:
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

def load_data(data_src):
    
    start_time = time.time()
    
    train = pd.read_csv('{}application_train.csv'.format(data_src)) 
    test = pd.read_csv('{}application_test.csv'.format(data_src))
    print('Train and test tables loaded.')
    
    bureau = pd.read_csv('{}bureau.csv'.format(data_src))
    bureau_bal = pd.read_csv('{}bureau_balance.csv'.format(data_src))
    print('Bureau data loaded.')
    
    prev = pd.read_csv('{}previous_application.csv'.format(data_src))
    print('Previous applications data loaded.')
    
    cred_card_bal = pd.read_csv('{}credit_card_balance.csv'.format(data_src))
    print('Credit card balance loaded.')
    
    pos_cash_bal = pd.read_csv('{}POS_CASH_balance.csv'.format(data_src))
    print('POS cash balance loaded.')
    
    ins = pd.read_csv('{}installments_payments.csv'.format(data_src))
    print('Installments data loaded.')
    
    print('Time it took to load all the data: {:.4f}s\n'.format(time.time() - start_time))
    
    return train, test, bureau, bureau_bal, prev, cred_card_bal, pos_cash_bal, ins


def sample_aggregates(aggs_cat_all, aggs_num_all):
    
    """
    Sample aggregates for categorical and numerical variables.
    
    # Arguments:
        aggs_cat_all: (list), list of aggregates to sample from for categorical variables.
        aggs_num_all: (list), list of aggregates to sample from for numerical variables.

    # Returns:
        aggs_cat: (list), list of selected aggregates for categorical variables.
        aggs_num: (list), list of selected aggregates for categorical variables.
    """
    
    print('\nSample aggregates for numerical and categorical variables.')
    # Sample number of aggregates for numerical and categorical variables.
    num_sampled_cat_aggregates = np.random.randint(1, len(aggs_cat_all))
    num_sampled_num_aggregates = np.random.randint(1, len(aggs_num_all))
    
    # Sample aggregates for categorical variables from aggs_cat_all.
    # Their number is equal to num_sampled_cat_aggregates.
    aggs_cat = np.random.choice(aggs_cat_all, num_sampled_cat_aggregates, replace=False).tolist()
    print('Selected aggregates for categorical variables: {}\n'.format(aggs_cat))

    # Sample aggregates for numerical variables from aggs_num_all.
    # Their number is equal to num_sampled_num_aggregates.
    aggs_num = np.random.choice(aggs_num_all, num_sampled_num_aggregates, replace=False).tolist()
    print('Selected aggregates for numerical variables: {}'.format(aggs_num))
    
    return aggs_cat, aggs_num


def sample_selected_aggregations(df, aggs_list):
    
    """
    Sample combinations of columns/aggregates for selected aggregations.
    
    # Arguments:
        aggs_list: (list), list of aggregates to sample from for all variables.

    # Returns:
        selected_aggregates: (dict), dictionary, where each key is a columns having it's own
        list of aggregates.
    """
    
    
    # Sample number of columns to aggregate in df.
    # It is assumed that number of columns sampled will be higher than a half of all columns.
    sampled_num_columns = np.random.randint(np.ceil(len(df.columns) * 0.5), (len(df.columns)))
    # Sample columns from df, their number is equal to sampled_num_columns.
    sampled_columns = np.random.choice(df.columns, sampled_num_columns, replace=False)
    print('\nSample aggregates for {} columns.'.format(len(sampled_columns)))

    selected_aggregates = {}

    # For each chosen column, select number of aggregates for it with sampled_column_num_aggregates.
    # Then choose aggregates from aggs_list
    # Save each entry into a dictionary which will specify aggregations in DF, where column is the key
    # and value is a list of aggregates for this column.
    for i in sampled_columns:
        sampled_column_num_aggregates = np.random.randint(1, len(aggs_list))
        sampled_column_aggregates = np.random.choice(aggs_list, sampled_column_num_aggregates, replace=False).tolist()
        selected_aggregates[i] = sampled_column_aggregates
        
    print('Selected aggregations:\n{}\n'.format(selected_aggregates))
    
    return selected_aggregates


def categorical_features_factorize(X):

    categorical_feats = [col for col in X.columns if X[col].dtype == 'object']
    print('Categorical features encoding: {}'.format(categorical_feats))

    for col in categorical_feats:
        X[col] = pd.factorize(X[col])[0]

    print('Categorical features encoded.\n')

    return X


def run_kfold_lgbm(X_train,
                   y_train,
                   X_test,
                   model_params,
                   n_folds=5,
                   seed=1337):
    
    
    # Prepare KFold split, Stratified works well in this competition.
    # Parametrize it's seed to enable easy change of splits.
    kf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Subset features to eliminate irrelevant ones.
    X_train = X_train[good_features]
    X_test = X_test[good_features]
    
    # Assert that both train and test have the same set of columns.
    assert np.all(X_train.columns == X_test.columns), '\
    Train and test sets must have the same set of columns.'

    # Create oof sets for prediction storage.
    # Create gbm_history for storage of best AUC per fold.
    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_folds))
    
    gbm_history = {}
    folds_auc = []

    # Helper variable to index oof
    i = 0
    
    for train_index, valid_index in kf.split(X=X_train, y=y_train):
        assert len(np.intersect1d(train_index, valid_index)) == 0, '\
        Train and test indices must not overlap.'
        
        print('Running on fold: {}'.format(i + 1))

        # Create train and validation sets based on KFold indices.
        X_tr = X_train.iloc[train_index]
        X_val = X_train.iloc[valid_index]
        y_tr = y_train.iloc[train_index]
        y_val = y_train.iloc[valid_index]

        dtrain = lgb.Dataset(X_tr, y_tr)
        dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)

        # Train LightGBM model, it's parameters can be changed easily
        # through model_params function variable.
        gbm = lgb.train(
            params=model_params,
            train_set=dtrain,
            evals_result=gbm_history,
            num_boost_round=10000,  # change to 10000 for proper training.
            valid_sets=[dtrain, dvalid],
            early_stopping_rounds=200,
            verbose_eval=100)

        # Predict validation and test data and store them in oof sets.
        oof_train[valid_index] = gbm.predict(
            X_val, num_iteration=gbm.best_iteration)
        oof_test[:, i] = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        
        # Show best AUC per fold based on GBM training history.
        best_fold_auc = np.max(gbm_history['valid_1']['auc'])
        folds_auc.append(best_fold_auc)
        print('Best fold GBM AUC: {:.4f}\n'.format(best_fold_auc))
        
        i += 1
        
    print('Mean KFold AUC: {:.4f}'.format(np.asarray(folds_auc).mean()))

    return oof_train, oof_test
train, test, bureau, bureau_bal, prev, cred_card_bal, pos_cash_bal, ins = load_data(data_src)
ins = ins.drop(['SK_ID_PREV'], axis=1)
prev = prev.drop(['SK_ID_PREV'], axis=1)
cred_card_bal = cred_card_bal.drop(['SK_ID_PREV'], axis=1)
pos_cash_bal = pos_cash_bal.drop(['SK_ID_PREV'], axis=1)


ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']


# Concatenate train and test.
X = pd.concat([train, test], ignore_index=True, sort=False)

# Encode categorical features in concatenated DF.
X = categorical_features_factorize(X)
if use_selected:
    selected_aggregates_bureau_bal = sample_selected_aggregations(bureau_bal, aggs_all_cat)
    selected_aggregates_bureau = sample_selected_aggregations(bureau, aggs_all_cat)
    selected_aggregates_ins = sample_selected_aggregations(ins, aggs_all_cat)
    selected_aggregates_prev = sample_selected_aggregations(prev, aggs_all_cat)
    selected_aggregates_cred_card = sample_selected_aggregations(cred_card_bal, aggs_all_cat)
    selected_aggregates_pos_cash_bal = sample_selected_aggregations(pos_cash_bal, aggs_all_cat)
else:
    aggs_cat, aggs_num = sample_aggregates(aggs_medium_num, aggs_all_cat)
if use_selected:
    bureau_bal_dfs = feature_aggregator_on_df_selected(
        bureau_bal,
        selected_aggregates_bureau_bal,
        to_group=['SK_ID_BUREAU'],
        prefix='bureau_bal', suffix='basic_selected', save=False)

    bureau_ = bureau.merge(
        bureau_bal_dfs[0],
        how='left',
        on='SK_ID_BUREAU', copy=False)
    bureau_dfs = feature_aggregator_on_df_selected(
        bureau_,
        selected_aggregates_bureau,
        to_group=['SK_ID_CURR'],
        prefix='bureau', suffix='basic_selected', save=False)


    ins_dfs = feature_aggregator_on_df_selected(
        ins,
        selected_aggregates_ins,
        to_group=['SK_ID_CURR'],
        prefix='ins', suffix='basic_selected', save=False)


    prev_dfs = feature_aggregator_on_df_selected(
        prev,
        selected_aggregates_prev,
        to_group=['SK_ID_CURR'],
        prefix='prev', suffix='basic_selected', save=False)


    cred_card_bal_dfs = feature_aggregator_on_df_selected(
        cred_card_bal,
        selected_aggregates_cred_card,
        to_group=['SK_ID_CURR'],
        prefix='cred_card_bal', suffix='basic_selected', save=False)


    pos_cash_bal_dfs = feature_aggregator_on_df_selected(
        pos_cash_bal,
        selected_aggregates_pos_cash_bal,
        to_group=['SK_ID_CURR'],
        prefix='pos_cash_bal', suffix='basic_selected', save=False)


    X = X.merge(bureau_dfs[0], how='left', on='SK_ID_CURR', copy=False)
    X = X.merge(ins_dfs[0], how='left', on='SK_ID_CURR', copy=False)
    X = X.merge(prev_dfs[0], how='left', on='SK_ID_CURR', copy=False)
    X = X.merge(cred_card_bal_dfs[0], how='left', on='SK_ID_CURR', copy=False)
    X = X.merge(pos_cash_bal_dfs[0], how='left', on='SK_ID_CURR', copy=False)
    
else:
    
    bureau_bal_dfs = feature_aggregator_on_df(
        bureau_bal, aggs_cat, aggs_num, ['SK_ID_BUREAU'], 'bureau_bal', 'basic', save=False)

    bureau_ = bureau.merge(bureau_bal_dfs[0], how='left', on='SK_ID_BUREAU', copy=False)
    bureau_ = bureau_.merge(bureau_bal_dfs[1], how='left', on='SK_ID_BUREAU', copy=False)
    bureau_dfs = feature_aggregator_on_df(
        bureau_, aggs_cat, aggs_num, ['SK_ID_CURR'], 'bureau', 'basic', save=False)


    ins_dfs = feature_aggregator_on_df(
        ins, aggs_cat, aggs_num, ['SK_ID_CURR'], 'ins', 'basic', save=False)


    prev_dfs = feature_aggregator_on_df(
        prev, aggs_cat, aggs_num, ['SK_ID_CURR'], 'prev', 'basic', save=False)


    cred_card_bal_dfs = feature_aggregator_on_df(
        cred_card_bal, aggs_cat, aggs_num, ['SK_ID_CURR'], 'cred_card_bal', 'basic', save=False)


    pos_cash_bal_dfs = feature_aggregator_on_df(
        pos_cash_bal, aggs_cat, aggs_num, ['SK_ID_CURR'], 'pos_cash_bal', 'basic', save=False)


    X = X.merge(bureau_dfs[0], how='left', on='SK_ID_CURR', copy=False)
    X = X.merge(bureau_dfs[1], how='left', on='SK_ID_CURR', copy=False)
    X = X.merge(bureau_dfs[2], how='left', on='SK_ID_CURR', copy=False)

    X = X.merge(ins_dfs[0], how='left', on='SK_ID_CURR', copy=False)
    X = X.merge(ins_dfs[1], how='left', on='SK_ID_CURR', copy=False)

    X = X.merge(prev_dfs[0], how='left', on='SK_ID_CURR', copy=False)
    X = X.merge(prev_dfs[1], how='left', on='SK_ID_CURR', copy=False)
    X = X.merge(prev_dfs[2], how='left', on='SK_ID_CURR', copy=False)

    X = X.merge(cred_card_bal_dfs[0], how='left', on='SK_ID_CURR', copy=False)
    X = X.merge(cred_card_bal_dfs[1], how='left', on='SK_ID_CURR', copy=False)
    X = X.merge(cred_card_bal_dfs[2], how='left', on='SK_ID_CURR', copy=False)

    X = X.merge(pos_cash_bal_dfs[0], how='left', on='SK_ID_CURR', copy=False)
    X = X.merge(pos_cash_bal_dfs[1], how='left', on='SK_ID_CURR', copy=False)
    X = X.merge(pos_cash_bal_dfs[2], how='left', on='SK_ID_CURR', copy=False)
# Split data into train and test once again, based on availability of TARGET variable.
X_train = X[X['TARGET'].notnull()]
X_test = X[X['TARGET'].isnull()]

# Select TARGET and create a new variable for it, useful for model training.
y_train = X_train.TARGET

# Remove X (concatenated DF), as it will not be needed anymore.
del X
gc.collect()

# Select only features relevant to the model, do not use ID or index ones!
good_features = [x for x in X_train.columns if x not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
gbm_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'nthread': 6,
    'learning_rate': 0.05,  # 02,
    'num_leaves': 20,
    'colsample_bytree': 0.9497036,
    'subsample': 0.8715623,
    'subsample_freq': 1,
    'max_depth': 8,
    'reg_alpha': 0.041545473,
    'reg_lambda': 0.0735294,
    'min_split_gain': 0.0222415,
    'min_child_weight': 60, # 39.3259775,
    'seed': 0,
    'verbose': -1,
    'metric': 'auc',
}


oof_train, oof_test = run_kfold_lgbm(X_train, y_train, X_test, gbm_params)
# Take mean of fold predictions for the test data.
submission_preds = oof_test.mean(axis=1)

# Prepare submission format and save it.
submission_df = X_test[['SK_ID_CURR']].copy()
submission_df['TARGET'] = submission_preds
submission_df[['SK_ID_CURR', 'TARGET']].to_csv('submission.csv', index= False)