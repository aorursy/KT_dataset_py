#import necessary modules

import os
import gc
import pandas as pd
import numpy as np
import lightgbm as lgbm
import featuretools as ft
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
# Please note that you could have read it with simple read_csv, without using os (operating system commands...)
sample_size = 30000 
""" Load and process inputs """
input_dir = os.path.join(os.pardir, 'input')
print('Input files:\n{}'.format(os.listdir(input_dir)))
print('Loading data sets...')
app_train_df = pd.read_csv(os.path.join(input_dir,'application_train.csv'), nrows=sample_size)
app_test_df = pd.read_csv(os.path.join(input_dir,'application_test.csv'))
prev_app_df = pd.read_csv(os.path.join(input_dir,'previous_application.csv'), nrows=sample_size)
bureau_df = pd.read_csv(os.path.join(input_dir,'bureau.csv'), nrows=sample_size)
bureau_balance_df = pd.read_csv(os.path.join(input_dir,'bureau_balance.csv'), nrows=sample_size)
installments_df = pd.read_csv(os.path.join(input_dir,'installments_payments.csv'), nrows=sample_size)
cc_balance_df = pd.read_csv(os.path.join(input_dir,'credit_card_balance.csv'), nrows=sample_size)
pos_balance_df = pd.read_csv(os.path.join(input_dir,'POS_CASH_balance.csv'), nrows=sample_size)

print('Data loaded.\nMain application training data set shape = {}'.format(app_train_df.shape))
print('Main application test data set shape = {}'.format(app_test_df.shape))
print('Positive target proportion = {:.2f}'.format(app_train_df['TARGET'].mean()))
# Merge the datasets into a single one for training
app_both = pd.concat([app_train_df, app_test_df])
# A lot of the continuous days variables have integers as missing value indicators.
prev_app_df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
prev_app_df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
prev_app_df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
prev_app_df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
prev_app_df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
#Add new features
# Amount loaned relative to salary
app_both['LOAN_INCOME_RATIO'] = app_both['AMT_CREDIT'] / app_both['AMT_INCOME_TOTAL']
app_both['ANNUITY_INCOME_RATIO'] = app_both['AMT_ANNUITY'] / app_both['AMT_INCOME_TOTAL']
    
# Number of overall payments (I think!)
app_both['ANNUITY LENGTH'] = app_both['AMT_CREDIT'] / app_both['AMT_ANNUITY']
    
# Social features
app_both['WORKING_LIFE_RATIO'] = app_both['DAYS_EMPLOYED'] / app_both['DAYS_BIRTH']
app_both['INCOME_PER_FAM'] = app_both['AMT_INCOME_TOTAL'] / app_both['CNT_FAM_MEMBERS']
app_both['CHILDREN_RATIO'] = app_both['CNT_CHILDREN'] / app_both['CNT_FAM_MEMBERS']
# Create new entityset
es = ft.EntitySet(id='home_credit_default_risk')


# Create an entity from the applications (app_both) dataframe
# This dataframe already has an index
es = es.entity_from_dataframe(entity_id='applications',
                              
                              dataframe=app_both, index='SK_ID_CURR')


# Create an entity from the bureau dataframe
# This dataframe already has an index
es = es.entity_from_dataframe(entity_id='bureau', 
                            
                              dataframe=bureau_df, index='SK_ID_BUREAU')

# Create an entity from the bureau balance dataframe
es = es.entity_from_dataframe(entity_id='bureau_balance', 
                             
                              make_index = True,
                              dataframe=bureau_balance_df, index='bureau_balance_id')

# Create an entity from the installments dataframe
es = es.entity_from_dataframe(entity_id='installments',
                              make_index = True,
                              dataframe=installments_df, index='installment_id')



# Create an entity from the previous applications dataframe
es = es.entity_from_dataframe(entity_id='previous_application',
                             
                              make_index = True,
                              dataframe=prev_app_df, index='prev_app_id')

# Create an entity from the credit card balance dataframe
es = es.entity_from_dataframe(entity_id='cc_balance',
                         
                              make_index = True,
                              dataframe=cc_balance_df, index='cc_balance_id')

# Create an entity from the POS Cash balance dataframe
es = es.entity_from_dataframe(entity_id='pos_balance',

                              make_index = True,
                              dataframe=pos_balance_df, index='pos_balance_id')
# Relationship between applications and credits bureau
r_applications_bureau = ft.Relationship(es['applications']['SK_ID_CURR'],
                                    es['bureau']['SK_ID_CURR'])
es = es.add_relationship(r_applications_bureau)

# Relationship between applications and credits bureau
r_applications_installment = ft.Relationship(es['applications']['SK_ID_CURR'],
                                    es['installments']['SK_ID_CURR'])
es = es.add_relationship(r_applications_installment)

# Relationship between applications and credits bureau
r_bureau_bureaubalance = ft.Relationship(es['bureau']['SK_ID_BUREAU'],
                                    es['bureau_balance']['SK_ID_BUREAU'])
es = es.add_relationship(r_bureau_bureaubalance)

# Relationship between applications and previous applications
r_applications_prev_apps = ft.Relationship(es['applications']['SK_ID_CURR'],
                                    es['previous_application']['SK_ID_CURR'])
es = es.add_relationship(r_applications_prev_apps)

# Relationship between applications and credit card balance
r_applications_cc_balance = ft.Relationship(es['applications']['SK_ID_CURR'],
                                    es['cc_balance']['SK_ID_CURR'])
es = es.add_relationship(r_applications_cc_balance)

# Relationship between applications and POS cash balance
r_applications_pos_balance = ft.Relationship(es['applications']['SK_ID_CURR'],
                                    es['pos_balance']['SK_ID_CURR'])
es = es.add_relationship(r_applications_pos_balance)

print(es)
"""
Deep Feature Synthesis (DFS) is an automated method for performing feature engineering on relational and transactional data.
https://docs.featuretools.com/automated_feature_engineering/afe.html
"""
# Create new features using specified primitives
feature_matrix, feature_defs = ft.dfs(entityset = es, target_entity = 'applications',
                                      drop_contains=['SK_ID_PREV'], max_depth=2, verbose=True)
feature_matrix.head(5)
def process_dataframe(input_df, encoder_dict=None):
    """ Process a dataframe into a form useable by LightGBM """

    # Label encode categoricals
    print('Label encoding categorical features...')
    categorical_feats = input_df.columns[input_df.dtypes == 'object']
    for feat in categorical_feats:
        encoder = LabelEncoder()
        input_df[feat] = encoder.fit_transform(input_df[feat].fillna('NULL'))
    print('Label encoding complete.')

    return input_df, categorical_feats.tolist(), encoder_dict
feature_matrix_enc, categorical_feats, encoder_dict = process_dataframe(input_df=feature_matrix)
all_data_na = (feature_matrix_enc.isnull().sum() / len(feature_matrix_enc)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)
# Separate into train and test
train_df = feature_matrix_enc[feature_matrix_enc['TARGET'].notnull()].copy()

test_df = feature_matrix_enc[feature_matrix_enc['TARGET'].isnull()].copy()
test_df.drop(['TARGET'], axis=1, inplace=True)

del feature_matrix, feature_defs, feature_matrix_enc
gc.collect()
""" Train the model """
target = train_df.pop('TARGET')

lgbm_train = lgbm.Dataset(data=train_df,
                          label=target,
                          categorical_feature=categorical_feats,
                          free_raw_data=False)
lgbm_params = {
    'boosting': 'dart',
    'application': 'binary',
    'learning_rate': 0.1,
    'min_data_in_leaf': 30,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.5,
    'scale_pos_weight': 2,
    'drop_rate': 0.02
}

cv_results = lgbm.cv(train_set=lgbm_train,
                     params=lgbm_params,
                     nfold=5,
                     num_boost_round=600,
                     early_stopping_rounds=50,
                     verbose_eval=20,
                     metrics=['auc'])

optimum_boost_rounds = np.argmax(cv_results['auc-mean'])
print('Optimum boost rounds = {}'.format(optimum_boost_rounds))
print('Best CV result = {}'.format(np.max(cv_results['auc-mean'])))

clf = lgbm.train(train_set=lgbm_train,
                 params=lgbm_params,
                 num_boost_round=optimum_boost_rounds)

""" Predict on test set and create submission """
y_pred = clf.predict(test_df)
out_df = pd.DataFrame({'SK_ID_CURR': test_df.index, 'TARGET': y_pred})
out_df.to_csv('submission.csv', index=False)
