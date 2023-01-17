import pandas as pd

import numpy as np

import gc

import graphviz

import seaborn as sns

import matplotlib.pyplot as plt

import lightgbm as lgb

import xgboost as xgb

import sklearn

import sys

from datetime import datetime

print(sklearn.__version__,pd.__version__, np.__version__, lgb.__version__, xgb.__version__)



from sklearn import model_selection

from sklearn.model_selection import train_test_split



%matplotlib inline
dtypestrain = {}

dtypestrain['ID_code'] = 'category'

dtypestrain['target'] = 'int8'

for i in range(0,200):

    dtypestrain['var_' + str(i)] = 'float32'

    

dtypestest = {}

dtypestest['ID_code'] = 'category'

for i in range(0,200):

    dtypestest['var_' + str(i)] = 'float32'
%%time

train = pd.read_csv('../input/train.csv', dtype=dtypestrain)

test = pd.read_csv('../input/test.csv', dtype=dtypestest)
X_train, X_val, y_train, y_val = train_test_split(train.drop(['ID_code','target'],axis=1), train['target'], test_size=0.3, shuffle=True)
LGBGBDT_PARAM = {

    'random_state' : 1981,

    'n_estimators' : 2000,

    'learning_rate': 0.1,

    'num_leaves': 16,

    'max_depth': 4,

    'metric' : ['auc','binary_logloss'],

    'boosting_type' : 'gbdt',

    'objective' : 'binary',

    'reg_alpha' : 2.03,

    'reg_lambda' : 4.7,

    'feature_fraction' : 0.8, #colsample_bytree

    'feature_fraction_seed' : 1981, 

    'max_bins' : 100,

    'min_split_gain': 0.0148,

    'min_child_weight' : 7.835, #min_sum_hessian_in_leaf 

    'min_data_in_leaf' : 1000, #min_child_samples

    'random_state' : 1981, # Updated from 'seed'

    'subsample' : .912, #also known as Bagging fraction!

    'subsample_freq' : 200, # also known as bagging frequency!

    'boost_from_average' : False,

    'verbose_eval' : 50,

    'is_unbalance' : True,

    #'scale_pos_weight' : 10.1,

    }



LGBGBDT = lgb.LGBMClassifier( **LGBGBDT_PARAM,

                             n_jobs=4, #Kaggle 4 cores 4 threads

                            silent=-1,

                            )
%%time

LGBGBDT_FIT = LGBGBDT.fit(X_train, y_train,eval_set=[(X_val,y_val)], eval_metric= ['auc','binary_logloss'], early_stopping_rounds=100, verbose=50)
XGBOOST_PARAM = {

    'random_state' : 1981,

    'n_estimators' : 1000, #very slow with 2000!

    'learning_rate': 0.15,

    'num_leaves': 36,

    'max_depth': 6,

    'metric' : ['auc'],

    'boosting_type' : 'gbdt',

    #'drop_rate' : 0.2,    ##only for DART

    #'max_drop' : 100,    ##only for DART

    #'objective' : 'binary',

    'reg_alpha' : 2.03,

    'reg_lambda' : 4.7,

    'feature_fraction' : 0.8, #colsample_bytree

    'feature_fraction_seed' : 1981, 

    'max_bins' : 100,

    'min_split_gain': 0.0148,

    'min_child_weight' : 7.835, #min_sum_hessian_in_leaf 

    'min_data_in_leaf' : 1000, #min_child_samples

    'random_state' : 1981, # Updated from 'seed'

    'subsample' : .912, #also known as Bagging fraction!

    'subsample_freq' : 200, # also known as bagging frequency!

    'boost_from_average' : False,

    #'verbose_eval' : 5,

    'is_unbalance' : True,

    #'scale_pos_weight' : 10,

    }



XGBGBDT = xgb.XGBClassifier(**XGBOOST_PARAM,

                            tree_method = 'gpu_hist',

                            #n_jobs =4,

                            silent=0,

                            )
%%time

XGBGBDT_FIT = XGBGBDT.fit(X_train, y_train,

                      eval_set=[(X_train, y_train), (X_val, y_val)],

                        eval_metric='auc',

                          early_stopping_rounds=100,

                        verbose=25

                     )
#Produces a JSON model dump for LightGBM

model_lgb = LGBGBDT.booster_.dump_model()



#if you have a LGBM model saves as a JSON file you can use the following

#model_lgb = json.load(open('MY JSON LIGHTGBM MODEL.json', 'r'))
def get_splits_gain(tree_num=0, parent=-1, tree=None, lev=0, node_name=None, split_gain=None, reclimit=50000):

    '''

    Function to recusively walk thru a single decision tree (only LIGHTGBM for now) and extract GAIN values and Feature interactions. 

    Since it uses YIELD the user of the function needs to walk through the function in a for loop to extract values. 

    ---Arguments---

    tree_num : The number of the tree node to analyze used only in output.

    parent : DO NOT PASS A VALUE, it used by the function for recusion to keep track of the interactions.

    tree : A single decision tree as a DICT. Required.

    lev : DO NOT PASS A VALUE, it used by the function for recusion to keep track of the level of the node/interaction.

    node_name : DO NOT PASS A VALUE, it used by the function for recusion to keep track of the interactions.

    split_gain : DO NOT PASS A VALUE, it used by the function for recusion to keep track of the gain values.

    inter : DO NOT PASS A VALUE, it used by the function for recusion to keep track of the interactions.

    reclimit: this sets the max recusive limit higher incase the model is very deep. USe with caution, I have no idea on how the system beaves with very large values!

    

    ---YIELD---

    A single line per recursion:

    tree_num : tree number

    tag : 'split_feature', the tag/key for which the value is being extracted for the split.

    old_parent : The actual parent for the column that is splitted on, for the first node of the tree it is '-1' by default.

    parent : The child node under the old_parent. Note: for the first node the value is passed here

    lev : The depth/Level of the node, for the first node the level is 1.

    node_name : The node from where the info was extracted.

    split_gain : the gain value at that level

    '''

    sys.setrecursionlimit(reclimit)

    if tree == None:

        raise Exception('No tree present to analyze!')

    for k, v in tree.items():

        if type(v) != dict and k in ['split_feature']:

            old_parent = parent

            parent = v

            tag = k

            yield tree_num, tag, old_parent, parent, lev, node_name, split_gain

        elif isinstance(v, dict):

            if v.get('split_gain') == None:

                continue

            else:

                tree = v

                lev_inc = lev + 1

                node_name = k

                split_gain = v['split_gain']

                for result in get_splits_gain(tree_num, parent, tree, lev_inc, node_name, split_gain):

                    yield result

        else:

            continue

            

#Creates a feature dictionary based on the features present in the LGBM model

def lgbm_create_feat_dict(model):

    feat_dict = dict(enumerate(model['feature_names']))

    feat_dict[-1] = 'base'

    return feat_dict



def analyze_model(model):

    '''

    Take a JSON dump of LGBM model, calls the recursive function to analyse all trees in the model, interprets feature index/names and returns a dataframe with teh model analysis and a feature interactions

    ---Arguments---

    model :  LGBM JSON model

    ---Returns---

    tree_info_df : pandas DF with model summarized and feature interactions.

    '''

    tree_info = []

    for j in range(0,len(model['tree_info'])):

        for i in get_splits_gain(tree_num=j, tree=model['tree_info'][j]):

            tree_info.append(list(i))

    tree_info_df = pd.DataFrame(tree_info, columns=['TreeNo','Type','ParentFeature', 'SplitOnfeature','Level','TreePos','Gain'])

    lgbm_feat_dict = lgbm_create_feat_dict(model_lgb)

    tree_info_df['ParentFeature'].replace(lgbm_feat_dict, inplace=True)

    tree_info_df['SplitOnfeature'].replace(lgbm_feat_dict, inplace=True)

    tree_info_df['Interactions'] = tree_info_df['ParentFeature'].map(str) + ' - ' + tree_info_df['SplitOnfeature'].map(str)

    return tree_info_df
lgb_df = analyze_model(model_lgb)

lgb_df= round(lgb_df, 2)

lgb_df.head()
#Produce some calculations for easier plotting

lgb_inter_calc = lgb_df.groupby('Interactions')['Gain'].agg(['count','sum','min','max','mean','std']).sort_values(by='sum', ascending=False).reset_index('Interactions').fillna(0)

lgb_inter_calc = round(lgb_inter_calc, 2) #if i dont round sns.barplot fails due to too large a precision.

#Created 2 datasets as i see that BASE (the first node of the tree) has a very hight gains and thus dilutes the interactions

lgb_inter_calc_nobase = lgb_inter_calc[lgb_inter_calc['Interactions'].str.contains('base')==False]
lgb_inter_calc.head()
lgb_inter_calc_nobase.head()
gc.collect()
data = lgb_inter_calc_nobase.sort_values('sum', ascending=False).iloc[0:75].reset_index(drop=True)
def plot_feat_interaction(data=None):

    plt.figure(figsize=(20, 14))

    ax = plt.subplot(121)

    sns.barplot(x='sum', y='Interactions', data=data.sort_values('sum', ascending=False), ax=ax)

    ax.set_title('Total Gain for Feature Interaction', fontweight='bold', fontsize=14)

    # Plot Gain importances

    ax = plt.subplot(122)

    sns.barplot(x='count', y='Interactions', data=data.sort_values('sum', ascending=False), ax=ax)

    ax.set_title('No. of times Feature interacted', fontweight='bold', fontsize=14)

    plt.tight_layout()



plot_feat_interaction(data)
# Function to take an list and a dictionary and replacate the order of the list. Needed for syncronizing XGBOOST feature importance extraction with LIGHTGBM

def order_dict_bylist(order=None, unordered_dict=None):

    '''

    Function to order a dict by keys, based on list or predefined dict.

    ---Arguments---

    order: a list of values in a desired order

    unordered_dict: the dict to be sorted based on key values.

    

    ---Returns---

    newdict : the dict unordered_dict in desired order

    '''

    if order == None or (isinstance(order, list) or isinstance(order, dict)) == False:

        print('No ordered list or dict provided')

        return None

    #Create ordered dict to perform and easy sort

    elif isinstance(order, list):

        order = list(X_train.columns.values)

        i = 0

        orderdict = {}

        for k in order:

            orderdict[k] = i

            i += 1

        order = orderdict

    #Replace values in the dict

    newdict = {}

    for k, _ in order.items():

        v = unordered_dict[k]

        newdict[k] = v

    del order, orderdict

    return newdict
def plot_feature_imp_gain(features=list(X_train.columns), models=[LGBGBDT], feature_count=50, plot_all=True, return_df = False):



    '''

    Plotting function that plots the Split(weight) and Gain importances bar plots and summary with meaned values for all models.

    ---Arguments---

    features : list of features ( ordered)

    models: list of models

    feature_count :  number of features to plot for the individual models. will not affect the final plot

    plot_all : BOOL value to plot all the model plots,  if False, this is the only plot with mean values is printed.

    return_df : Bool to returns a pandas DF for further analysis.

    

    ---Returns---

    x : pandas df returned if return_df = True.

    '''

    

    x = pd.DataFrame()

    

    for model in models:

        scores_df = pd.DataFrame()

        scores_df['feature'] = features

        model_name = str(model).split('(')[0]

        if 'XGB' in str(model):

            name =  model_name + ' : ' + str(model.get_params(deep=False)['boosting_type'])

            scores_df['model'] = name

            xgbdictweight = model.get_booster().get_score(importance_type='weight')

            xgbdictgain = model.get_booster().get_score(importance_type='total_gain')           

            xgbdictweight  = order_dict_bylist(order=features, unordered_dict=xgbdictweight)

            xgbdictgain  = order_dict_bylist(order=features, unordered_dict=xgbdictgain)

            scores_df['split_score'] = list(xgbdictweight.values())

            scores_df['gain_score'] = list(xgbdictgain.values())           

            del xgbdictweight, xgbdictgain

        else:

            name = model_name + ' : ' + str(model.get_params(deep=False)['boosting_type'])

            scores_df['model'] = name

            scores_df['split_score'] = model.booster_.feature_importance(importance_type='split')

            scores_df['gain_score'] = model.booster_.feature_importance(importance_type='gain')



        x = pd.concat([scores_df,x])



        if plot_all == True:

            plt.figure(figsize=(20, 10))

            ax = plt.subplot(121)

            sns.barplot(x='split_score', y='feature', data=scores_df.sort_values('split_score', ascending=False).iloc[0:feature_count], ax=ax)

            ax.set_title('Feature scores wrt split importances - ' + name, fontweight='bold', fontsize=14)

            # Plot Gain importances

            ax = plt.subplot(122)

            sns.barplot(x='gain_score', y='feature', data=scores_df.sort_values('gain_score', ascending=False).iloc[0:feature_count], ax=ax)

            ax.set_title('Feature scores wrt gain importances - ' + name, fontweight='bold', fontsize=14)

            plt.tight_layout()

        else:

            continue

    if len(models) > 1:

        plt.figure(figsize=(20, 25))

        ax = plt.subplot(121)

        sns.barplot(x='split_score', y='feature', data=x.sort_values('split_score', ascending=False), ax=ax)

        ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)

        # Plot Gain importances

        ax = plt.subplot(122)

        sns.barplot(x='gain_score', y='feature', data=x.sort_values('gain_score', ascending=False), ax=ax)

        ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)

        plt.tight_layout()

    if return_df == True:

        return x.reset_index(drop=True)

scores_df = plot_feature_imp_gain(models=[XGBGBDT, LGBGBDT], plot_all=True, return_df=True)

#Just to see how the dataset looks

scores_df[scores_df['feature']=='var_81']