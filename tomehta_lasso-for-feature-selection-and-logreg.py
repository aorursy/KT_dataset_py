import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import ElasticNetCV, ElasticNet

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

from sklearn.model_selection import train_test_split

#sns.set_palette("bright")

%matplotlib inline
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
# this function reduces the memory print for dataset. it helps since we are using gridsearch

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
train_df=reduce_mem_usage(train_df)

test_df=reduce_mem_usage(test_df)
train_df.head()
# drop the columns which are not required for modelling like ID. 

target = train_df["target"]

train_df = train_df.drop(["target","id"],axis=1)

test_id = test_df["id"]

test_df = test_df.drop(["id"],axis=1)
#model

clf = LogisticRegression(class_weight = 'balanced', 

                         penalty='l1', 

                         C=0.1, 

                         solver='liblinear').fit(train_df, target)

#number of features 

np.sum(clf.coef_!=0)
def plot_feature_importance(model,df):

    df_res =pd.DataFrame({"Features" : df.columns,

                          "Importance" : model.coef_[0]})

    df_res.sort_values(by='Importance', ascending=False, inplace=True)

    df_res = df_res.iloc[:40]

    df_res.set_index("Features",drop=True)

    plt.figure(figsize=(15,12))

    sns.barplot(x= "Features", y = "Importance", data = df_res,orient="v")

    plt.ylabel("Importance",fontsize=12)

    plt.xlabel("Features",fontsize=12)

    plt.title("Top 40 Features of data set",fontsize=16)

    return df_res
plot_feature_importance(clf,train_df)
train_df.shape
train_short=train_df.iloc[:,clf.coef_[0]!=0]

test_short=test_df.iloc[:,clf.coef_[0]!=0]
#from rf top 15 features

['33',

 '65',

 '117',

 '217',

 '91',

 '295',

 '214',

 '268',

 '189',

 '199',

 '24',

 '56',

 '39',

 '237',

 '201']
train_short.shape
test_short.shape
cmb_df= pd.concat([train_short,test_short])
cmb_df.shape
cmb_df.head()
#now plan is to bin the continous features. before that again check if anything co-related.

sns.heatmap(cmb_df.corr())
cmb_df.describe()
# binn the columns as per their percentile

for col in cmb_df.columns:

    #bins = np.linspace(-5,5,11)

    #bins = np.percentile(cmb_df[col],range(0,101,10))

    #cmb_df[col+'_binned' ] = pd.cut(cmb_df[col], bins=bins)

    cmb_df[col+'_binned' ] = pd.qcut(cmb_df[col],10, duplicates='drop')
cmb_df.head()

cmb_df.info()
cmb_df = pd.get_dummies(cmb_df,dummy_na= False)
cmb_df.head(10)
#cmb_df =  cmb_df.select_dtypes(['category'])
train_X = cmb_df.iloc[:250]
test_X = cmb_df.iloc[250:]
# split the training into 0.75 train and 0.25 test for cross validation

from sklearn.model_selection import StratifiedShuffleSplit

shuffle_split = StratifiedShuffleSplit(test_size=0.25,train_size=0.75,n_splits=20)
param_grid = {'n_estimators'     : [1200],

              'max_features' :  [100,150],

              'max_depth'     : [5],

              'class_weight' : [{1:0.5, 0:0.5} ,{1:0.6, 0:0.4},{1:0.4, 0:0.6}]

             }

grid_search = GridSearchCV(

    estimator = RandomForestClassifier(random_state=12,n_jobs=-1),

    param_grid = param_grid, 

    cv = shuffle_split,

    return_train_score=True

   )

grid_search.fit(train_X,target)
'''param_grid = {'C'     : [1,0.01,0.1,10,100],

              'penalty' : ["l2"],

             

          }

grid_search = GridSearchCV(

    estimator = LogisticRegression(random_state=12,solver="liblinear"),

    param_grid = param_grid, 

    cv = shuffle_split,

    return_train_score=True

   )'''
#grid_search.fit(train_X,target)
print("Best parameters : {}".format(grid_search.best_params_))

print("Best cross validation score: {:.2f}".format(grid_search.best_score_))

print("Best estimator: {}".format(grid_search.best_estimator_))

results = pd.DataFrame(grid_search.cv_results_)
results.head()
scores_mean = np.array(results.mean_test_score).reshape(-1)

scores_std = np.array(results.std_test_score).reshape(-1)
print("mean CV scores for each fold {} ".format(scores_mean))

print("std CV scores for each fold {} ".format(scores_std))

pred_lr = grid_search.predict_proba(test_X)[:,1]
test_X.shape
pred_lr.shape
#print test file 

sub_df = pd.DataFrame()

sub_df["id"] = test_id 

sub_df["target"] = pred_lr

sub_df.to_csv("baseline_lr.csv", index=False)
sub_df.head()