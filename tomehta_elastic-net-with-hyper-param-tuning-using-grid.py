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

from sklearn.preprocessing import StandardScaler

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
train_df.head()
# drop the columns which are not required for modelling like ID. 

target = train_df["target"]

train_df = train_df.drop(["target","id"],axis=1)

test_id = test_df["id"]

test_df = test_df.drop(["id"],axis=1)
#sc = StandardScaler()

#train_df = pd.DataFrame(sc.fit_transform(train_df))

#test_df = pd.DataFrame(sc.transform(test_df))



type(train_df)
#Reduce memoray usage

train_df=reduce_mem_usage(train_df)

test_df=reduce_mem_usage(test_df)
#lets define grid for elastic net

# split the training into 0.75 train and 0.25 test for cross validation

from sklearn.model_selection import StratifiedShuffleSplit,RepeatedStratifiedKFold

shuffle_split = StratifiedShuffleSplit(test_size=0.25,train_size=0.75,n_splits=35,random_state=87951)

#kfold = RepeatedStratifiedKFold(n_splits=25, n_repeats=10, random_state=87951)

param_grid = {

                'alpha'     : [0.1,1,10,0.01],

                'l1_ratio'  :  np.arange(0.40,1.00,0.10),

                'tol'       : [0.0001,0.001]

            }

eNet = ElasticNet(max_iter=10000)

grid_search = GridSearchCV(eNet, 

                           param_grid, 

                           scoring='roc_auc', 

                           cv = shuffle_split,

                           return_train_score=True,

                           n_jobs = -1)

grid_search.fit(train_df,target)
print("Best parameters : {}".format(grid_search.best_params_))

print("Best cross validation score: {:.2f}".format(grid_search.best_score_))

print("Best estimator: {}".format(grid_search.best_estimator_))
results = pd.DataFrame(grid_search.cv_results_)

results.sort_values(['mean_test_score'],ascending = False)[:10]

#results.loc["params", "mean_test_score", "std_test_score"]
scores_mean = np.array(results.mean_test_score).reshape(-1)

scores_std = np.array(results.std_test_score).reshape(-1)

print("mean CV scores for each fold {} ".format(scores_mean))

print("std CV scores for each fold {} ".format(scores_std))
clf = grid_search.best_estimator_

type(clf)
#plot the features and their coeff which model has found

el_df =pd.Series(clf.coef_,index=train_df.columns)

el_df = el_df[clf.coef_!=0]

plt.figure(figsize=(8,6))

el_df.plot(kind='barh')

plt.xlabel("Importance",fontsize=12)

plt.ylabel("Features",fontsize=12)

plt.title("Top Features",fontsize=16)

plt.show()
train_short=train_df.iloc[:,clf.coef_!=0]

test_short=test_df.iloc[:,clf.coef_!=0]
train_short.columns
# we can try bucketing the features,,may be another day

#cmb_df= pd.concat([train_short,test_short])

#now plan is to bin the continous features. before that again check if anything co-related.

#sns.heatmap(cmb_df.corr())

# bin the columns as per their percentile

#for col in cmb_df.columns:

    #bins = np.linspace(-5,5,11)

    #bins = np.percentile(cmb_df[col],range(0,101,10))

    #cmb_df[col+'_binned' ] = pd.cut(cmb_df[col], bins=bins)

    #cmb_df[col+'_binned' ] = pd.qcut(cmb_df[col],10, duplicates='drop')

#create dummies for binned cols

#cmb_df = pd.get_dummies(cmb_df,dummy_na= False)

#train_X = cmb_df.iloc[:250]

#test_X = cmb_df.iloc[250:]
pred_el = grid_search.predict(test_df)
pred_el.shape
#print test file 

sub_df = pd.DataFrame()

sub_df["id"] = test_id 

sub_df["target"] = pred_el

sub_df.to_csv("baseline_el.csv", index=False)
sub_df.head()