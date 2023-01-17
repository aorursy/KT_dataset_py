# pandas, numpy, matplotlib, seaborn

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import pipeline, metrics



# Machine Learning Packages

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV



from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

import xgboost as xgb

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.svm import SVC

from sklearn.cross_validation import KFold;



# Metrics testing

from sklearn.metrics import roc_auc_score
# Import the dataset

train = pd.read_csv("../input/train.csv", na_values=-1)

test = pd.read_csv("../input/test.csv", na_values=-1)

sample_submission = pd.read_csv("../input/sample_submission.csv")
test_id = test[['id']]



number_variables = train.shape[1]

number_observations = train.shape[0]

print("The train dataset has {} observtions and {} variables".format(number_observations, number_variables))



number_variables_test = test.shape[1]

number_observations_test = test.shape[0]

print("The test dataset has {} observtions and {} variables".format(number_observations_test, number_variables_test))
# To ckeck if we have missing values, we can use :

print(train.info())

print("----------")

print(test.info())
missing_values_df = pd.DataFrame()

missing_values_df['train'] = train.copy().drop(['target'], axis=1).isnull().sum()

missing_values_df['test'] = test.isnull().sum()

missing_values_df
train_copy = train.copy().drop(['target'], axis = 1)

test_copy = test.copy()

all_df = pd.concat([train, test])

# We use the package missingno to visualize missing values

# It's easier to see missing values thanks t this package instead of seeing numbers

import missingno as msno

# Nullity or missing values by columns

msno.matrix(df=all_df.iloc[:,1:20], figsize=(20, 14), color=(0.42, 0.1, 0.05))
msno.matrix(df=all_df.iloc[:,21:58], figsize=(20, 14), color=(0.42, 0.1, 0.05))
# We try our first model by removing variables with missing values

list_miss = ['ps_ind_02_cat','ps_ind_04_cat','ps_ind_05_cat','ps_reg_03','ps_car_01_cat','ps_car_02_cat',

                'ps_car_03_cat','ps_car_05_cat','ps_car_07_cat','ps_car_09_cat','ps_car_11','ps_car_14', 'ps_car_12']

train_miss = train.copy()

train_miss = train_miss.drop(list_miss, axis = 1)

test = test.drop(list_miss, axis = 1)
print("The propotion of people who didn't buy is {:.2f}%".format(train_miss.loc[train['target'] == 0].shape[0]/train.shape[0]))

print("The propotion of people who did buy is {:.2f}%".format(train_miss.loc[train['target'] == 1].shape[0]/train.shape[0]))



sns.countplot(train_miss['target'])

plt.title('Proportion of 0 and 1 for the target variable')

# The dataset is quite unbalanced
# Let's work on the binary values, cf https://www.kaggle.com/arthurtok/interactive-porto-insights-a-plot-ly-tutorial
import plotly.graph_objs as go

import plotly.tools as tls

import plotly.offline as py

py.init_notebook_mode(connected=True)



bin_col = [col for col in train.columns if '_bin' in col]

zero_list = []

one_list = []

for col in bin_col:

    zero_list.append((train[col]==0).sum())

    one_list.append((train[col]==1).sum())

    

trace1 = go.Bar(

    x=bin_col,

    y=zero_list ,

    name='Zero count'

)

trace2 = go.Bar(

    x=bin_col,

    y=one_list,

    name='One count'

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='stack',

    title='Count of 1 and 0 in binary variables'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')
# Things to try, OHE on categorial variable and same thing than above

# We can remove the variables ['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin']
# From this diagram, we can remove some variables because of the huge proportion of 0.

# We can remove : 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 
train_miss = train_miss.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'], axis = 1)

test = test.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'], axis = 1)
colormap = plt.cm.viridis

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train_miss.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, linecolor='white')

plt.show()
corr_matrix = train_miss.astype(float).corr()



for i in range(1,corr_matrix.shape[1]):

    for j in range(1,corr_matrix.shape[1]):

        if (i>j and corr_matrix.iat[i,j] > 0.5):

            print("Corrélation entre {} et {} : {}".format(corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iat[i,j])) 
# Dataset fir for modelization

df_model = train_miss.copy()

df_model.head(3)
# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897

def gini(actual, pred, cmpcol = 0, sortcol = 1):

    assert( len(actual) == len(pred) )

    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

    totalLosses = all[:,0].sum()

    giniSum = all[:,0].cumsum().sum() / totalLosses

    

    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)

 

def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return 'gini', gini_score
X = train_miss.drop(['target', 'id'], axis = 1)

y = train_miss['target']

test = test.drop(['id'], axis = 1)
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits = 5, random_state = 42)

cv_gini_train = []

cv_gini_val = []

cv_AUC_train = []

cv_AUC_val = []

y_test_logreg_pred = 0

i = 1



for train_index,test_index in skf.split(X, y):

    print("Step {} of skfold {}".format(i,skf.n_splits))

    X_train, X_val = X.loc[train_index],X.loc[test_index]

    y_train, y_val = y.loc[train_index],y.loc[test_index]

    lr = LogisticRegression().fit(X_train, y_train)

    y_train_logreg_pred = lr.predict_proba(X_train)[:,1]

    y_val_logreg_pred = lr.predict_proba(X_val)[:,1]

    y_test_logreg_pred += lr.predict_proba(test)[:,1]  #Sum all predictions for each folds

    cv_gini_train.append(gini_normalized(y_train, y_train_logreg_pred))

    cv_gini_val.append(gini_normalized(y_val, y_val_logreg_pred))

    

    cv_AUC_train.append(roc_auc_score(y_train, y_train_logreg_pred))

    cv_AUC_val.append(roc_auc_score(y_val, y_val_logreg_pred))

    i+=1

print("Train Average Gini : {}".format(np.mean(cv_gini_train)))

print("Validation Average Gini : {}".format(np.mean(cv_gini_val)))

print("Train Average AUC : {}".format(np.mean(cv_AUC_train)))

print("Validation Average AUC : {}".format(np.mean(cv_AUC_val)))
y_prediction = y_test_logreg_pred / 5
df_send = pd.DataFrame()

df_send['target'] = y_prediction

df_send['id'] = test_id

df_send.to_csv("submission_file_kaggle.csv", index=True, float_format="%.9f")
for n_estimator_i in [1000]:

    for max_depth_i in [2,3,4,5,6]:

        random_forest = RandomForestClassifier(n_estimators=n_estimator_i, max_depth = max_depth_i)

        random_forest.fit(X_train, y_train.values.ravel())

        y_train_rf_pred = random_forest.predict_proba(X_train) 

        y_test_rf_pred = random_forest.predict_proba(X_test) 

        print("N estim : {}, depth : {}".format(n_estimator_i, max_depth_i))

        print("Training set score RF Gini: {:.3f}".format(gini_normalized(y_train, y_train_rf_pred[:,1])))

        print("Test set score RF Gini: {:.3f}".format(gini_normalized(y_test, y_test_rf_pred[:,1])))
skf = StratifiedKFold(n_splits = 5, random_state = 42)

cv_gini_train = []

cv_gini_val = []

cv_AUC_train = []

cv_AUC_val = []

y_test_logreg_pred = 0

i = 1



for n_estimator_i in [1000]:

    for max_depth_i in [2,3,4,5,6]:

        cv_gini_train = []

        cv_gini_val = []

        cv_AUC_train = []

        cv_AUC_val = []

        y_test_logreg_pred = 0

        i = 1

        for train_index,test_index in skf.split(X, y):

            print("Step {} of skfold {}".format(i,skf.n_splits))

            X_train, X_val = X.loc[train_index],X.loc[test_index]

            y_train, y_val = y.loc[train_index],y.loc[test_index]

            lr = Ridge(alpha = alpha_i).fit(X_train, y_train)

            y_train_logreg_pred = lr.predict(X_train)

            y_val_logreg_pred = lr.predict(X_val)

            y_test_logreg_pred += lr.predict(test) #Sum all predictions for each folds

            cv_gini_train.append(gini_normalized(y_train, y_train_logreg_pred))

            cv_gini_val.append(gini_normalized(y_val, y_val_logreg_pred))



            cv_AUC_train.append(roc_auc_score(y_train, y_train_logreg_pred))

            cv_AUC_val.append(roc_auc_score(y_val, y_val_logreg_pred))

            i+=1

        print("N_estimator : {}, Max_depth : {}".format(n_estimator_i, n_depth_i))

        print("Train Average Gini : {}".format(np.mean(cv_gini_train)))

        print("Validation Average Gini : {}".format(np.mean(cv_gini_val)))

        print("Train Average AUC : {}".format(np.mean(cv_AUC_train)))

        print("Validation Average AUC : {}".format(np.mean(cv_AUC_val)))

y_train_rf_pred[:,1]
# Create Dmatrix

dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test, label=y_test)



# Setting parameters

param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}

param['nthread'] = 4

param['eval_metric'] = 'auc'



evallist = [(dtest, 'eval'), (dtrain, 'train')]



# Training the model

progress = dict()

num_round = 10

bst = xgb.train(param, dtrain, num_round, evallist, feval=gini_xgb, evals_result=progress)
x1 = progress['eval']['gini']

x2 = progress['train']['gini']



#To plot multiple graphs on the same figure you will have to do:



plt.plot(x1, 'r') # plotting t, a separately 

plt.plot(x2, 'b') # plotting t, b separately 

plt.show()
list_miss = ['ps_ind_02_cat','ps_ind_04_cat','ps_ind_05_cat','ps_reg_03','ps_car_01_cat','ps_car_02_cat',

                'ps_car_03_cat','ps_car_05_cat','ps_car_07_cat','ps_car_09_cat','ps_car_11','ps_car_12','ps_car_14']

test_sample = test.copy()

test_sample = test_sample.drop(list_miss, axis = 1)
dtest = xgb.DMatrix(test_sample)

test_sample_pred = bst.predict(dtest)
y_train_pred_xgb = bst.predict(dtrain)

y_test_pred_xgb = bst.predict(dtest)



print("Training set score XGB Gini: {:.3f}".format(gini_normalized(y_train, y_train_pred_xgb)))

print("Test set score XGB Gini: {:.3f}".format(gini_normalized(y_test, y_test_pred_xgb)))
params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 

          'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}



X = train.drop(['id', 'target'], axis=1)

features = X.columns

X = X.values

y = train['target'].values

sub=test['id'].to_frame()

sub['target']=0



nrounds=200  # need to change to 2000

kfold = 2  # need to change to 5

skf = StratifiedKFold(n_splits=kfold, random_state=0)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    print(' xgb kfold: {}  of  {} : '.format(i+1, kfold))

    X_train, X_valid = X[train_index], X[test_index]

    y_train, y_valid = y[train_index], y[test_index]

    d_train = xgb.DMatrix(X_train, y_train) 

    d_valid = xgb.DMatrix(X_valid, y_valid) 

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    xgb_model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=100, 

                          feval=gini_xgb, maximize=True, verbose_eval=100)

    sub['target'] += xgb_model.predict(xgb.DMatrix(test[features].values), 

                        ntree_limit=xgb_model.best_ntree_limit+50) / (2*kfold)

gc.collect()

sub.head(2)
ps_ind_03

ps_car_13

ps_reg_02

ps_ind_17_bin

ps_car_15

ps_ind_15

ps_reg_01

ps_car_04_cat