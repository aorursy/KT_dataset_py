# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib 

import seaborn as sns

from matplotlib import rcParams

from sklearn import preprocessing

from sklearn import model_selection

from sklearn import metrics

from sklearn import impute

from sklearn import ensemble

import lightgbm as lgb

import catboost as cat

import xgboost as xgb

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("/kaggle/input/janatahack-customer-segmentation/Train.csv")

test = pd.read_csv("/kaggle/input/janatahack-customer-segmentation/Test.csv")

submission = pd.read_csv("/kaggle/input/janatahack-customer-segmentation/sample_submission.csv")
train.head()
print(train.shape)

print(test.shape)
rcParams["figure.figsize"] = 15,10

train.isna().sum().plot(kind="bar")
rcParams["figure.figsize"] = 15,10

test.isna().sum().plot(kind="bar")
train_most_frequent = train.copy()

mean_imputer = impute.SimpleImputer(strategy='most_frequent')# strategy can also be mean or median 

train_most_frequent.iloc[:,:] = mean_imputer.fit_transform(train_most_frequent)
train_most_frequent.isna().sum()
train_most_frequent.head()
test_most_frequent = test.copy()

#setting strategy to 'mean' to impute by the mean

mean_imputer = impute.SimpleImputer(strategy='most_frequent')# strategy can also be mean or median 

test_most_frequent.iloc[:,:] = mean_imputer.fit_transform(test_most_frequent)
test_most_frequent.isna().sum()
train_most_frequent.describe()
train_most_frequent["kfold"] = -1

train_most_frequent = train.sample(frac=1).reset_index(drop=True)



y = train_most_frequent.Segmentation.values



kf =model_selection.StratifiedKFold(n_splits=5)



for f,(t_,v_) in enumerate(kf.split(X=train_most_frequent,y=y)):

    train_most_frequent.loc[v_,'kfold'] = f
train_most_frequent.head()
test_most_frequent["Ever_Married"].value_counts()
# train_most_frequent["odd_experience"] = (train_most_frequent["Graduated"]!="Yes") & (train_most_frequent["Profession"].isin(['Healthcare',  'Engineer', 'Doctor', 'Lawyer',

#        'Executive', 'Marketing'])).astype(int)

# test_most_frequent["odd_experience"] = ((test_most_frequent["Graduated"]!="Yes") & (test_most_frequent["Profession"].isin(['Healthcare',  'Engineer', 'Doctor', 'Lawyer',

#        'Executive', 'Marketing']))).astype(int)
lb1 = preprocessing.LabelEncoder()

lb = preprocessing.LabelEncoder()
train_most_frequent["Spending_Score"] = lb.fit_transform(train_most_frequent["Spending_Score"])

train_most_frequent["Gender"] = lb.fit_transform(train_most_frequent["Gender"])

train_most_frequent["Ever_Married"] = lb.fit_transform(train_most_frequent["Ever_Married"].astype(str))

train_most_frequent["Profession"] = lb.fit_transform(train_most_frequent["Profession"].astype(str))

train_most_frequent["Var_1"] = lb.fit_transform(train_most_frequent["Var_1"].astype(str))

train_most_frequent["Graduated"] = lb.fit_transform(train_most_frequent["Graduated"].astype(str))

# train_most_frequent["odd_experience"] = lb.fit_transform(train_most_frequent["odd_experience"].astype(str))

train_most_frequent["Segmentation"] = lb1.fit_transform(train_most_frequent["Segmentation"].astype(str))

test_most_frequent["Spending_Score"] = lb.fit_transform(test_most_frequent["Spending_Score"])

test_most_frequent["Gender"] = lb.fit_transform(test_most_frequent["Gender"])

test_most_frequent["Ever_Married"] = lb.fit_transform(test_most_frequent["Ever_Married"].astype(str))

test_most_frequent["Profession"] = lb.fit_transform(test_most_frequent["Profession"].astype(str))

test_most_frequent["Var_1"] = lb.fit_transform(test_most_frequent["Var_1"].astype(str))

# test_most_frequent["odd_experience"] = lb.fit_transform(test_most_frequent["odd_experience"].astype(str))

test_most_frequent["Graduated"] = lb.fit_transform(test_most_frequent["Graduated"].astype(str))
# train_most_frequent["exp_div_age"] = train_most_frequent["Work_Experience"].div( train_most_frequent["Age"])

# test_most_frequent["exp_div_age"] = test_most_frequent["Work_Experience"].div(test_most_frequent["Age"])
corr = train_most_frequent.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)
train_most_frequent.head()
train_most_frequent.ID.nunique()
train["Segmentation"].value_counts()
train_most_frequent["Spending_Score"].value_counts().plot(kind="bar")
train_most_frequent["Profession"].value_counts().plot(kind="bar")
train_most_frequent["Segmentation"].value_counts().plot(kind="bar")
sns.lineplot(x=train_most_frequent["Age"],y=train_most_frequent["Work_Experience"])
sns.lineplot(x=train_most_frequent["ID"],y=train_most_frequent["Work_Experience"])
sns.lineplot(x=train_most_frequent["ID"],y=train_most_frequent["Age"])
# def run(train,fold):

#     scores = []



#     df_train = train[train.kfold !=fold].reset_index(drop=True)

#     df_valid = train[train.kfold == fold].reset_index(drop=True)

#     df_train = df_train.drop(["kfold",],axis=1)

#     df_valid = df_valid.drop(["kfold",],axis=1)

    

#     x_train = df_train.drop(["Segmentation"],axis=1)

#     y_train = df_train["Segmentation"].values

    

#     x_valid = df_valid.drop(["Segmentation"],axis=1)

#     y_valid = df_valid["Segmentation"].values

    

# #     model = cat.CatBoostClassifier(cat_features=["Gender","Ever_Married","Graduated","Profession","Var_1"])

#     model = lgb.LGBMClassifier(n_estimators = 1000, min_samples_in_leaf = 10, learning_rate = .02, 

#                           feature_fraction = .8, max_depth = 8)

#     model.fit(x_train, y_train)

#     preds = model.predict(x_valid)

#     score = metrics.accuracy_score(y_valid, preds)

#     print(f"Fold = {fold}, AUC = {score}")
# for fold_ in range(0,5):

#     run(train=train_most_frequent,fold=fold_)
final_folds = []

def run(train,fold):

    folds = {}

    

    df_train = train[train.kfold !=fold].reset_index(drop=True)

    df_valid = train[train.kfold == fold].reset_index(drop=True)

    df_train = df_train.drop(["kfold"],axis=1)

    df_valid = df_valid.drop(["kfold"],axis=1)

    

    x_train = df_train.drop(["Segmentation"],axis=1)

    y_train = df_train["Segmentation"].values

    

    x_valid = df_valid.drop(["Segmentation"],axis=1)

    y_valid = df_valid["Segmentation"].values

    

    xgb_model = xgb.XGBClassifier(objective="multi:softmax",eval_metric="auc",learning_rate =0.1,

                                                         n_estimators=1000,

                                                         max_depth=5,

                                                         min_child_weight=1,

                                                         gamma=0,

                                                         subsample=0.8,

                                                         colsample_bytree=0.8)

#     lgb_model = lgb.LGBMClassifier(objective="multi:softmax",eval_metric="auc",max_depth=6,n_estimators=1000,reg_alpha=0.1,colsample_bytree=0.8,min_child_weight=1)

    

    xgb_model.fit(x_train, y_train)

    preds = xgb_model.predict(x_valid)

    score = metrics.accuracy_score(y_valid, preds)

    final_folds.append({"lgb_model_Fold":fold,

                                 "AUC":score})

    print(final_folds)
for fold_ in range(0,5):

    run(train=train_most_frequent,fold=fold_)
x = train_most_frequent.drop(["Segmentation","kfold"],axis=1)

y = train_most_frequent["Segmentation"]
xgb_model = xgb.XGBClassifier(objective="multi:softmax",n_jobs=-1,num_class=4,eval_metric="auc",learning_rate =0.1,

                                                         n_estimators=1000,

                                                         max_depth=5,

                                                         min_child_weight=1,

                                                         gamma=0,

                                                         subsample=0.8,

                                                         colsample_bytree=0.8,)

xgb_model.fit(x,y)
y_pred = xgb_model.predict(test_most_frequent)

print(y_pred)

new_y_pred = lb1.inverse_transform(y=y_pred)

print(new_y_pred)

submission["Segmentation"]=new_y_pred

submission.to_csv("av_XGB_New_Features1.csv",index=False)
lgb_model = lgb.LGBMClassifier(n_estimators = 1000, min_samples_in_leaf = 10, learning_rate = .02, feature_fraction = .8, max_depth = 8)

lgb_model.fit(x,y)
## Label Encode and preprocess

train_copy  = train.copy()

test_copy = test.copy()

train_copy['tr'] = 1

test_copy['tr'] = 0



appended = pd.concat([train_copy, test_copy], axis = 0)



cat_cols = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1']

label_enc = {}

for col in cat_cols:

    appended[col] = appended[col].astype(str)

    enc = preprocessing.LabelEncoder().fit(appended[col])

    appended[col] = enc.transform(appended[col])

    label_enc[col] = enc

    

    

#appended = pd.get_dummies(appended, columns = cat_cols)

train_copy = appended.loc[appended['tr'] == 1]

test_copy = appended.loc[appended['tr'] == 0]

Xcols = appended.drop(['Segmentation', 'tr'], axis = 1).columns

'''Xcols = ['Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession',

       'Work_Experience', 'Spending_Score', 'Family_Size', 'Var_1']'''

ycol = 'Segmentation'



X = train_copy[Xcols]

y = train_copy[ycol]



Xtest = test_copy[Xcols]
model = pipeline.make_pipeline(impute.KNNImputer(), xgb.XGBClassifier(objective="multi:softmax",n_jobs=-1,num_class=4,eval_metric="auc",learning_rate =0.1,

                                                         n_estimators=1000,

                                                         max_depth=5,

                                                         min_child_weight=1,

                                                         gamma=0,

                                                         subsample=0.8,

                                                         colsample_bytree=0.8,)).fit(X, y)
y_pred_pipe = model.predict(Xtest)
submission["Segmentation"]=y_pred_pipe

submission.to_csv("av_knn_imputer_Random.csv",index=False)