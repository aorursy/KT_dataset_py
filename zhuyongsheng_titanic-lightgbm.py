# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

from contextlib import contextmanager

import time

from sklearn.model_selection import KFold, StratifiedKFold

from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
@contextmanager

def timer(title):

    t0 = time.time()

    yield

    print("{} - done in {:.0f}s".format(title, time.time() - t0))
def one_hot_encoder(df, nan_as_category = True):

    original_columns = list(df.columns)

    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']

    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)

    new_columns = [c for c in df.columns if c not in original_columns]

    return df, new_columns
def application_train_test(nan_as_category=True):

    df=pd.read_csv('/kaggle/input/titanic/train.csv')

    df_test=pd.read_csv('/kaggle/input/titanic/test.csv')

    df=df.append(df_test)

    #feature engineer

    #1.Create Title feature from Name

    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    #2.Factorize Sex feature

    df['Sex'], uniques = pd.factorize(df['Sex'])

    #3.drop Cabin,Ticket,Name,PassagerID

    df=df.drop(['Cabin','Ticket','Name'],axis=1)

    #4.one hot encoder for object features

    df, cat_cols = one_hot_encoder(df, nan_as_category)

    #5.Age features

    guess_Ages=np.zeros((2,3))

    for i in range(0,2):

        for j in range(0,3):

            guess_df=df[(df['Sex']==i)&(df['Pclass']==j+1)]['Age'].dropna()

            Age_guess=guess_df.median()

            guess_Ages[i,j] = int( Age_guess/0.5 + 0.5 ) * 0.5

    for i in range(0,2):

        for j in range(0,3):

            df.loc[(df.Age.isnull())&(df['Sex']==i)&(df['Pclass']==j+1),'Age']=guess_Ages[i,j]

    df.loc[ df['Age'] <= 16, 'Age'] = 0

    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1

    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2

    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3

    df.loc[ df['Age'] > 64, 'Age']=4

    #6.create IsAlone feature,drop SibSp and Parch

    df['FamilySize']=df['SibSp']+df['Parch']+1

    df['IsAlone']=0

    df.loc[(df['FamilySize']==1),'IsAlone']=1

    df=df.drop(['SibSp','Parch'],axis=1)

    #7.Create AgePclass

    df['AgePclass']=df.Age*df.Pclass

    #8.bin fare

    df['Fare'].fillna(df['Fare'].dropna().median(), inplace=True)

    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0

    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1

    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2

    df.loc[ df['Fare'] > 31, 'Fare'] = 3

    df['Fare'] = df['Fare'].astype(int)



    gc.collect()

    return df
def kfold_lightgbm(df,num_folds,stratified = False):

    # Divide in training/validation and test data

    train_df = df[df['Survived'].notnull()]

    test_df = df[df['Survived'].isnull()]

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    del df

    gc.collect()

    # cross validation

    if stratified:

        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)

    else:

        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)

        

    oof_preds = np.zeros(train_df.shape[0])

    sub_preds = np.zeros(test_df.shape[0])

    feature_importance_df = pd.DataFrame()

    feats = [f for f in train_df.columns if f not in ['Survived','Cabin','Ticket','Name','PassengerId','SibSp','Parch']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['Survived'])):

        train_x, train_y = train_df[feats].iloc[train_idx], train_df['Survived'].iloc[train_idx]

        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['Survived'].iloc[valid_idx]



        # parameters

        clf = LGBMClassifier(

            nthread=4,

            n_estimators=1000,

            learning_rate=0.02,

            num_leaves=34,

            colsample_bytree=0.9497036,

            subsample=0.8715623,

            max_depth=8,

            reg_alpha=0.041545473,

            reg_lambda=0.0735294,

            min_split_gain=0.0222415,

            min_child_weight=39.3259775,

            random_state=100,

            silent=-1,

            verbose=-1, )

        #

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 

            eval_metric= 'accuracy', verbose= 100, early_stopping_rounds= 200)



        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits   

       



        #根据模型运行结果，储存特征重要性数据

        fold_importance_df = pd.DataFrame()

        fold_importance_df["feature"] = feats

        fold_importance_df["importance"] = clf.feature_importances_

        fold_importance_df["fold"] = n_fold + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d accuracy : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        del clf, train_x, train_y, valid_x, valid_y

        gc.collect()

    for i in range(len(sub_preds)):

        if sub_preds[i]<0.5:

            sub_preds[i]=0

        else:

            sub_preds[i]=1

    sub_preds=sub_preds.astype(int)

    print('Full accuracy score %.6f' % roc_auc_score(train_df['Survived'], oof_preds))

    # 保存测试集结果并画出特征重要性图标

    test_df['Survived'] = sub_preds

    test_df[['PassengerId', 'Survived']].to_csv(submission_file_name, index= False)

    display_importances(feature_importance_df)

    return feature_importance_df
def display_importances(feature_importance_df_):

    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))

    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))

    plt.title('LightGBM Features (avg over folds)')

    plt.tight_layout()

    plt.savefig('lgbm_importances01.png')
df = application_train_test()

submission_file_name = "submission_LightGBM.csv"

with timer("Run LightGBM with kfold"):

    feat_importance = kfold_lightgbm(df, num_folds= 5, stratified= False)