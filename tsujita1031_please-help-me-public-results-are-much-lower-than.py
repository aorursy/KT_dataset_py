import numpy as np

import pandas as pd

from pandas import DataFrame, Series

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier



import warnings

warnings.filterwarnings('ignore')
# load the data

path = "../input/titanic/"



df = pd.read_csv(path + 'train.csv')

df_test = pd.read_csv(path + 'test.csv')
# simple preprocessing



def preprocessing1(df_all):

    # Name -> extract title ('Mr', 'Mrs', and so on)

    df_all['title'] = df_all['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

    df_all['title'].replace(['Dr','Major','Sir','Col','Jonkheer','Don','Capt'], 'Mr', inplace=True)

    df_all['title'].replace(['Lady','Mme','the Countess','Dona'], 'Mrs', inplace=True)

    df_all['title'].replace(['Ms', 'Mlle'], 'Miss', inplace=True)

    

    df_all.drop('Name', axis=1, inplace=True)

    

    

    # deal with missing Age

    titles = df_all['title'].unique().tolist() 

    for title in titles:

        age_to_impute = df_all.groupby('title')['Age'].median()[titles.index(title)]

        df_all.loc[(df_all['Age'].isnull()) & (df_all['title'] == title), 'Age'] = age_to_impute

    

    # deal with missing Embarked

    df_all.fillna({'Embarked':'C'}, inplace=True)

    

    # drop Cabin

    df_all.drop('Cabin', axis=1, inplace=True)

    

    # drop Ticket

    df_all.drop('Ticket', axis=1, inplace=True)

    

    # deal with missing Fare

    df_all.fillna({'Fare': 7.8}, inplace=True)

    

    # drop Id

    df_all.drop('PassengerId', axis=1, inplace=True)

    

    return df_all

    
# single preprocessing such as standarlization and label encoding



from sklearn.preprocessing import StandardScaler, LabelEncoder



def preprocessing2(df_all, is_onehot=False):

    # Normarlize

    for column in ['Age', 'SibSp', 'Parch', 'Fare']:

        scaler = StandardScaler()

        df_all[[column]] = scaler.fit_transform(df_all[[column]].values)  #[[]]にしないと1次元配列でエラー

        



    # Label Encoding

    

    ## Sex

    df_all.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)

    

    # One Hot Encoding

    if is_onehot==True:

        df_all = pd.get_dummies(df_all, columns=['Pclass', 'Embarked', 'title'], drop_first=True)  

    # Label Encoding

    else:

        for column in ['Embarked', 'title']:

            le = LabelEncoder()

            le.fit(df_all[column])

            df_all[column] = le.transform(df_all[column])

        



    return df_all
# To preprocess train and test data at the same time

df_all = pd.concat([df, df_test], axis=0)
df_all = preprocessing1(df_all)

df_all = preprocessing2(df_all, is_onehot=True)
# divide train and test

df_cp = df_all[:891]

df_test_cp = df_all[891:1309]

df_cp
X = df_cp.drop('Survived', axis=1).values

y = df_cp['Survived'].values

X_test = df_test_cp.drop('Survived',axis=1).values
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.metrics import accuracy_score



# optuna tries to maximize return value of this function

def objective_rfc(trial):

    

    # tuning parameters

    min_samples_split = trial.suggest_int('min_samples_split', 3, 16)

    max_leaf_nodes = int(trial.suggest_int('max_leaf_nodes', 4, 64))

    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    max_depth = trial.suggest_int('max_depth', 1, 50)

    n_estimators = trial.suggest_int('n_estimators', 10, 40)



    # use Random Forest to predict

    rfc_1 = RandomForestClassifier(min_samples_split = min_samples_split,

                                   max_leaf_nodes = int(max_leaf_nodes),

                                   n_estimators = n_estimators,

                                   criterion = criterion,

                                   max_depth = max_depth,

                                   random_state=0)

    

    # cross validation (I think this is suspicious)

    kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

    preds = cross_val_score(rfc_1, X, y, cv=kf, scoring='accuracy').mean()

    

    return preds 
# Perform optimization of hyper parameters

import optuna



study = optuna.create_study(direction='maximize')

study.optimize(objective_rfc, n_trials=100)
print(study.best_params)

print(study.best_value)  # Wow! Amazing! This simple preprocessing and simple model scores 0.845!!
# I could get the best hyperparameters, so I used this model.



params=study.best_params

rfc_opt = RandomForestClassifier(**params)

rfc_opt.fit(X, y)
# Predict test data

rfc_prob = rfc_opt.predict_proba(X_test)

rfc_pred = rfc_prob.argmax(axis=1)
# make submission file

submission = pd.read_csv(path + "gender_submission.csv")

submission['Survived'] = rfc_pred
# make submission file

submission[:418].to_csv('submission.csv', index=False)