import itertools
import re
import random
import csv
import ast
import json
import pickle
import pprint

from timeit import default_timer as timer
from datetime import datetime

# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Model
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score,roc_auc_score
from mlens.ensemble import SuperLearner

# Hyperopt
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from hyperopt.pyll.stochastic import sample

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

from IPython.display import display

pd.options.display.max_rows = 100

%matplotlib inline
# Set a few plotting defaults
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14
plt.rcParams['patch.edgecolor'] = 'k'
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
train_raw = pd.read_csv('../input/train.csv')
test_raw = pd.read_csv('../input/test.csv')

train_raw['train'] = 1
test_raw['train'] = 0
data_all = pd.concat([train_raw, test_raw], axis=0).reset_index(drop=True)

data_all.Embarked = data_all.Embarked.fillna(data_all.Embarked.mode()[0])

le = LabelEncoder()

data_all.Sex = le.fit_transform(data_all[['Sex']])
data_all.Embarked = le.fit_transform(data_all[['Embarked']])

data_all['family_size'] = data_all.SibSp + data_all.Parch + 1

def calc_family_size_bin(family_size):
    if family_size == 1:
        return 0
    elif family_size <= 4: 
        return 1
    else:
        return 2
        
data_all['family_size_bin'] = data_all.family_size.map(calc_family_size_bin)
data_all['name_title'] = data_all.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

name_title_dict = {
    'Capt': 'Mr',
    'Col': 'Mr',
    'Don': 'Mr',
    'Dona': 'Mrs',    
    'Dr': 'Dr',
    'Jonkheer': 'Mr',
    'Lady': 'Mrs',
    'Major': 'Mr',
    'Master': 'Master',
    'Miss': 'Miss',
    'Mlle': 'Miss',
    'Mme': 'Miss',
    'Mr': 'Mr',
    'Mrs': 'Mrs',
    'Ms': 'Mrs',
    'Rev': 'Mr',
    'Sir': 'Mr',
    'Countess': 'Mrs'
}

data_all['name_title_cat'] = data_all.name_title.map(name_title_dict)

data_all['last_name'] = data_all.Name.str.extract('([A-Za-z]+),', expand=False)
data_all['last_name_family_size'] = data_all.apply(lambda row: row.last_name + '_' + str(row.family_size), axis=1)
data_all['last_name_ticket'] = data_all.apply(lambda row: row.last_name + '_' + row.Ticket, axis=1)

last_name_family_size_check = data_all[data_all.family_size > 1].groupby('last_name_family_size').agg({'Survived': lambda x: x.isnull().sum()}).reset_index()
last_name_family_size_check.columns = ['last_name_family_size','last_name_family_size_feature']

data_all = pd.merge(data_all, last_name_family_size_check, on='last_name_family_size', how='left')
data_all = data_all.sort_values('PassengerId').reset_index(drop=True)
data_all.last_name_family_size_feature = data_all.last_name_family_size_feature.fillna(0)

data_all.loc[data_all.last_name_family_size_feature == 0, 'last_name_family_size'] = 'X'

family_count = data_all.groupby(['last_name','Ticket']).PassengerId.count().reset_index()
family_count.columns = ['last_name','Ticket','family_count']

family_survival = data_all.groupby(['last_name','Ticket']).Survived.sum().reset_index()
family_survival.columns = ['last_name','Ticket','family_survival_sum']

family_survival = pd.merge(family_count, family_survival, on=['last_name','Ticket'])

def cal_family_survival(row):
    family_survival = 0.5
    if row['family_count'] > 1 and row['family_survival_sum'] > 0:
        family_survival = 1
    elif row['family_count'] > 1 and row['family_survival_sum'] == 0:
        family_survival = 0
        
    return family_survival

family_survival['family_survival'] = family_survival.apply(cal_family_survival, axis=1)

data_all = pd.merge(data_all, family_survival, on=['last_name','Ticket'], how='left')
data_all = data_all.sort_values('PassengerId').reset_index(drop=True)

ticket_df = data_all.groupby('Ticket', as_index=False)['PassengerId'].count()
ticket_df.columns = ['Ticket','ticket_count']

data_all = pd.merge(data_all, ticket_df, on=['Ticket'])
data_all = data_all.sort_values('PassengerId').reset_index(drop=True)

last_name_ticket_check = data_all[data_all.ticket_count > 1].groupby('last_name_ticket').agg({'Survived': lambda x: x.isnull().sum()}).reset_index()
last_name_ticket_check.columns = ['last_name_ticket','last_name_ticket_feature']

data_all = pd.merge(data_all, last_name_ticket_check, on='last_name_ticket', how='left')
data_all = data_all.sort_values('PassengerId').reset_index(drop=True)
data_all.last_name_ticket_feature = data_all.last_name_ticket_feature.fillna(0)

data_all.loc[data_all.last_name_ticket_feature == 0, 'last_name_ticket'] = 'X'

def calc_family_count_bin(family_count):
    if family_count == 1:
        return 0
    elif family_count <= 2: 
        return 1
    elif family_count <= 4:
        return 2
    else:
        return 3
        
data_all['family_count_bin'] = data_all.family_count.map(calc_family_count_bin)

data_all.name_title_cat = le.fit_transform(data_all[['name_title_cat']])
data_all.last_name_family_size = le.fit_transform(data_all[['last_name_family_size']])
data_all.last_name_ticket = le.fit_transform(data_all[['last_name_ticket']])

data_all['fare_fixed'] = data_all.Fare/data_all.ticket_count
fare_median = data_all.fare_fixed.median()
data_all['fare_fixed'] = data_all.fare_fixed.fillna(fare_median)
data_all['fare_fixed_log'] = np.log1p(data_all.fare_fixed)

age_median_by_sex_title = data_all.groupby(['Sex', 'name_title'], as_index=False).Age.median()

data_all = pd.merge(data_all, age_median_by_sex_title, on=['Sex', 'name_title'])
data_all['Age'] = data_all.apply(lambda row: row.Age_x if not np.isnan(row.Age_x) else row.Age_y, axis=1)
data_all = data_all.drop(['Age_x','Age_y'], axis=1).sort_values('PassengerId').reset_index(drop=True)

def calc_age_bin(age):
    if age <= 15:
        return 0
    elif age <= 30:
        return 1
    elif age <= 60:
        return 2
    else:
        return 3
        
data_all['age_bin'] = data_all.Age.map(calc_age_bin)

def parse_ticket_str(ticket):
    arr = ticket.split()
    if not arr[0].isdigit():
        txt = arr[0].replace('.', '')
        txt = txt.split('/')[0]
        return re.findall('[a-zA-Z]+', txt)[0]
    else:
        return None
        
data_all['ticket_str'] = data_all.Ticket.map(parse_ticket_str)

def parse_ticket_number(ticket):
    arr = ticket.split()
    if len(arr) == 1 and arr[0].isdigit():
        return int(arr[0])
    elif len(arr) == 2 and arr[1].isdigit():
        return int(arr[1])
    else:
        if arr[-1].isdigit():
            return int(arr[-1])
        else:
            return np.nan
    
data_all['ticket_number'] = data_all.Ticket.map(parse_ticket_number)

def parse_ticket_num_len(ticket):
    arr = ticket.split()
    if len(arr) == 1 and arr[0].isdigit():
        return len(arr[0])
    elif len(arr) == 2 and arr[1].isdigit():
        return len(arr[1])
    else:
        if arr[-1].isdigit():
            return len(arr[-1])
        else:
            return -1
    
data_all['ticket_num_len'] = data_all.Ticket.map(parse_ticket_num_len)

data_all['ticket_num_len_4_prefix'] = data_all[data_all.ticket_num_len == 4].ticket_number.map(lambda x: int(str(x)[0]))
data_all['ticket_num_len_4_prefix_2'] = data_all[data_all.ticket_num_len == 4].ticket_number.map(lambda x: int(str(x)[:2]))

data_all['ticket_num_len_5_prefix'] = data_all[data_all.ticket_num_len == 5].ticket_number.map(lambda x: int(str(x)[0]))
data_all['ticket_num_len_5_prefix_2'] = data_all[data_all.ticket_num_len == 5].ticket_number.map(lambda x: int(str(x)[:2]))

data_all['ticket_num_len_6_prefix'] = data_all[data_all.ticket_num_len == 6].ticket_number.map(lambda x: int(str(x)[0]))
data_all['ticket_num_len_6_prefix_2'] = data_all[data_all.ticket_num_len == 6].ticket_number.map(lambda x: int(str(x)[:2]))
data_all['ticket_num_len_6_prefix_3'] = data_all[data_all.ticket_num_len == 6].ticket_number.map(lambda x: int(str(x)[:3]))

data_all.ticket_str = data_all.ticket_str.fillna('X')

data_all.ticket_num_len_4_prefix = data_all.ticket_num_len_4_prefix.fillna(-1)
data_all.ticket_num_len_4_prefix_2 = data_all.ticket_num_len_4_prefix_2.fillna(-1)
data_all.ticket_num_len_5_prefix = data_all.ticket_num_len_5_prefix.fillna(-1)
data_all.ticket_num_len_5_prefix_2 = data_all.ticket_num_len_5_prefix_2.fillna(-1)
data_all.ticket_num_len_6_prefix = data_all.ticket_num_len_6_prefix.fillna(-1)
data_all.ticket_num_len_6_prefix_2 = data_all.ticket_num_len_6_prefix_2.fillna(-1)
data_all.ticket_num_len_6_prefix_3 = data_all.ticket_num_len_6_prefix_3.fillna(-1)

data_all.ticket_num_len_4_prefix = data_all.ticket_num_len_4_prefix.astype(int)
data_all.ticket_num_len_4_prefix_2 = data_all.ticket_num_len_4_prefix_2.astype(int)
data_all.ticket_num_len_5_prefix = data_all.ticket_num_len_5_prefix.astype(int)
data_all.ticket_num_len_5_prefix_2 = data_all.ticket_num_len_5_prefix_2.astype(int)
data_all.ticket_num_len_6_prefix = data_all.ticket_num_len_6_prefix.astype(int)
data_all.ticket_num_len_6_prefix_2 = data_all.ticket_num_len_6_prefix_2.astype(int)
data_all.ticket_num_len_6_prefix_3 = data_all.ticket_num_len_6_prefix_3.astype(int)

data_all.ticket_str = le.fit_transform(data_all[['ticket_str']])

data_all['cabin_cat'] = data_all.Cabin.fillna('X').str[0]

def calc_cabin_len(cabin):
    if type(cabin) == float:
        return 0
    else:
        return len(cabin.split())

data_all['cabin_len'] = data_all.Cabin.map(calc_cabin_len)

data_all.cabin_cat = le.fit_transform(data_all[['cabin_cat']])
features = [
    'Pclass', 'Sex',
    'family_size_bin',
    'name_title_cat', 
    'last_name_family_size', 'last_name_ticket',
    'fare_fixed_log', 'age_bin',
    'ticket_str',
    'ticket_num_len_4_prefix_2',
    'ticket_num_len_5_prefix_2',
    'ticket_num_len_6_prefix_3',
    'cabin_cat', 'cabin_len',
]

X_train = data_all[data_all.train == 1][features]
X_test = data_all[data_all.train == 0][features]
y_train = data_all[data_all.train == 1].Survived.astype(int)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train.shape, X_test.shape
def cv_model(train, train_labels, model, name, cv=10, scoring='accuracy'):
    """Perform k fold cross validation of a model"""
    
    cv_scores = cross_val_score(model, train, train_labels, cv=cv, scoring=scoring, n_jobs=-1)
    print(f'{cv} Fold CV {scoring} for {name}: {round(cv_scores.mean(), 5)} with std: {round(cv_scores.std(), 5)}')
    
def make_prediction(train, target, test, model, model_name):
    #cv_model(train, target, model, model_name)

    model.fit(train, target)
    pred = model.predict(test).astype(int)

    output = f'{model_name}_submission_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
    submit_df = pd.DataFrame()
    submit_df['PassengerId'] = test_raw.PassengerId
    submit_df['Survived'] = pred
    
    submit_df[['PassengerId','Survived']].to_csv(output, index=False)
    print(f'submission file {output} is generated.')
# cv val mean high
xgb_params = {
    'booster': 'gbtree',
    'colsample_bytree': 0.8271901311986853,
    'eval_metric': 'error',
    'gamma': 0.3523281663529724,
    'learning_rate': 0.030522858251466296,
    'max_depth': 8,
    'min_child_weight': 2,
    'n_estimators': 750,
    'objective': 'binary:logistic',
    'reg_alpha': 0.4211368866611682,
    'reg_lambda': 0.09667286427878569,
    'scale_pos_weight': 1,
    'silent': True,
    'subsample': 0.7604104067052311
}

rf_params = {
    'class_weight': 'balanced',
    'max_features': 0.9650625848433683,
    'min_samples_leaf': 2,
    'min_samples_split': 5,
    'n_estimators': 440
}

knn_params = {
    'algorithm': 'brute',
    'n_neighbors': 13,
    'weights': 'distance'
}

lr_params = {
    'C': 0.02,
    'class_weight': 'balanced',
    'penalty': 'l2'
}

svc_params = {
    'C': 10,
    'class_weight': 'balanced'
}

xgb_model = xgb.XGBClassifier(**xgb_params, random_state=RANDOM_SEED)
rf = RandomForestClassifier(**rf_params, n_jobs=-1, random_state=RANDOM_SEED)
knn = KNeighborsClassifier(**knn_params, n_jobs=-1)
lr = LogisticRegression(**lr_params, n_jobs=-1, random_state=RANDOM_SEED)
svc = SVC(**svc_params, probability=True, random_state=RANDOM_SEED)
voting_hard_model = VotingClassifier(estimators=[('xgb',xgb_model), ('rf',rf), ('knn',knn), ('lr',lr), ('svc',svc)],
                                voting='hard')

make_prediction(X_train, y_train, X_test, voting_hard_model, 'VOTING_HARD')
voting_soft_model = VotingClassifier(estimators=[('xgb',xgb_model), ('rf',rf), ('knn',knn), ('lr',lr), ('svc',svc)],
                                voting='soft')

make_prediction(X_train, y_train, X_test, voting_hard_model, 'VOTING_SOFT')
voting_soft_model = VotingClassifier(estimators=[('xgb',xgb_model), ('rf',rf)],
                                voting='soft')

make_prediction(X_train, y_train, X_test, voting_hard_model, 'VOTING_SOFT_XGB_RF')
stacking_model = SuperLearner(scorer='accuracy', random_state=RANDOM_SEED, verbose=2)

stacking_model.add([('xgb',xgb_model), ('rf',rf), ('knn',knn), ('lr',lr), ('svc',svc)])
stacking_model.add_meta(LogisticRegressionCV(cv=10, n_jobs=-1, random_state=RANDOM_SEED))

make_prediction(X_train, y_train, X_test, stacking_model, 'STACKING')
stacking_model = SuperLearner(scorer='accuracy', random_state=RANDOM_SEED, verbose=2)

stacking_model.add([('xgb',xgb_model), ('rf',rf)])
stacking_model.add_meta(LogisticRegressionCV(cv=10, n_jobs=-1, random_state=RANDOM_SEED))

make_prediction(X_train, y_train, X_test, stacking_model, 'STACKING_XGB_RF')
stacking_model = SuperLearner(scorer='auc', random_state=RANDOM_SEED, verbose=2)

stacking_model.add([('xgb',xgb_model), ('rf',rf), ('knn',knn), ('lr',lr), ('svc',svc)])
stacking_model.add_meta(LogisticRegressionCV(cv=10, scoring=make_scorer(roc_auc_score),
                                             n_jobs=-1, random_state=RANDOM_SEED))

make_prediction(X_train, y_train, X_test, stacking_model, 'STACKING_AUC')
# stacking xgb rf with model tuning using auc 0.84688
stacking_model = SuperLearner(scorer='auc', random_state=RANDOM_SEED, verbose=2)

stacking_model.add([('xgb',xgb_model), ('rf',rf)])
stacking_model.add_meta(LogisticRegressionCV(cv=10, scoring=make_scorer(roc_auc_score),
                                             n_jobs=-1, random_state=RANDOM_SEED))

make_prediction(X_train, y_train, X_test, stacking_model, 'STACKING_XGB_RF_AUC')
train_raw = pd.read_csv('../input/train.csv')
test_raw = pd.read_csv('../input/test.csv')

train_raw['train'] = 1
test_raw['train'] = 0
data_all = pd.concat([train_raw, test_raw], axis=0).reset_index(drop=True)

data_all.Embarked = data_all.Embarked.fillna(data_all.Embarked.mode()[0])

le = LabelEncoder()

data_all.Sex = le.fit_transform(data_all[['Sex']])
data_all.Embarked = le.fit_transform(data_all[['Embarked']])

data_all['family_size'] = data_all.SibSp + data_all.Parch + 1

def calc_family_size_bin(family_size):
    if family_size == 1:
        return 0
    elif family_size <= 4: 
        return 1
    else:
        return 2
        
data_all['family_size_bin'] = data_all.family_size.map(calc_family_size_bin)
data_all['name_title'] = data_all.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

name_title_dict = {
    'Capt': 'Mr',
    'Col': 'Mr',
    'Don': 'Mr',
    'Dona': 'Mrs',    
    'Dr': 'Dr',
    'Jonkheer': 'Mr',
    'Lady': 'Mrs',
    'Major': 'Mr',
    'Master': 'Master',
    'Miss': 'Miss',
    'Mlle': 'Miss',
    'Mme': 'Miss',
    'Mr': 'Mr',
    'Mrs': 'Mrs',
    'Ms': 'Mrs',
    'Rev': 'Mr',
    'Sir': 'Mr',
    'Countess': 'Mrs'
}

data_all['name_title_cat'] = data_all.name_title.map(name_title_dict)

data_all['last_name'] = data_all.Name.str.extract('([A-Za-z]+),', expand=False)
data_all['last_name_family_size'] = data_all.apply(lambda row: row.last_name + '_' + str(row.family_size), axis=1)
data_all['last_name_ticket'] = data_all.apply(lambda row: row.last_name + '_' + row.Ticket, axis=1)

ticket_df = data_all.groupby('Ticket', as_index=False)['PassengerId'].count()
ticket_df.columns = ['Ticket','ticket_count']

data_all = pd.merge(data_all, ticket_df, on=['Ticket'])
data_all = data_all.sort_values('PassengerId').reset_index(drop=True)

last_name_family_size_check = data_all[data_all.family_size > 1].groupby('last_name_family_size').agg({'Survived': lambda x: x.isnull().sum()}).reset_index()
last_name_family_size_check.columns = ['last_name_family_size','last_name_family_size_feature']

last_name_ticket_check = data_all[data_all.ticket_count > 1].groupby('last_name_ticket').agg({'Survived': lambda x: x.isnull().sum()}).reset_index()
last_name_ticket_check.columns = ['last_name_ticket','last_name_ticket_feature']

data_all = pd.merge(data_all, last_name_family_size_check, on='last_name_family_size', how='left')
data_all = data_all.sort_values('PassengerId').reset_index(drop=True)
data_all.last_name_family_size_feature = data_all.last_name_family_size_feature.fillna(0)

data_all = pd.merge(data_all, last_name_ticket_check, on='last_name_ticket', how='left')
data_all = data_all.sort_values('PassengerId').reset_index(drop=True)
data_all.last_name_ticket_feature = data_all.last_name_ticket_feature.fillna(0)

data_all.loc[data_all.last_name_family_size_feature == 0, 'last_name_family_size'] = 'X'
data_all.loc[data_all.last_name_ticket_feature == 0, 'last_name_ticket'] = 'X'

family_survival = data_all.groupby(['last_name_family_size','family_size']).Survived.sum().reset_index()
family_survival.columns = ['last_name_family_size','family_size','family_survival_sum']

family_ticket_count = data_all.groupby(['last_name','Ticket']).PassengerId.count().reset_index()
family_ticket_count.columns = ['last_name','Ticket','family_ticket_count']

family_ticket_survival = data_all.groupby(['last_name','Ticket']).Survived.sum().reset_index()
family_ticket_survival.columns = ['last_name','Ticket','family_ticket_survival_sum']

family_ticket_survival = pd.merge(family_ticket_count, family_ticket_survival, on=['last_name','Ticket'])

def calc_family_survival(row):
    family_survival = 0.5
    if row['family_size'] > 1 and row['family_survival_sum'] > 0:
        family_survival = 1
    elif row['family_size'] > 1 and row['family_survival_sum'] == 0:
        family_survival = 0
        
    return family_survival

family_survival['family_survival'] = family_survival.apply(calc_family_survival, axis=1)

def calc_family_ticket_survival(row):
    family_ticket_survival = 0.5
    if row['family_ticket_count'] > 1 and row['family_ticket_survival_sum'] > 0:
        family_ticket_survival = 1
    elif row['family_ticket_count'] > 1 and row['family_ticket_survival_sum'] == 0:
        family_ticket_survival = 0
        
    return family_ticket_survival

family_ticket_survival['family_ticket_survival'] = family_ticket_survival.apply(calc_family_ticket_survival,
                                                                                axis=1)

data_all = pd.merge(data_all, family_survival, on=['last_name_family_size','family_size'], how='left')
data_all = data_all.sort_values('PassengerId').reset_index(drop=True)

data_all = pd.merge(data_all, family_ticket_survival, on=['last_name','Ticket'], how='left')
data_all = data_all.sort_values('PassengerId').reset_index(drop=True)

def calc_family_ticket_count_bin(family_ticket_count):
    if family_ticket_count == 1:
        return 0
    elif family_ticket_count <= 2: 
        return 1
    elif family_ticket_count <= 4:
        return 2
    else:
        return 3
        
data_all['family_ticket_count_bin'] = data_all.family_ticket_count.map(calc_family_ticket_count_bin)

data_all.name_title_cat = le.fit_transform(data_all[['name_title_cat']])
data_all.last_name_family_size = le.fit_transform(data_all[['last_name_family_size']])
data_all.last_name_ticket = le.fit_transform(data_all[['last_name_ticket']])

data_all['fare_fixed'] = data_all.Fare/data_all.ticket_count
fare_median = data_all.fare_fixed.median()
data_all['fare_fixed'] = data_all.fare_fixed.fillna(fare_median)
data_all['fare_fixed_log'] = np.log1p(data_all.fare_fixed)

age_median_by_sex_title = data_all.groupby(['Sex', 'name_title'], as_index=False).Age.median()

data_all = pd.merge(data_all, age_median_by_sex_title, on=['Sex', 'name_title'])
data_all['Age'] = data_all.apply(lambda row: row.Age_x if not np.isnan(row.Age_x) else row.Age_y, axis=1)
data_all = data_all.drop(['Age_x','Age_y'], axis=1).sort_values('PassengerId').reset_index(drop=True)

def calc_age_bin(age):
    if age <= 15:
        return 0
    elif age <= 30:
        return 1
    elif age <= 60:
        return 2
    else:
        return 3
        
data_all['age_bin'] = data_all.Age.map(calc_age_bin)

def parse_ticket_str(ticket):
    arr = ticket.split()
    if not arr[0].isdigit():
        txt = arr[0].replace('.', '')
        txt = txt.split('/')[0]
        return re.findall('[a-zA-Z]+', txt)[0]
    else:
        return None
        
data_all['ticket_str'] = data_all.Ticket.map(parse_ticket_str)

def parse_ticket_number(ticket):
    arr = ticket.split()
    if len(arr) == 1 and arr[0].isdigit():
        return int(arr[0])
    elif len(arr) == 2 and arr[1].isdigit():
        return int(arr[1])
    else:
        if arr[-1].isdigit():
            return int(arr[-1])
        else:
            return np.nan
    
data_all['ticket_number'] = data_all.Ticket.map(parse_ticket_number)

def parse_ticket_num_len(ticket):
    arr = ticket.split()
    if len(arr) == 1 and arr[0].isdigit():
        return len(arr[0])
    elif len(arr) == 2 and arr[1].isdigit():
        return len(arr[1])
    else:
        if arr[-1].isdigit():
            return len(arr[-1])
        else:
            return -1
    
data_all['ticket_num_len'] = data_all.Ticket.map(parse_ticket_num_len)

data_all['ticket_num_len_4_prefix'] = data_all[data_all.ticket_num_len == 4].ticket_number.map(lambda x: int(str(x)[0]))
data_all['ticket_num_len_4_prefix_2'] = data_all[data_all.ticket_num_len == 4].ticket_number.map(lambda x: int(str(x)[:2]))

data_all['ticket_num_len_5_prefix'] = data_all[data_all.ticket_num_len == 5].ticket_number.map(lambda x: int(str(x)[0]))
data_all['ticket_num_len_5_prefix_2'] = data_all[data_all.ticket_num_len == 5].ticket_number.map(lambda x: int(str(x)[:2]))

data_all['ticket_num_len_6_prefix'] = data_all[data_all.ticket_num_len == 6].ticket_number.map(lambda x: int(str(x)[0]))
data_all['ticket_num_len_6_prefix_2'] = data_all[data_all.ticket_num_len == 6].ticket_number.map(lambda x: int(str(x)[:2]))
data_all['ticket_num_len_6_prefix_3'] = data_all[data_all.ticket_num_len == 6].ticket_number.map(lambda x: int(str(x)[:3]))

data_all.ticket_str = data_all.ticket_str.fillna('X')

data_all.ticket_num_len_4_prefix = data_all.ticket_num_len_4_prefix.fillna(-1)
data_all.ticket_num_len_4_prefix_2 = data_all.ticket_num_len_4_prefix_2.fillna(-1)
data_all.ticket_num_len_5_prefix = data_all.ticket_num_len_5_prefix.fillna(-1)
data_all.ticket_num_len_5_prefix_2 = data_all.ticket_num_len_5_prefix_2.fillna(-1)
data_all.ticket_num_len_6_prefix = data_all.ticket_num_len_6_prefix.fillna(-1)
data_all.ticket_num_len_6_prefix_2 = data_all.ticket_num_len_6_prefix_2.fillna(-1)
data_all.ticket_num_len_6_prefix_3 = data_all.ticket_num_len_6_prefix_3.fillna(-1)

data_all.ticket_num_len_4_prefix = data_all.ticket_num_len_4_prefix.astype(int)
data_all.ticket_num_len_4_prefix_2 = data_all.ticket_num_len_4_prefix_2.astype(int)
data_all.ticket_num_len_5_prefix = data_all.ticket_num_len_5_prefix.astype(int)
data_all.ticket_num_len_5_prefix_2 = data_all.ticket_num_len_5_prefix_2.astype(int)
data_all.ticket_num_len_6_prefix = data_all.ticket_num_len_6_prefix.astype(int)
data_all.ticket_num_len_6_prefix_2 = data_all.ticket_num_len_6_prefix_2.astype(int)
data_all.ticket_num_len_6_prefix_3 = data_all.ticket_num_len_6_prefix_3.astype(int)

data_all.ticket_str = le.fit_transform(data_all[['ticket_str']])

data_all['cabin_cat'] = data_all.Cabin.fillna('X').str[0]

def calc_cabin_len(cabin):
    if type(cabin) == float:
        return 0
    else:
        return len(cabin.split())

data_all['cabin_len'] = data_all.Cabin.map(calc_cabin_len)

data_all.cabin_cat = le.fit_transform(data_all[['cabin_cat']])
features = [
    'Pclass', 'Sex', 'Age', 
    'family_size_bin',
    'name_title_cat', 
    'last_name_family_size',
    'last_name_ticket',
    'family_ticket_survival',
    'fare_fixed_log',
    'ticket_str',
    'ticket_num_len_4_prefix_2',
    'ticket_num_len_5_prefix_2',
    'ticket_num_len_6_prefix_3',
]

X_train = data_all[data_all.train == 1][features]
X_test = data_all[data_all.train == 0][features]
y_train = data_all[data_all.train == 1].Survived.astype(int)

X_train.shape, X_test.shape
def cv_model(train, train_labels, model, name, cv=10, scoring=make_scorer(precision_score)):
    """Perform k fold cross validation of a model"""
    
    cv_scores = cross_val_score(model, train, train_labels, cv=cv, scoring=scoring, n_jobs=-1)
    print(f'{cv} Fold CV {scoring} for {name}: {round(cv_scores.mean(), 5)} with std: {round(cv_scores.std(), 5)}')
    
def make_prediction(train, target, test, model, model_name):
    cv_model(train, target, model, model_name)

    model.fit(train, target)
    pred = model.predict(test).astype(int)

    output = f'{model_name}_submission_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
    submit_df = pd.DataFrame()
    submit_df['PassengerId'] = test_raw.PassengerId
    submit_df['Survived'] = pred
    
    submit_df[['PassengerId','Survived']].to_csv(output, index=False)
    print(f'submission file {output} is generated.')
# Public, Private Both 0.82296.
# n_estimators=1000, max_depth=2, using family_ticket_survival feature
rf = RandomForestClassifier(n_estimators=1000, max_depth=2, n_jobs=-1, random_state=RANDOM_SEED)
make_prediction(X_train, y_train, X_test, rf, 'RF_FINAL')
def plot_feature_importances(estimator, x_cols, n=20, threshold = 0.95):
    try:
        df = pd.DataFrame({'feature': x_cols, 'importance': estimator.feature_importances_})
    except AttributeError:
        print('model does not provide feature importances')
        return
    
    # Sort features with most important at the head
    df = df.sort_values('importance', ascending = False).reset_index(drop = True)
    
    # Normalize the feature importances to add up to one and calculate cumulative importance
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    
    plt.rcParams['font.size'] = 12
    
    # Bar plot of n most important features
    df.loc[:n, :].plot.barh(y = 'importance_normalized', 
                            x = 'feature', color = 'darkgreen', 
                            edgecolor = 'k', figsize = (12, 8),
                            legend = False, linewidth = 2)

    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 
    plt.title(f'{min(n, len(df))} Most Important Features', size = 18)
    plt.gca().invert_yaxis()
    
    if threshold:
        # Cumulative importance plot
        plt.figure(figsize = (8, 6))
        plt.plot(list(range(1, len(df)+1)), df['cumulative_importance'], 'b-')
        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 
        plt.title('Cumulative Feature Importance', size = 18);
        
        # Number of features needed for threshold cumulative importance
        # This is the index (will need to add 1 for the actual number)
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
        
        # Add vertical line to plot
        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.05, linestyles = '--', colors = 'red')
        plt.show();
        
        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 
                                                                                  100 * threshold))
    
    print(f'zero importance feature count : {len(df[df.importance == 0])}')
    
    return df
rf.fit(X_train, y_train)
plot_feature_importances(rf, features)
