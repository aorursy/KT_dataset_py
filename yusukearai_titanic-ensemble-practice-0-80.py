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
'''
■Contents
✔ Feature engineering
✔ Ensemble:stacking
   ・xgboost
   ・NeuralNet
   ・LogisticRegression
'''
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.layers.advanced_activations import ReLU, PReLU
from keras.optimizers import SGD, Adam
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
#tensorflow warning suppression
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#xgboost model
class Model1Xgb:

    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        #parameter list
        params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eta': 0.1,
            'gamma': 0.1,
            'reg_alpha':0.01 ,
            'reg_lambda': 0,
            'min_child_weight': 4,
            'max_depth': 9,
            'subsample': 0.8,
            'colsample_bytree': 0.75,
            'scale_pos_weight': 1,
            'random_state': 1
}
        
        num_round = 30
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.model = xgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred

#NeuralNet model
class Model1NN:

    def __init__(self, params):
        self.params = params
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        
        input_dropout = self.params['input_dropout']
        hidden_layers = int(self.params['hidden_layers'])
        hidden_units = int(self.params['hidden_units'])
        hidden_activation = self.params['hidden_activation']
        hidden_dropout = self.params['hidden_dropout']
        batch_norm = self.params['batch_norm']
        optimizer_type = self.params['optimizer']['type']
        optimizer_lr = self.params['optimizer']['lr']
        batch_size = int(self.params['batch_size'])
        
        self.scaler = StandardScaler()
        self.scaler.fit(tr_x)

        batch_size = 32
        epochs = 20

        tr_x = self.scaler.transform(tr_x)
        va_x = self.scaler.transform(va_x)
        
        model = Sequential()

        #input layer 
        model.add(Dropout(input_dropout, input_shape=(tr_x.shape[1],)))

        #middle layer
        for i in range(hidden_layers):
            model.add(Dense(hidden_units))
            if batch_norm == 'before_act':
                model.add(BatchNormalization())
            if hidden_activation == 'prelu':
                model.add(PReLU())
            elif hidden_activation == 'relu':
                model.add(ReLU())
            else:
                raise NotImplementedError
            model.add(Dropout(hidden_dropout))

        #output layer
        model.add(Dense(1, activation='sigmoid'))

        #optimizer
        if optimizer_type == 'sgd':
            optimizer = SGD(lr=optimizer_lr, decay=1e-6, momentum=0.9, nesterov=True)
        elif optimizer_type == 'adam':
            optimizer = Adam(lr=optimizer_lr, beta_1=0.9, beta_2=0.999, decay=0.)
        else:
            raise NotImplementedError

        #Setting of objective function and evaluation index
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        history = model.fit(tr_x, tr_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(va_x, va_y))
        self.model = model

    def predict(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict_proba(x).reshape(-1)
        return pred

#LogisticRegression model
class Model2Linear:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.scaler = StandardScaler()
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        self.model = LogisticRegression(solver='lbfgs', C=100, random_state=1)
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict_proba(x)[:, 1]
        return pred

#Output predicted value
def predict_cv(model, train_x, train_y, test_x):
    preds = []
    preds_test = []
    va_idxes = []

    #Performs learning/prediction with cross-validation 
    kf = KFold(n_splits=4, shuffle=True, random_state=1)
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        model.fit(tr_x, tr_y, va_x, va_y)
        pred = model.predict(va_x)
        preds.append(pred)
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)
        va_idxes.append(va_idx)

    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_idxes)
    pred_train = preds[order]

    preds_test = np.mean(preds_test, axis=0)

    return pred_train, preds_test

#Data preprocessing for xgboost---------------------------------------------------------------------------
train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')
train_y = train_df['Survived']

#Removed Ticket and Cabin features from test_df
train_df = train_df.drop(["Ticket", "Cabin"], axis=1)
test_df = test_df.drop(["Ticket", "Cabin"], axis=1)

#Store name in 'Title'
combine = [train_df, test_df]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#Cross-tabulate using 'Sex' and 'Title'
df_list = pd.crosstab(train_df['Title'], train_df['Sex'])

#Replace all but frequent values with 'Rare'
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    #Replace Mile with Miss.
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    #MmeはMrsに書き換えてください。
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#Calculate Survived average 
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

#Convert to ordinal data
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

combine = [train_df, test_df]
for dataset in combine:
    dataset["Title"] = dataset["Title"] .map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    dataset['Title'] = dataset['Title'].fillna(0)

#Removed 'Name' and 'PassangerId' from train_df
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

#Removed 'Name' from test_df
test_df = test_df.drop(['Name'], axis=1)

#Numerical conversion of categorical variables
combine = [train_df, test_df]
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#'Age' Convert to discrete value
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

#Create a pivot table for 'AgeBand' and 'Survived'
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
    
#Removed 'AgeBand' from train_df
train_df = train_df.drop(['AgeBand'], axis=1)

#Create a new feature called 'FamilySize' ('SibSp' + 'Parch')
combine = [train_df, test_df]
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
#Create a pivot table for 'FamilySize' and 'Survived'
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#Create a new feature called isAlone
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
#Create a pivot table for 'IsAlone' and 'Survived'
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

#train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

#.dropna()
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

#Create a pivot table for 'Embarked' and 'Survived'
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#Replace Embarked with {'S': 0,'C': 1,'Q': 2}
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#Fill with missing values
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

#'Fare' Convert to discrete value
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

#Create a pivot table for 'FareBand' and 'Survived'
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

combine = [train_df, test_df]
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand','Survived'], axis=1)
test_df = test_df.drop(['PassengerId'], axis=1)

#Fill with missing values
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean()) 
train_df['Age*Class'] = train_df['Age*Class'].fillna(train_df['Age*Class'].mean()) 
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean()) 
test_df['Age*Class'] = test_df['Age*Class'].fillna(test_df['Age*Class'].mean()) 
    
train_x = train_df.copy()
test_x = test_df.copy()

#Data preprocessing for NeuralNet-------------------------------------------------------------------------
train_nn = pd.read_csv('../input/titanic/train.csv')
train_x_nn = train_df.copy()
train_y_nn = train_nn['Survived']
test_x_nn = pd.read_csv('../input/titanic/test.csv')
test_x_nn = test_df.copy()

base_param = {
    'input_dropout': 0.05,
    'hidden_layers': 2.0,
    'hidden_units': 96.0,
    'hidden_activation': 'prelu',
    'hidden_dropout': 0.05,
    'batch_norm': 'before_act',
    'optimizer': {'type': 'adam', 'lr': 0.00037640141509672924},
    'batch_size': 32.0}

#Stacking first layer------------------------------------------------------------------------------------
model_1a = Model1Xgb()
pred_train_1a, pred_test_1a = predict_cv(model_1a, train_x, train_y, test_x)

model_1b = Model1NN(base_param)
pred_train_1b, pred_test_1b = predict_cv(model_1b, train_x_nn, train_y_nn, test_x_nn)

#Evaluation for first layer
print(f'logloss: {log_loss(train_y, pred_train_1a, eps=1e-7):.4f}')
print(f'logloss: {log_loss(train_y, pred_train_1b, eps=1e-7):.4f}')

#Create a DataFrame using predicted values as features
train_x_2 = pd.DataFrame({'pred_1a': pred_train_1a, 'pred_1b': pred_train_1b})
test_x_2 = pd.DataFrame({'pred_1a': pred_test_1a, 'pred_1b': pred_test_1b})

#Stacking second layer----------------------------------------------------------------------------------
model_2 = Model2Linear()
pred_train_2, pred_test_2 = predict_cv(model_2, train_x_2, train_y, test_x_2)

#Evaluation for second layer
print(f'logloss: {log_loss(train_y, pred_train_2, eps=1e-7):.4f}')

#Binary conversion--------------------------------------------------------------------------------------
pred_test_2 = np.where(pred_test_2 < 0.5, 0, 1)
pred_test_2 = pred_test_2.tolist()
print(pred_test_2)

#Convert data for submission----------------------------------------------------------------------------
sub = pd.read_csv('../input/titanic/gender_submission.csv')
sub['Survived'] = list(map(int, pred_test_2))
sub.to_csv('submission_stacking3.csv', index=False)

