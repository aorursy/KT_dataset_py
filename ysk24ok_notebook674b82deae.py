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
input_dir = "/kaggle/input/lish-moa/"

train_features_df = pd.read_csv(input_dir + "train_features.csv")

print('train_features shape: ', train_features_df.shape)

test_features_df = pd.read_csv(input_dir + "test_features.csv")

print('test_features shape: ', test_features_df.shape)

train_targets_scored_df = pd.read_csv(input_dir + "train_targets_scored.csv")

print('train_targets_scored shape: ', train_targets_scored_df.shape)

train_targets_nonscored_df = pd.read_csv(input_dir + "train_targets_nonscored.csv")

print('train_targets_nonscored shape: ', train_targets_nonscored_df.shape)
train_features_df
train_targets_scored_df
from sklearn.preprocessing import LabelEncoder



def preprocess(df):

    cp_type, cp_time, cp_dose = df['cp_type'], df['cp_time'], df['cp_dose']

    encoded_cp_type = LabelEncoder().fit(cp_type).transform(cp_type)

    encoded_cp_time = LabelEncoder().fit(cp_time).transform(cp_time)

    encoded_cp_dose = LabelEncoder().fit(cp_dose).transform(cp_dose)

    X = df.to_numpy()[:, 4:]  # delete sig_id, cp_type, cp_time and cp_dose

    X = np.concatenate([

        #cp_type.to_numpy()[:, numpy.newaxis],

        encoded_cp_type[:, np.newaxis],

        encoded_cp_time[:, np.newaxis],

        encoded_cp_dose[:, np.newaxis],

        X,

    ], axis=1)

    #cat_features = [0]

    return X#, cat_features
X_train = preprocess(train_features_df)

X_test = preprocess(test_features_df)

print(X_train.shape, X_test.shape)
def create_targets(df):

    Y = df.to_numpy()[:, 1:]  # delete sig_id

    return Y.astype(np.int32)
Y_train = create_targets(train_targets_scored_df)

Y_train.shape
from catboost import CatBoostClassifier

from sklearn.multioutput import MultiOutputClassifier





def create_multioutput_model():

    catboost_model = CatBoostClassifier(

        iterations=10,

        depth=2,

        learning_rate=1,

        loss_function='Logloss',

        verbose=False)

    print(catboost_model.get_params())

    return MultiOutputClassifier(catboost_model)
model = create_multioutput_model()

print(model.get_params())
from sklearn.model_selection import KFold



def train_with_cv(model, X, Y, n_splits):

    kfold = KFold(n_splits)

    for k, (train_indices, test_indices) in enumerate(kfold.split(X)):

        X_train, Y_train, X_test, Y_test = X[train_indices], Y[train_indices], X[test_indices], Y[test_indices]

        print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

        model.fit(X_train, Y_train)

        print("{} fold, log_loss: {}".format(k, evaluate(model, X_test, Y_test)))
#train_with_cv(model, X_train, Y_train, 3)
def train_without_cv(model, X, Y):

    model.fit(X, Y)
train_without_cv(model, X_train, Y_train)
def predict(model, X):

    Y_pred = model.predict_proba(X)

    return np.array([y_pred[:, 1] for y_pred in Y_pred]).T
from sklearn.metrics import log_loss



def evaluate(model, X, Y_true):

    Y_pred = predict(model, X)

    return log_loss(Y_true, Y_pred)
#evaluate(model, X_train, Y_train)
import os

import pickle



def save(model, name):

    if not os.path.exists('models'):

        os.mkdir('models')

    with open('models/{}.pkl'.format(name), mode='wb') as f:

        pickle.dump(model, f)

        

def load(name):

    with open('models/{}.pkl'.format(name), mode='rb') as f:

        return pickle.load(f)
save(model, 'basic_model')
#model = load('basic_model')
def create_test_targets_dummy(columns, test_features_df):

    sig_id = test_features_df['sig_id'].to_numpy()[:, np.newaxis]  # 3982 x 1

    predicted = np.zeros((test_features_df.shape[0], columns.shape[0]-1))  # 3982 x 206

    return pd.DataFrame(

        np.concatenate([sig_id, predicted], axis=1),

        columns=train_targets_scored_df.columns

    )
def create_test_targets(columns, test_features_df, predicted):

    assert predicted.shape[0] == test_features_df.shape[0] and predicted.shape[1] == columns.shape[0]-1

    sig_id = test_features_df['sig_id'].to_numpy()[:, np.newaxis]  # 3982 x 1

    df = pd.DataFrame(

        np.concatenate([sig_id, predicted], axis=1),

        columns=train_targets_scored_df.columns

    )

    return df
test_targets = create_test_targets(train_targets_scored_df.columns, test_features_df, predict(model, X_test))

#test_targets = create_test_targets_dummy(train_targets_scored_df.columns, test_features_df)

print(test_targets)

test_targets.to_csv('submission.csv', index=False)