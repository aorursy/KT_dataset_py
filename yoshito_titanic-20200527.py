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
import numpy as np

import matplotlib.pyplot as plt

from IPython.display import display

import seaborn as sns
pd.set_option('display.max_columns' , 100)
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
gensub_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
test_df.head()
gensub_df.head()
train_df.head()
train_df.describe(include='all')
train_df.info()
test_df.info()
# dtype

for x in train_df.columns:

    print(x, ' ', train_df[x].dtype)
# check if contains null

train_df.loc[:, train_df.isnull().any()]
# check null num

for x in train_df.columns:

    val = train_df[x].isnull().sum()

    if val > 0:

        print(x, val)

    
import missingno
missingno.matrix(train_df, figsize = (10,5))

plt.title("Missing Values (train)")

plt.show()
train_df[train_df['Age'].isnull()]['Survived'].value_counts()
train_df[train_df['Cabin'].isnull()]['Survived'].value_counts()
from sklearn.model_selection import train_test_split

from sklearn.metrics import *



def train_val(df, clf, target_col='Survived'):

    # train val split

    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target_col]), df[target_col], 

                                                        test_size=0.33, random_state=0, stratify=df[target_col])

    

    print('label dist')

    print('train =', np.unique(y_train, return_counts=True))

    print('val   =', np.unique(y_test, return_counts=True))

    

    clf.fit(X_train, y_train) # train

    preds = clf.predict(X_test) # predict

    print('accuracy =', accuracy_score(y_test, preds)) # score

    return clf
import lightgbm as lgb

import xgboost as xgb
clf = lgb.LGBMClassifier(objective='binary')
# exclude object

_train_df = train_df.copy()

_test_df = test_df.copy()



cols = _train_df.columns

for x in cols:

    if _train_df[x].dtype == 'object':

        del _train_df[x]

        del _test_df[x]
train_val(_train_df, clf)
clf = xgb.XGBClassifier()

train_val(_train_df, clf)
pd.crosstab(train_df['Sex'], train_df['Survived'], normalize=True)
plt.figure(figsize = (14,2))

sns.countplot(data = train_df, y = "Sex", hue = "Survived", palette = "viridis")

plt.title("Gender")

plt.show()

# exclude object

_train_df = train_df.copy()

_test_df = test_df.copy()



cols = _train_df.columns

for x in cols:

    if _train_df[x].dtype == 'object':

        del _train_df[x]

        del _test_df[x]
_train_df = pd.concat([_train_df, pd.get_dummies(train_df['Sex'], drop_first=True)], axis='columns')

_test_df = pd.concat([_test_df, pd.get_dummies(test_df['Sex'], drop_first=True)], axis='columns')
_train_df.head()
_train_df['Age'] = _train_df['Age'].fillna(_train_df['Age'].median())

_test_df['Age'] = _test_df['Age'].fillna(_train_df['Age'].median())
clf = lgb.LGBMClassifier(objective='binary')

train_val(_train_df, clf)

# improve accuracy!
clf = xgb.XGBClassifier()

train_val(_train_df, clf)

# improve
plt.figure(figsize=(20, 4))

plt.scatter(train_df['Age'], train_df['Survived'])
plt.figure(figsize=(20, 4))

train_df.query('Survived == 1')['Age'].value_counts().sort_index().plot.bar()

plt.xlim([0, train_df['Age'].max()]); plt.show()
plt.figure(figsize=(20, 4))

train_df.query('Survived == 0')['Age'].value_counts().sort_index().plot.bar()

plt.xlim([0, train_df['Age'].max()]); plt.show()
# real value -> bin by 10

_train_df['Age_bin_10'] = (_train_df['Age'] // 10) * 10

_train_df['Age_bin_5'] = (_train_df['Age'] // 5) * 5



_test_df['Age_bin_10'] = (_test_df['Age'] // 10) * 10

_test_df['Age_bin_5'] = (_test_df['Age'] // 5) * 5
_train_df.head()
clf = lgb.LGBMClassifier(objective='binary')

train_val(_train_df, clf)

# no change
clf = xgb.XGBClassifier()

train_val(_train_df, clf)

# no change
for x in train_df['Embarked'].unique():

    print(x)

    if x != x:

        tmp = train_df[train_df['Embarked'].isnull()]        

    else:

        tmp = train_df.query('Embarked == @x')



    display(tmp['Survived'].value_counts())
_train_df = pd.concat([_train_df, pd.get_dummies(train_df['Embarked'], drop_first=True)], axis='columns')

_test_df = pd.concat([_test_df, pd.get_dummies(test_df['Embarked'], drop_first=True)], axis='columns')
clf = lgb.LGBMClassifier(objective='binary')

train_val(_train_df, clf)

# improve accuracy!
clf = xgb.XGBClassifier()

train_val(_train_df, clf)

# improve
train_df['Cabin'].value_counts()
train_df[train_df['Cabin'].isnull()]['Survived'].value_counts().plot.bar()
val = train_df['Cabin'].fillna('').astype(str).str.split(' ').apply(len)

val[train_df['Cabin'].isnull()] = 0
for i in val.unique():

    print(i)

    train_df[val == i]['Survived'].value_counts().plot.bar()

    plt.show()
# _train_df['Cabin_num'] = val
def add_feat4(df, orig_df):

    val = orig_df['Cabin'].fillna('').astype(str).str.split(' ').apply(len)

    val[orig_df['Cabin'].isnull()] = 0

    df['Cabin_num'] = val

    return df
_train_df = add_feat4(_train_df, train_df)

_test_df = add_feat4(_test_df, test_df)
_train_df.head()
clf = lgb.LGBMClassifier(objective='binary')

train_val(_train_df, clf)

# improve!
clf = xgb.XGBClassifier()

train_val(_train_df, clf)

# worse
# del _train_df['Cabin_num']
# cabin null flag

# _train_df['Cabin_null'] = 0

# _train_df.loc[train_df['Cabin'].isnull(), 'Cabin_null'] = 1
def add_feat5(df, orig_df):

    df['Cabin_null'] = 0

    df.loc[orig_df['Cabin'].isnull(), 'Cabin_null'] = 1

    return df
_train_df = add_feat5(_train_df, train_df)

_test_df = add_feat5(_test_df, test_df)
_train_df.head()
clf = lgb.LGBMClassifier(objective='binary')

train_val(_train_df, clf)

# no change
clf = xgb.XGBClassifier()

train_val(_train_df, clf)

# no change
train_df['Cabin']
train_df['Cabin'].fillna('').str.split(' ')
train_df['Cabin'].fillna('').str.split(' ').apply(lambda x : [y.rstrip('1234567890') for y in x] )
val = train_df['Cabin'].fillna('').str.split(' ').apply(lambda x : [y.rstrip('1234567890') for y in x] ).apply(

    lambda x : 1 if 'C' in ''.join(x) else 0)
train_df.loc[val, 'Survived'].value_counts().plot.bar()
train_df.loc[val, 'Survived'].value_counts().plot.bar()
train_df['Cabin'].fillna('').str.split(' ').apply(lambda x : ''.join([y.rstrip('1234567890') for y in x]) )
train_df['Cabin'].fillna('').str.split(' ').apply(lambda x : ''.join([y.rstrip('1234567890') for y in x]) ).value_counts()
# add one hot

# _train_df = pd.concat([_train_df, pd.get_dummies(train_df['Cabin'].fillna('').str.split(' ').apply(lambda x : ''.join([y.rstrip('1234567890') for y in x]) ),

#                                                prefix='Cabin_onehot', drop_first=True)], axis='columns')
# def add_feat6(df, orig_df):

#     def _func(x):

#         x.sort()

#         return ''.join([y.rstrip('1234567890') for y in x])

    

#     df = pd.concat([df, 

#                     pd.get_dummies(orig_df['Cabin'].fillna('').str.split(' ').apply(_func), prefix='Cabin_onehot', drop_first=True)],

#                    axis='columns')

#     return df
# _train_df = add_feat6(_train_df, train_df)

# _test_df = add_feat6(_test_df, test_df)
_train_df.head()
clf = lgb.LGBMClassifier(objective='binary')

train_val(_train_df, clf)

# improve!
clf = xgb.XGBClassifier()

train_val(_train_df, clf)

# worse
def add_feat7(df, orig_df):

    for x in list('ABCDEF'):

        df['Cabin_contains_' + x] = 0

        df.loc[orig_df['Cabin'].fillna('').str.split(' ').apply(lambda x : ''.join([y.rstrip('1234567890') for y in x])

                                                                       ).str.contains(x),

                  'Cabin_contains_' + x] = 1

    return df
_train_df = add_feat7(_train_df, train_df)

_test_df = add_feat7(_test_df, test_df)
_train_df.head()
train_df['Cabin'], _train_df['Cabin_contains_C']
clf = lgb.LGBMClassifier(objective='binary')

train_val(_train_df, clf)

# improve!
clf = xgb.XGBClassifier()

train_val(_train_df, clf)

# worse
def add_feat8(df, orig_df):

    vc = orig_df['Sex'].value_counts(normalize=True)

    df['Gender_Freq'] = orig_df['Sex'].apply(lambda x : vc[x])

    return df
_train_df = add_feat8(_train_df, train_df)

_test_df = add_feat8(_test_df, test_df)
_train_df.head()
clf = lgb.LGBMClassifier(objective='binary')

train_val(_train_df, clf)

# no change
clf = xgb.XGBClassifier()

train_val(_train_df, clf)

# no change
def add_feat9(df, orig_df):

    vc = orig_df['Cabin'].astype(str).value_counts(normalize=True)

    df['Cabin_Freq'] = orig_df['Cabin'].apply(lambda x : vc[str(x)])

    return df
_train_df = add_feat9(_train_df, train_df)

_test_df = add_feat9(_test_df, test_df)
_train_df.head()
clf = lgb.LGBMClassifier(objective='binary')

train_val(_train_df, clf)

# worse
clf = xgb.XGBClassifier()

train_val(_train_df, clf)

# improve
def add_feat10(df, orig_df):

    df['Age'] = df['Age'].fillna(df['Age'].median())

    df['Age_null'] = 0

    df.loc[df['Age'].isnull(), 'Age_null'] = 1

    return df
_train_df = add_feat10(_train_df, train_df)

_test_df = add_feat10(_test_df, test_df)
clf = lgb.LGBMClassifier(objective='binary')

train_val(_train_df, clf)

# worse
clf = xgb.XGBClassifier()

train_val(_train_df, clf)

# improve
train_df['Name']
train_df.Name.str.split(' ').apply(lambda xs : [x.rstrip('.,') for x in xs])
train_df['Name'].str.replace('. ', ',').str.split(',', expand=True).stack().value_counts()
def add_feat11(df, orig_df):

    for x in ['Mr', 'Miss', 'Mrs']:

        col = 'Name_contains_' + x

        df[col] = 0

        df.loc[orig_df['Name'].str.contains(x, na=False), col] = 1

    return df
_train_df = add_feat11(_train_df, train_df)

_test_df = add_feat11(_test_df, test_df)
_train_df.head()
clf = lgb.LGBMClassifier(objective='binary')

train_val(_train_df, clf)

# worse
clf = xgb.XGBClassifier()

train_val(_train_df, clf)
import tensorflow.keras

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization, concatenate, AveragePooling2D, Add

def mymodel():

    _input = Input((len(_train_df.columns) - 1))

    

#     x = Conv2D(64, [7, 7], strides=2, padding='same', activation=None, name="conv1")(x)

#     x = BatchNormalization(name="bn1")(x)

#     x = Activation("relu")(x)

#     x = MaxPooling2D([3, 3], strides=2, padding='same')(x)



#     x = ResBlock(x, 64, 64, name="res2_1")

#     x = ResBlock(x, 64, 64, name="res2_2")



#     x = ResBlock(x, 64, 128, stride=2, name="res3_1")

#     x = ResBlock(x, 128, 128, name="res3_2")



#     x = ResBlock(x, 128, 256, stride=2, name="res4_1")

#     x = ResBlock(x, 256, 256, name="res4_2")



#     x = ResBlock(x, 256, 512, stride=2, name="res5_1")

#     x = ResBlock(x, 512, 512, name="res5_2")



#     x = AveragePooling2D([7, 7], strides=1, padding='valid')(x)

#     x = Flatten()(x)

    x = Dense(10, activation='relu')(_input)

    x = BatchNormalization()(x)

    x = Dropout(0.2)(x)

    x = Dense(10, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.2)(x)

    x = Dense(1, activation='sigmoid')(x)



    model = Model(inputs=_input, outputs=x)

    return model
model = mymodel()



model.compile(

        loss='binary_crossentropy',

        optimizer=tensorflow.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),

        metrics=['accuracy'])
def train_val_keras(df, clf, target_col='Survived'):

    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target_col]), df[target_col], 

                                                        test_size=0.33, random_state=0, stratify=df[target_col])

    

    print('label dist')

    print('train =', np.unique(y_train, return_counts=True))

    print('val   =', np.unique(y_test, return_counts=True))

    

    clf.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test)) # train

    preds = clf.predict(X_test) # predict

    preds = preds // 0.5

    print('accuracy =', accuracy_score(y_test, preds)) # score

    return clf
train_val_keras(_train_df, model)
clf = lgb.LGBMClassifier(objective='binary')

clf = train_val(_train_df, clf)
plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
import eli5

from eli5.sklearn import PermutationImportance
_train_df
# check if contains null

_train_df.loc[:, _train_df.isnull().any()]
perm = PermutationImportance(clf).fit(_train_df.drop(columns='Survived'), _train_df['Survived'])

eli5.show_weights(perm)
perm.feature_importances_ / perm.feature_importances_.max()
import optuna

import sklearn
# verbose?

def objective(trial):

#     n_estimators = trial.suggest_categorical('n_estimators', range(100, 350, 50))

#     #max_depth = int(trial.suggest_loguniform('max_depth', 1, 32))

#     criterion = trial.suggest_categorical('learning_rate', np.arange(0.01, 0.11, 0.01))

#     num_leaves = trial.suggest_categorical('num_leaves', [11, 21, 31, 41])

#     max_depth = trial.suggest_categorical('max_depth', [-1, 10, 20, 30])

    

    param = {

        'metric': 'binary_logloss',

        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),

        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),

        'num_leaves': trial.suggest_int('num_leaves', 2, 256),

        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),

        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),

        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),

        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),

    }

    

    # train val split

    target_col = 'Survived'

    df = _train_df

    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target_col]), df[target_col], 

                                                        test_size=0.33, random_state=0, stratify=df[target_col])

    

    clf = lgb.LGBMClassifier(objective='binary', **param)

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds) # accuracy

    return acc

    #clf.fit(_train_df.drop(columns='Survived'), _train_df['Survived'])

    #preds = clf.predict(_train_df.drop(columns='Survived')) # predict

    #return accuracy_score(_train_df['Survived'], preds) # accuracy

    #return sklearn.model_selection.cross_val_score(clf, _train_df.drop(columns='Survived'), _train_df['Survived'], n_jobs=-1, cv=3).mean()



study = optuna.create_study(direction='maximize')

study.optimize(objective, n_trials=100)



trial = study.best_trial
print('Accuracy: {}'.format(trial.value))

print("Best parameters:")

print(trial.params)
clf = lgb.LGBMClassifier(objective='binary', **trial.params)

clf.fit(_train_df.drop(columns='Survived'), _train_df['Survived'])

from sklearn.model_selection import cross_validate

from sklearn.model_selection import StratifiedKFold


skf = StratifiedKFold(n_splits=5)

skf.get_n_splits(_train_df.drop(columns='Survived'), _train_df['Survived'])
clf = lgb.LGBMClassifier(objective='binary', **trial.params)

cv_results = cross_validate(clf, _train_df.drop(columns='Survived'), _train_df['Survived'], cv=skf, return_train_score=True, return_estimator=True, n_jobs=-1,

                            scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']) # cross validation
cv_results
_pred = np.zeros(len(_test_df))



for _clf in cv_results['estimator']:

    pred = _clf.predict(_test_df)

    _pred += pred
_pred
preds = (_pred // 2.6).astype(int)
preds.astype(int)
# preds = clf.predict(_test_df)
preds
[x for x in _test_df.columns if x not in _train_df.columns]
gensub_df.head()
submit_df = pd.DataFrame({

    'PassengerId' : gensub_df['PassengerId'],

    'Survived' : preds

})
submit_df.to_csv('submission.csv', index=False)