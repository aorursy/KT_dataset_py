from IPython.core.display import display, HTML

display(HTML("<style>.container { width:90% !important; }</style>"))
%matplotlib inline



import pandas as pd

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns



sns.set() # set seaborn default
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")



#from enum import Enum

class Columns:

    # existing features

    PassengerId = "PassengerId"

    Survived = "Survived"

    Pclass = "Pclass"

    Name = "Name"

    Sex = "Sex"

    Age = "Age"

    SibSp = "SibSp"

    Parch = "Parch"

    Ticket = "Ticket"

    Fare = "Fare"

    Cabin = "Cabin"

    Embarked = "Embarked"

    

    # new features

    Title = "Title"

    FareBand = "FareBand"

    Family = "Family"

    Deck = "Deck" # get character from existing 'Cabin' values

    CabinExists = "CabinExists"
train.head()
test.head()
print(train[[Columns.Pclass, Columns.Survived]].head())

train[[Columns.Pclass, Columns.Survived]].groupby([Columns.Pclass]).mean().plot.bar()
train[[Columns.Sex, Columns.Survived]].groupby([Columns.Sex]).mean().plot.bar()
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))

sns.countplot(x=Columns.Sex, hue=Columns.Survived, data=train, ax=ax[0])

sns.countplot(x=Columns.Sex, hue=Columns.Pclass, data=train, ax=ax[1])

sns.countplot(x=Columns.Pclass, hue=Columns.Survived, data=train, ax=ax[2])
train[Columns.Age].plot.kde()
df = train[train[Columns.Age].isnull() == False] # drop rows have no age



bincount = 12

age_min = int(df[Columns.Age].min())

age_max = int(df[Columns.Age].max())

print("Age :", age_min, " ~ ", age_max)

gap = int((age_max - age_min) / bincount)

print('gap:', gap)



bins = [-1]

for i in range(bincount):

    bins.append(i * gap)

bins.append(np.inf)

print(bins)



_df = df

_df['AgeGroup'] = pd.cut(_df[Columns.Age], bins)

fig, ax = plt.subplots(figsize=(20, 10))

sns.countplot(x='AgeGroup', hue=Columns.Survived, data=_df, ax=ax)
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16, 20))

sns.violinplot(x=Columns.Pclass, y=Columns.Age, hue=Columns.Survived, data=train, scale='count', split=True, ax=ax[0])

sns.violinplot(x=Columns.Sex, y=Columns.Age, hue=Columns.Survived, data=train, scale='count', split=True, ax=ax[1])

sns.violinplot(x=Columns.Pclass, y=Columns.Sex, hue=Columns.Survived, data=train, scale='count', split=True, ax=ax[2])

_train = train

_train['Family'] = _train[Columns.SibSp] + _train[Columns.Parch] + 1

_train[['Family', Columns.Survived]].groupby('Family').mean().plot.bar()
train[Columns.Age].plot.hist()
sns.countplot(x='Family', data=_train)
sns.countplot(x='Family', hue=Columns.Survived, data=_train)
train_len = train.shape[0]



merged = train.append(test, ignore_index=True) 

print("train len : ", train.shape[0])

print("test len : ", test.shape[0])

print("merged len : ", merged.shape[0])
merged[Columns.Family] = merged[Columns.Parch] + merged[Columns.SibSp] + 1

if Columns.Parch in merged:    

    merged = merged.drop([Columns.Parch], axis=1)

if Columns.SibSp in merged:

    merged = merged.drop([Columns.SibSp], axis=1)

    

merged.head()
most_embarked_label = merged[Columns.Embarked].value_counts().index[0]



merged = merged.fillna({Columns.Embarked : most_embarked_label})

merged.describe(include="all")
merged[Columns.Title] = merged.Name.str.extract('([A-Za-z]+)\. ', expand=False)



print("initial titles : ", merged[Columns.Title].value_counts().index)

#initial titles :  Index(['Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Col', 'Ms', 'Mlle', 'Major',

#                         'Sir', 'Jonkheer', 'Don', 'Mme', 'Countess', 'Lady', 'Dona', 'Capt'],



merged[Columns.Title] = merged[Columns.Title].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

merged[Columns.Title] = merged[Columns.Title].replace(['Countess', 'Lady', 'Sir'], 'Royal')

merged[Columns.Title] = merged[Columns.Title].replace(['Miss', 'Mlle', 'Ms', 'Mme'], 'Mrs')



print("Survival rate by title:")

print("========================")

print(merged[[Columns.Title, Columns.Survived]].groupby(Columns.Title).mean())



idxs = merged[Columns.Title].value_counts().index

print(idxs)



mapping = {}

for i in range(len(idxs)):

    mapping[idxs[i]] = i + 1

print("Title mapping : ", mapping)

merged[Columns.Title] = merged[Columns.Title].map(mapping)



if Columns.Name in merged:

    merged = merged.drop([Columns.Name], axis=1)

    

merged.head()
sns.countplot(x=Columns.Title, hue=Columns.Survived, data=merged)

print(merged[Columns.Title].value_counts())
mapping = {'male':0, 'female':1}

merged[Columns.Sex] = merged[Columns.Sex].map(mapping)
merged.head(n=10)
# {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5, 'Royal': 6}



mapping = {1:21, 2:28, 3:28, 4:40, 5:50, 6:60}

def guess_age(row):

    return mapping[row[Columns.Title]]



def fixup_age(df):

    for idx, row in df[df[Columns.Age].isnull() == True].iterrows():

        df.loc[idx, Columns.Age] = guess_age(row)

    return df

    

merged = fixup_age(merged)

merged.describe(include='all')
def make_deck(df):

    df[Columns.Deck] = df[Columns.Cabin].str.extract('([A-Za-z]+)', expand=True)

    return df



merged = make_deck(merged)

merged.describe(include='all')
merged[[Columns.Deck, Columns.Fare]].groupby(Columns.Deck).mean().sort_values(by=Columns.Fare)
sns.countplot(x=Columns.Deck, hue=Columns.Survived, data=merged)
print("total survived rate: ", merged[Columns.Survived].mean())

print("deck survived rate: ", merged[merged[Columns.Deck].isnull() == False][Columns.Survived].mean())

print("no deck survived rate: ", merged[merged[Columns.Deck].isnull()][Columns.Survived].mean())



fig, ax = plt.subplots(2, 1, figsize=(16, 16))

merged[[Columns.Deck, Columns.Survived]].groupby(Columns.Deck).mean().plot.bar(ax=ax[0])



def generate_fare_group(df, slicenum):

    if "FareGroup" in df:

        df.drop("FareGroup", axis=1)    

    _min = int(df[Columns.Fare].min())

    _max = int(df[Columns.Fare].max())

    print("Fare :", _min, " ~ ", _max)

    gap = int((_max - _min) / slicenum)

    print('gap:', gap)



    bins = [-1]

    for i in range(slicenum):

        bins.append(i * gap)

    bins.append(np.inf)

    print(bins)

    df['FareGroup'] = pd.cut(df[Columns.Fare], bins)    

    return df



df = generate_fare_group(merged.copy(), 16)



sns.countplot(x="FareGroup", hue=Columns.Survived, data=df[df[Columns.Deck].isnull()], ax=ax[1])
merged[Columns.CabinExists] = (merged[Columns.Cabin].isnull() == False)

merged[Columns.CabinExists] = merged[Columns.CabinExists].map({True:1, False:0})
merged.head()
merged[merged[Columns.Fare].isnull()]
merged.loc[merged[Columns.Fare].isnull(), [Columns.Fare]] = merged[Columns.Fare].mean()
merged.head()
sns.distplot(merged[Columns.Fare])
'''

log를 취하는 방법

'''



#merged[Columns.Fare] = merged[Columns.Fare].map(lambda i : np.log(i) if i > 0 else 0)



'''

등급을 4단계로 나누는 방법

'''

merged[Columns.FareBand] = pd.qcut(merged[Columns.Fare], 4, labels=[1,2,3,4]).astype('float')

#merged[Columns.Fare] = merged[Columns.FareBand]



merged.head(n=20)

merged[Columns.Fare] = merged[Columns.FareBand]

merged = merged.drop([Columns.FareBand], axis=1)

merged.head()
merged.head()
sns.distplot(merged[Columns.Fare])
merged.head()
if Columns.Ticket in merged:

    merged = merged.drop(labels=[Columns.Ticket], axis=1)

if Columns.Cabin in merged:

    merged = merged.drop(labels=[Columns.Cabin], axis=1)

if Columns.Deck in merged:

    merged = merged.drop(labels=[Columns.Deck], axis=1)
merged.describe(include='all')
merged.head()
merged = pd.get_dummies(merged, columns=[Columns.Pclass], prefix='Pclass')

merged = pd.get_dummies(merged, columns=[Columns.Title], prefix='Title')

merged = pd.get_dummies(merged, columns=[Columns.Embarked], prefix='Embarked')

merged = pd.get_dummies(merged, columns=[Columns.Sex], prefix='Sex')

merged = pd.get_dummies(merged, columns=[Columns.CabinExists], prefix='CabinExists')

merged.head()
from sklearn.preprocessing import MinMaxScaler



class NoColumnError(Exception):

    """Raised when no column in dataframe"""

    def __init__(self, value):

        self.value = value

    # __str__ is to print() the value

    def __str__(self):

        return(repr(self.value))



# normalize AgeGroup

def normalize_column(data, columnName):

    scaler = MinMaxScaler(feature_range=(0, 1))    

    if columnName in data:

        aaa = scaler.fit_transform(data[columnName].values.reshape(-1, 1))

        aaa = aaa.reshape(-1,)

        #print(aaa.shape)

        data[columnName] = aaa

        return data

    else:

        raise NoColumnError(str(columnName) + " is not exists!")



def normalize(dataset, columns):

    for col in columns:

        dataset = normalize_column(dataset, col)

    return dataset
merged.head()
merged = normalize(merged, [Columns.Age, Columns.Fare, Columns.Family])
merged.head(n=10)
train = merged[:train_len]

test = merged[train_len:]

test = test.drop([Columns.Survived], axis=1)



train = train.drop([Columns.PassengerId], axis=1)



test_passenger_id = test[Columns.PassengerId]

test = test.drop([Columns.PassengerId], axis=1)



print(train.shape)

print(test.shape)
train_X = train.drop([Columns.Survived], axis=1).values

train_Y = train[Columns.Survived].values.reshape(-1, 1)

print(train_X.shape)

print(train_Y.shape)
test.shape
test.describe(include='all')
train.head()
test.head()
import lightgbm as lgb

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import cohen_kappa_score, mean_squared_error

from sklearn import linear_model



from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold



import warnings



from bayes_opt import BayesianOptimization



n_splits = 6



def get_param(learning_rate, max_depth, lambda_l1, lambda_l2, 

              bagging_fraction, bagging_freq, colsample_bytree, subsample_freq, feature_fraction):

    params = {'n_estimators':5000,

                'boosting_type': 'gbdt',

                'objective': 'regression',

                'metric': 'rmse',

                #'eval_metric': 'cappa',

                'subsample' : 1.0,

                'subsample_freq' : subsample_freq,

                'feature_fraction' : feature_fraction,

                'n_jobs': -1,

                'seed': 42,                    

                'learning_rate': learning_rate,

                'max_depth': int(max_depth),

                'lambda_l1': lambda_l1,

                'lambda_l2': lambda_l2,

                'bagging_fraction' : bagging_fraction,

                'bagging_freq': int(bagging_freq),

                'colsample_bytree': colsample_bytree,

                'early_stopping_rounds': 100,

                'verbose' : 0

              }

    return params



def opt_test_func(learning_rate, max_depth, lambda_l1, lambda_l2, 

             bagging_fraction, bagging_freq, colsample_bytree, subsample_freq, feature_fraction):

    

    params = get_param(learning_rate, max_depth, lambda_l1, lambda_l2, 

             bagging_fraction, bagging_freq, colsample_bytree, subsample_freq, feature_fraction)

    acc, _ = train(params)

    return acc

    

def train(params):

    models = []

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=7)

    oof = np.zeros(len(train_X))



    for train_idx, test_idx in kfold.split(train_X, train_Y):

        X_train, y_train = train_X[train_idx], train_Y[train_idx]

        X_valid, y_valid = train_X[test_idx], train_Y[test_idx]



        y_train = y_train.reshape(-1)

        y_valid = y_valid.reshape(-1)        



        model = lgb.LGBMClassifier()

        model.set_params(**params)



        eval_set = [(X_valid, y_valid)]

        eval_names = ['valid']

        model.fit(X=X_train, y=y_train, eval_set=eval_set, eval_names=eval_names, verbose=0)

        

        pred = model.predict(X_valid).reshape(len(test_idx))

        oof[test_idx] = pred

        

        models.append(model)

    

    

    result = np.equal(oof, train_Y.reshape(-1))    

    accuracy = np.count_nonzero(result.astype(int)) / oof.shape[0]

    #print("accuracy : ", accuracy)    

    return accuracy, models



def get_optimized_hyperparameters():    

    bo_params = {'learning_rate' : (0.001, 0.1),

                   'max_depth': (10, 20),

                   'lambda_l1': (1, 10),

                   'lambda_l2': (1, 10),

            'bagging_fraction': (0.8, 1.0),

                'bagging_freq': (1, 10),

            'colsample_bytree': (0.7, 1.0),

              'subsample_freq': (1, 10),

            'feature_fraction': (0.9, 1.0)

            }

    

    optimizer = BayesianOptimization(opt_test_func, bo_params, random_state=1030)

    

    with warnings.catch_warnings():

        warnings.filterwarnings('ignore')

        init_points = 16

        n_iter = 16

        optimizer.maximize(init_points = init_points, n_iter = n_iter, acq='ucb', xi=0.0, alpha=1e-6)

        return optimizer.max['params']



def predict(models, test):

    preds = []

    for model in models:

        pred = model.predict(test).reshape(test.shape[0])

        preds.append(pred)

    

    preds = np.array(preds)

    preds = np.mean(preds, axis=0) > 0.5

    

    return preds



params = get_optimized_hyperparameters()

params = get_param(**params)

acc, models = train(params)

print("train accuracy : ", acc)



test_pred = predict(models,test)

print(test_pred.shape)

submission = pd.DataFrame({"PassengerId" : test_passenger_id, "Survived":test_pred.reshape(-1).astype(np.int)})





submission = pd.DataFrame({"PassengerId" : test_passenger_id, "Survived":test_pred.reshape(-1).astype(np.int)})

submission.to_csv('submission.csv', index=False)