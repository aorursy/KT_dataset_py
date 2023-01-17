import numpy as np

import pandas as pd
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
##display the first five rows of the train dataset.

train.head(5)
##display the first five rows of the test dataset.

test.head(5)
##display shapes of datasets

train.shape, test.shape
import seaborn as sns

import matplotlib.pyplot as plt



corrmat = train.corr()

f, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corrmat, square=True, annot=True)
var = 'Year_of_Release'

target = 'JP_Sales'



data = pd.concat([train[target], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y=target, data=data)

fig.axis(ymin=0, ymax=max(train[target]))

plt.xticks(rotation=90)
from scipy import stats



arr = np.log1p(train['JP_Sales'])

sns.distplot(arr, fit=stats.norm)

fig = plt.figure()

stats.probplot(arr, plot=plt);
from sklearn import preprocessing as prep



target = 'JP_Sales'

id_col = 'Id'



def trans(y):

    return np.log1p(y)



def inv_trans(y):

    return np.exp(y) - 1



def replace_na_on_mean(data, column):

    return all_data.replace({column: pd.NaT}, 

                            all_data[column].dropna().mean())



def str_to_int(data, column):

    enc = prep.LabelEncoder()

    return enc.fit_transform(data[column])



def str_to_columns(data, column):

    for value in set(data[column]):

        data[f"{column}_{value}"] = pd.Series(1 if el == value else 0 for el in data[column])

    data = data.drop(columns=column)

    return data
y = trans(train[target])

train = train.drop(columns=target)

Id = test[id_col]

test = test.drop(columns=id_col)



all_data = pd.concat((train, test), sort=False).reset_index(drop=True)
all_data_na = (all_data.isnull().mean()) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data
all_data = all_data.drop(columns='Name')
all_data = all_data.drop(columns='User_Count')

all_data = replace_na_on_mean(all_data, 'Critic_Count')
all_data = replace_na_on_mean(all_data, 'User_Score')

all_data = replace_na_on_mean(all_data, 'Critic_Score')



all_data.head()
all_data['Total_Score'] = (all_data['User_Score'] + all_data['Critic_Score']/10) / 2
all_data = all_data.replace({'Developer': pd.NaT}, 'None')

all_data = all_data.replace({'Publisher': pd.NaT}, 'None')



all_data['Developer'] = str_to_int(all_data, 'Developer')

all_data['Publisher'] = str_to_int(all_data, 'Publisher')
all_data = replace_na_on_mean(all_data, 'Year_of_Release')
all_data = str_to_columns(all_data, 'Rating')

all_data = str_to_columns(all_data, 'Platform')

all_data = str_to_columns(all_data, 'Genre')
train, test = all_data[:train.shape[0]], all_data[train.shape[0]:]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.33, random_state=42)
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(subsample=0.9, loss='lad',

                                 n_estimators=300, learning_rate=0.27,

                                 max_depth=7, max_features='sqrt')

model.fit(X_train, y_train)
def error(y_true, y_pred):

    return np.mean(np.abs(np.log1p(y_true) - np.log1p(y_pred)))



print(f"On train: {error(inv_trans(y_train), inv_trans(model.predict(X_train)))}")

print(f"On test: {error(inv_trans(y_test), inv_trans(model.predict(X_test)))}")
class ManyModels(object):

    def __init__(self, models):

        self.models = models

        

    def fit(self, X, y):

        for model in self.models:

            model.fit(X, y)

            

    def predict(self, X):

        preds = [model.predict(X) for model in self.models]

        return sum(preds) / len(preds)
models = ManyModels([GradientBoostingRegressor(subsample=0.9, loss='lad',

                                               n_estimators=300, learning_rate=0.27,

                                               max_depth=7, max_features='sqrt')

                     for _ in range(10)])



models.fit(X_train, y_train)
print(f"On train: {error(inv_trans(y_train), inv_trans(models.predict(X_train)))}")

print(f"On test: {error(inv_trans(y_test), inv_trans(models.predict(X_test)))}")
models = ManyModels([GradientBoostingRegressor(subsample=0.9, loss='lad',

                                               n_estimators=300, learning_rate=0.27,

                                               max_depth=7, max_features='sqrt')

                     for _ in range(25)])



models.fit(train, y)
res = pd.DataFrame({id_col: Id, 

                    target: inv_trans(models.predict(test))})

res.to_csv("res.csv", index=False)