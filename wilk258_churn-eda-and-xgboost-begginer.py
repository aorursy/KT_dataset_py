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
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.tsa.stattools as stattools
df = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')
df.head(3)
fig, ax = plt.subplots(4, 1,figsize=(10,20))
sns.countplot(x='Geography', data=df, ax=ax[0])
sns.countplot(x='Gender', data=df, ax= ax[1])
sns.countplot(y='Tenure', data=df,ax= ax[2])
sns.distplot(df.Age, ax=ax[3])
fig, ax = plt.subplots(4, 1,figsize=(10,20))
sns.countplot(x='NumOfProducts', data=df, ax=ax[0])
sns.countplot(x='HasCrCard', data=df, ax= ax[1])
sns.countplot(x='IsActiveMember', data=df,ax= ax[2])
sns.countplot(x='Exited', data=df,ax= ax[3])
df.shape
x_var = 'EstimatedSalary'
groupby_var = 'Geography'
df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [df[x_var].values.tolist() for i, df in df_agg]

# Draw
plt.figure(figsize=(16,9), dpi= 80)
colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, 30, stacked=True, density=False, color=colors[:len(vals)])

# Decoration
plt.legend({group:col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
plt.xlabel(x_var)
plt.ylabel("Frequency")
plt.ylim(0, 400)
x_var = 'Balance'
groupby_var = 'Geography'
df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [df[x_var].values.tolist() for i, df in df_agg]

# Draw
plt.figure(figsize=(16,9), dpi= 80)
colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, 30, stacked=True, density=False, color=colors[:len(vals)])

# Decoration
plt.legend({group:col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
plt.xlabel(x_var)
plt.ylabel("Frequency")
plt.ylim(0, 800)
x_var = 'CreditScore'
groupby_var = 'Geography'
df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [df[x_var].values.tolist() for i, df in df_agg]

# Draw
plt.figure(figsize=(16,9), dpi= 80)
colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, 30, stacked=True, density=False, color=colors[:len(vals)])

# Decoration
plt.legend({group:col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
plt.xlabel(x_var)
plt.ylabel("Frequency")
plt.ylim(0, 800)
x_var = 'CreditScore'
groupby_var = 'Gender'
df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [df[x_var].values.tolist() for i, df in df_agg]

# Draw
plt.figure(figsize=(16,9), dpi= 80)
colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, 30, stacked=True, density=False, color=colors[:len(vals)])

# Decoration
plt.legend({group:col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
plt.xlabel(x_var)
plt.ylabel("Frequency")
plt.ylim(0, 800)
x_var = 'EstimatedSalary'
groupby_var = 'Gender'
df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [df[x_var].values.tolist() for i, df in df_agg]

# Draw
plt.figure(figsize=(16,9), dpi= 80)
colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, 30, stacked=True, density=False, color=colors[:len(vals)])

# Decoration
plt.legend({group:col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
plt.xlabel(x_var)
plt.ylabel("Frequency")
plt.ylim(0, 400)
x_var = 'Balance'
groupby_var = 'Gender'
df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [df[x_var].values.tolist() for i, df in df_agg]

# Draw
plt.figure(figsize=(16,9), dpi= 80)
colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, 30, stacked=True, density=False, color=colors[:len(vals)])

# Decoration
plt.legend({group:col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
plt.xlabel(x_var)
plt.ylabel("Frequency")
plt.ylim(0, 800)
country = df[['EstimatedSalary', 'Geography']].groupby('Geography').apply(lambda x: x.mean())
country.sort_values('EstimatedSalary', inplace=True)
country.reset_index(inplace=True)

fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.hlines(y=country.index, xmin=11, xmax=26, color='gray', alpha=0.7, linewidth=1, linestyles='dashdot')
ax.scatter(y=country.index, x=country.EstimatedSalary, s=75, color='firebrick', alpha=0.7)

ax.set_title('Dot Plot for Country Average  EstimatedSalary', fontdict={'size':22})
ax.set_xlabel('EstimatedSalary')
ax.set_yticks(country.index)
ax.set_yticklabels(country.Geography.str.title(), fontdict={'horizontalalignment': 'right'})
ax.set_xlim(90000, 110000)
plt.show()
balan = df[['Balance', 'Geography']].groupby('Geography').apply(lambda x: x.mean())
balan.sort_values('Balance', inplace=True)
balan.reset_index(inplace=True)

fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.hlines(y=balan.index, xmin=11, xmax=26, color='gray', alpha=0.7, linewidth=1, linestyles='dashdot')
ax.scatter(y=balan.index, x=balan.Balance, s=75, color='firebrick', alpha=0.7)

ax.set_title('Dot Plot for Country Average  Balance', fontdict={'size':22})
ax.set_xlabel('Balance')
ax.set_yticks(balan.index)
ax.set_yticklabels(balan.Geography.str.title(), fontdict={'horizontalalignment': 'right'})
ax.set_xlim(60000, 125000)
plt.show()
df.head(3)
plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
df.drop(['CustomerId', 'Surname'], axis = 1, inplace = True)
X = df.iloc[:, :-1]
y = df.iloc[:, -1].astype('float')
X.head()
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 40)
len(y_train), len(y_val)
X_train.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)
y_val.reset_index(drop=True, inplace=True)
categorical_cols = [col for col in df.select_dtypes(exclude='number').columns]
categorical_cols
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

X_train_categorical, X_val_categorical = pd.DataFrame(), pd.DataFrame()

for col in categorical_cols:
    X_train_categorical[col] = label_encoder.fit_transform(X_train[col])
    X_val_categorical[col] = label_encoder.transform(X_val[col])

X_train.drop(categorical_cols, axis = 1, inplace = True)
X_val.drop(categorical_cols, axis = 1, inplace=True)

X_train = X_train.join(X_train_categorical)
X_val = X_val.join(X_val_categorical)
import xgboost as xgb
from sklearn.metrics import mean_squared_error
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_val, label=y_val)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
model1 = xgb.cv(params, d_train,  num_boost_round=500, early_stopping_rounds=100)
model = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
model1.loc[30:,["train-logloss-mean", "test-logloss-mean"]].plot()
xgb.plot_importance(model)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()