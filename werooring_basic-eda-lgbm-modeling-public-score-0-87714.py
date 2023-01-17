# 데이터 분석 라이브러리

import numpy as np

import pandas as pd



# 시각화 라이브러리

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from plotnine import *



# 모델링 라이브러리

from category_encoders.ordinal import OrdinalEncoder

from sklearn.model_selection import KFold

from lightgbm import LGBMClassifier



# 기타 라이브러리

import random

import gc

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

sns.set_style("whitegrid")
train = pd.read_csv('/kaggle/input/kakr-4th-competition/train.csv')

test = pd.read_csv('/kaggle/input/kakr-4th-competition/test.csv')

sample_submission = pd.read_csv('/kaggle/input/kakr-4th-competition/sample_submission.csv')
train.shape, test.shape, sample_submission.shape
train.head()
test.head()
train.info()
train.describe()
train.describe(include='O')
print(f'income count: {train["income"].count()}')

print(f'income not null count(pct): {np.round(train["income"].count()/len(train)*100, 2)}%')
num_of_null = train.isnull().sum()

percent = (train.isnull().sum() / train.isnull().count() * 100)

pd.concat([num_of_null, percent], axis=1, keys=['# of null', 'Percent']).sort_values(by='Percent', ascending=False)
num_of_null = test.isnull().sum()

percent = (test.isnull().sum() / test.isnull().count() * 100)

pd.concat([num_of_null, percent], axis=1, keys=['# of null', 'Percent']).sort_values(by='Percent', ascending=False)
def get_min_max_avg(df, feature):

    print('Feature: ', feature)

    print('--------------------------------------')

    print('The max value is:',df[feature].max())

    print('The min value is:',df[feature].min())

    print('The average value is:',df[feature].mean())

    print('The median value is:',df[feature].median())
def plot_hist(df, feature, max_ylim, bins=10):

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.hist(df[feature], bins=bins)

    ax.set_ylim(0, max_ylim)

    ax.set_title(feature+' distribution (bins='+str(bins)+')')

    plt.show()
get_min_max_avg(train, 'age')
plot_hist(train, 'age', 4000, 15)
get_min_max_avg(train, 'fnlwgt')
plot_hist(train, 'fnlwgt', 13000, 20)
get_min_max_avg(train, 'education_num')
plot_hist(train, 'education_num', 9000, 16)
get_min_max_avg(train, 'capital_gain')
plot_hist(train, 'capital_gain', 25000, 20)
get_min_max_avg(train, 'capital_loss')
plot_hist(train, 'capital_loss', 25000, 20)
get_min_max_avg(train, 'hours_per_week')
plot_hist(train, 'hours_per_week', 15000, 10)
fig = plt.figure()



ggplot(train, aes(x='age', fill='income')) + geom_density(alpha=0.7) + ggtitle("The age distribution by income")
fig = plt.figure()

ggplot(train, aes(x='fnlwgt', fill='income')) + geom_density(alpha=0.7) + ggtitle('The final weight distribution by income')
fig = plt.figure()

ggplot(train, aes(x='education_num', fill='income')) + geom_density(alpha=0.7) + ggtitle('The education_num distribution by income')
train[train['income'] == '>50K']
fig = plt.figure()



ggplot(train[train['income'] == '>50K'], aes(x='education_num', fill='income')) + geom_density(alpha=0.7) + ggtitle('The education_num distribution by income')
sns.boxplot(x='income', y='education_num', data=train, palette="Set2", linewidth=2);
fig = plt.figure()

ggplot(train.loc[train['capital_gain'] > 0], aes(x='capital_gain', fill='income')) + geom_density(alpha=0.7) + ggtitle('The capital loss distribution by income')
sns.boxplot(x='income', y='capital_gain', data=train.loc[train['capital_gain'] > 0], palette="Set2", linewidth=2);
fig = plt.figure();

ggplot(train.loc[train['capital_loss'] > 0], aes(x='capital_loss', fill='income')) + geom_density(alpha=0.7) + ggtitle('The capital loss distribution by income')
sns.boxplot(x='income', y='capital_loss', data=train.loc[train['capital_loss'] > 0], palette="Set2", linewidth=2);
fig = plt.figure()

ggplot(train, aes(x='hours_per_week', fill='income')) + geom_density(alpha=0.7) + ggtitle('The hours per week distribution by income')
sns.boxplot(x='income', y='hours_per_week', data=train, palette="Set2", linewidth=2);
for col in train.columns:

    if train[col].dtype == 'object':

        all_categories = train[col].unique()

        print(f'Column "{col}" has {len(all_categories)} unique categroies')

        print('The categories are:', ', '.join(all_categories))

        print()
for col in train.columns:

    if train[col].dtype == 'object':

        categories = train[col].unique()

        print(f'The number of unique values in [{col}]: {len(categories)}')
def get_unique_values(df, feature):

    all_categories = train[feature].unique()

    print(f'Column "{feature}" has {len(all_categories)} unique categroies')

    print('------------------------------------------')

    print('\n'.join(all_categories))
get_unique_values(train, 'workclass')
fig, ax = plt.subplots(1, 1, figsize=(15, 7))

col = 'workclass'

value_counts = train[col].value_counts()

sns.countplot(x=col, data=train, palette="Set2", edgecolor='black', order = value_counts.index)



# bar 상단에 count 숫자로 입력

for i, v in value_counts.reset_index().iterrows():

    ax.text(i-0.1, v[col]+150 , v[col])
fig, ax = plt.subplots(1, 1, figsize=(15, 7))

value_counts = train[col].value_counts()

sns.countplot(x=col, hue='income', data=train, palette="Set2", edgecolor='black', order = value_counts.index);
get_unique_values(train, 'education')
fig, ax = plt.subplots(1, 1, figsize=(20, 7))

col = 'education'

value_counts = train[col].value_counts()

sns.countplot(x=col, data=train, palette="Set2", edgecolor='black', order = value_counts.index)

plt.xticks(rotation=45)



for i, v in value_counts.reset_index().iterrows():

    ax.text(i-0.1, v[col]+150 , v[col])
fig, ax = plt.subplots(1, 1, figsize=(15, 7))

value_counts = train[col].value_counts()

sns.countplot(x=col, hue='income', data=train, palette="Set2", edgecolor='black', order = value_counts.index);
train[train['education']=='Preschool']['income'].value_counts()
get_unique_values(train, 'marital_status')
fig, ax = plt.subplots(1, 1, figsize=(20, 7))

col = 'marital_status'

value_counts = train[col].value_counts()

sns.countplot(x=col, data=train, palette="Set2", edgecolor='black', order = value_counts.index)



for i, v in value_counts.reset_index().iterrows():

    ax.text(i-0.1, v[col]+150 , v[col])
fig, ax = plt.subplots(1, 1, figsize=(15, 7))

value_counts = train[col].value_counts()

sns.countplot(x=col, hue='income', data=train, palette="Set2", edgecolor='black', order = value_counts.index);
train.loc[train[col] == 'Married-AF-spouse', 'income']
get_unique_values(train, 'occupation')
fig, ax = plt.subplots(1, 1, figsize=(20, 7))

col = 'occupation'

value_counts = train[col].value_counts()

sns.countplot(x=col, data=train, palette="Set2", edgecolor='black', order = value_counts.index)

plt.xticks(rotation=45)



for i, v in value_counts.reset_index().iterrows():

    ax.text(i-0.12, v[col]+50 , v[col])
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

value_counts = train[col].value_counts()

sns.countplot(y=col, hue='income', data=train, palette="Set2", edgecolor='black', order = value_counts.index);
get_unique_values(train, 'relationship')
fig, ax = plt.subplots(1, 1, figsize=(15, 7))

col = 'relationship'

value_counts = train[col].value_counts()

sns.countplot(x=col, data=train, palette="Set2", edgecolor='black', order = value_counts.index)



for i, v in value_counts.reset_index().iterrows():

    ax.text(i-0.1, v[col]+150 , v[col])
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

value_counts = train[col].value_counts()

sns.countplot(x=col, hue='income', data=train, palette="Set2", edgecolor='black', order = value_counts.index);
get_unique_values(train, 'race')
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

col = 'race'

value_counts = train[col].value_counts()

sns.countplot(x=col, data=train, palette="Set2", edgecolor='black', order = value_counts.index)



for i, v in value_counts.reset_index().iterrows():

    ax.text(i-0.1, v[col]+150 , v[col])
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

value_counts = train[col].value_counts()

sns.countplot(x=col, hue='income', data=train, palette="Set2", edgecolor='black', order = value_counts.index);
get_unique_values(train, 'sex')
fig, ax = plt.subplots(1, 1, figsize=(7, 4))

col = 'sex'

value_counts = train[col].value_counts()

sns.countplot(x=col, data=train, palette="Set2", edgecolor='black', order = value_counts.index)



for i, v in value_counts.reset_index().iterrows():

    ax.text(i-0.05, v[col]+150 , v[col])
fig, ax = plt.subplots(1, 1, figsize=(7, 4))

value_counts = train[col].value_counts()

sns.countplot(x=col, hue='income', data=train, palette="Set2", edgecolor='black', order = value_counts.index);
train['native_country'].value_counts()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))

train_us = train[train['native_country']=='United-States']

col = 'native_country'

sns.countplot(x=col, hue='income', data=train_us, palette="Set2", edgecolor='black');
fig, ax = plt.subplots(1, 1, figsize=(15, 20))

train_other = train[train['native_country']!='United-States']

sns.countplot(y=col, hue='income', data=train_other, palette="Set2", edgecolor='black');
train.drop(['id'], axis=1, inplace=True)

test.drop(['id'], axis=1, inplace=True)
y = train['income'] != '<=50K'

X = train.drop(['income'], axis=1)
# 라벨 인코더 생성

LE_encoder = OrdinalEncoder(list(X.columns))



# train, test 데이터에 인코딩 적용

X = LE_encoder.fit_transform(X, y)

test = LE_encoder.transform(test)
NFOLDS = 5

folds = KFold(n_splits=NFOLDS)



columns = X.columns

splits = folds.split(X, y)

y_preds = np.zeros(test.shape[0])



feature_importances = pd.DataFrame()

feature_importances['feature'] = columns
model = LGBMClassifier(objective='binary', verbose=400, random_state=91)





for fold_n, (train_index, valid_index) in enumerate(splits):

    print('Fold: ', fold_n+1)

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]



    evals = [(X_train, y_train), (X_valid, y_valid)]

    model.fit(X_train, y_train, eval_metric='f1', eval_set=evals, verbose=True)

    

    feature_importances[f'fold_{fold_n + 1}'] = model.feature_importances_

        

    y_preds += model.predict(test).astype(int) / NFOLDS

    

    del X_train, X_valid, y_train, y_valid

    gc.collect()
feature_importances
sample_submission['prediction'] = y_preds



for ix, row in sample_submission.iterrows():

    if row['prediction'] > 0.5:

        sample_submission.loc[ix, 'prediction'] = 1

    else:

        sample_submission.loc[ix, 'prediction'] = 0



sample_submission = sample_submission.astype({"prediction": int})

sample_submission.to_csv('submission.csv', index=False)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

col = 'prediction'

value_counts = sample_submission[col].value_counts()

sns.countplot(x=col, data=sample_submission, palette="Set2", edgecolor='black', order = value_counts.index)



for i, v in value_counts.reset_index().iterrows():

    ax.text(i-0.05, v[col]+150 , v[col])