# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

#sns.set();

sns.set(rc={'figure.figsize':(10, 8)}); # you can change this if needed
df = pd.read_csv('../input/bigml_59c28831336c6604c800002a.csv', sep=',')
# Просмотр первых 10-и строк в виде транспонированной таблицы, в связи с большим количеством признаков.

df.head(10).T
df.info()
# Статистика по числовым признакам

df.describe().T
import seaborn as sns

sns.catplot(x = "churn", kind = "count", palette = "ch:.25", data = df, order = df['churn'].value_counts().index)
# Выведем процентное соотношение

df['churn'].value_counts(normalize=True)
# Сгрупируем данные по штатам, учитывая разные виды звонков (total day minutes, total eve minutes, total night minutes, total intl minutes).

# Добавим новый признак total minutes

df['total minutes'] = df['total day minutes'] + df['total eve minutes'] + df['total night minutes'] + df['total intl minutes']

# df.head().T

df.groupby('state')['total minutes'].sum(axis=1)
# Отобразим результаты в виде гистограммы

plt.figure(figsize=(15, 8)) # увеличим размер картинки



df.groupby('state')['total minutes'].sum().sort_values(ascending=False).plot(kind='bar', color='green')



plt.xlabel("State", size=12)

plt.ylabel("Total minutes", size=12)

#plt.show();
df.groupby('state')['total minutes'].sum(axis=1).idxmax()
df.groupby('state')['total minutes'].sum(axis=1).idxmin()
from scipy.stats import chi2_contingency, fisher_exact

chi2_contingency(pd.crosstab(df['total intl calls'], df['international plan']))
# Построим кросс-таблицу.

pd.crosstab(df['voice mail plan'], df['churn'])
sns.countplot(x='voice mail plan', hue='churn', data=df, order = df['voice mail plan'].value_counts().index);
sns.heatmap(pd.crosstab(df['voice mail plan'], df['churn']), cmap="YlGnBu", annot=True, cbar=False);
from scipy.stats import chi2_contingency, fisher_exact

chi2_contingency(pd.crosstab(df['voice mail plan'], df['churn']))
fisher_exact(pd.crosstab(df['voice mail plan'], df['churn']))
# Построим кросс-таблицу.

pd.crosstab(df['international plan'], df['churn'])
sns.countplot(x='international plan', hue='churn', data=df, order = df['international plan'].value_counts().index);
sns.heatmap(pd.crosstab(df['international plan'], df['churn']), cmap="YlGnBu", annot=True, cbar=False);
pd.crosstab(df['churn'], df['international plan'], margins=True)
chi2_contingency(pd.crosstab(df['international plan'], df['churn']))
fisher_exact(pd.crosstab(df['international plan'], df['churn']))
# Применяем бисериальный коэффициент корреляции.

from scipy.stats import pointbiserialr



pb2 = pointbiserialr(df['customer service calls'], df['churn'])

print('Point biserialr correlation:', pb2[0], 'p-value:', pb2[1])
plt.figure(figsize=(20, 20))

sns.countplot(x='customer service calls', hue='churn', data=df, order = df['customer service calls'].value_counts().index)
numeric = ['account length', 'number vmail messages', 'total day minutes', 

           'total day calls', 'total day charge', 'total eve minutes','total eve calls', 

           'total eve charge', 'total night minutes', 'total night calls', 'total night charge', 

           'total intl minutes', 'total intl calls', 'total intl charge', 'customer service calls']

sns.pairplot(df[numeric]);
# Построим матрицу корреляций Спирмена

df[numeric].corr(method='spearman')
# Визуализируем матрицы

sns.heatmap(df[numeric].corr(method='spearman'));
df[numeric].corr(method='pearson')
sns.heatmap(df[numeric].corr(method='pearson'));
from scipy.stats import pearsonr, spearmanr, kendalltau

r1 = pearsonr(df['account length'], df['customer service calls'])

print('Pearson correlation:', r1[0], 'p-value:', r1[1])
r2 = spearmanr(df['account length'], df['customer service calls'])

print('Spearman correlation:', r2[0], 'p-value:', r2[1])
r3 = kendalltau(df['account length'], df['customer service calls'])

print('Kendall correlation:', r3[0], 'p-value:', r3[1])
df['total calls'] = df['total day calls'] + df['total eve calls'] + df['total night calls'] + df['total intl calls']
df['total charge'] = df['total day charge'] + df['total eve charge'] + df['total night charge'] + df['total intl charge']
test = ['total calls', 'total minutes', 'total charge']

sns.pairplot(df[test]);
df[test].corr(method='spearman')
sns.heatmap(df[test].corr(method='spearman'));
df.groupby('churn')['account length'].mean().plot(kind='bar') 

plt.ylabel('account length') 

plt.show();
df.groupby('churn')['account length'].hist()

plt.xlabel('account length') 

plt.ylabel('number of subscribers')

plt.show();
# Применяем бисериальный коэффициент корреляции.



pb1 = pointbiserialr(df['account length'], df['churn'])

print('Point biserialr correlation:', pb1[0], 'p-value:', pb1[1]) 