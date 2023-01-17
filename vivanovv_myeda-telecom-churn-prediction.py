# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)}); # you can change this if needed
df = pd.read_csv('../input/bigml_59c28831336c6604c800002a.csv', sep = ',')

df.head(6)
df.info()
df['churn'].value_counts().plot(kind = 'bar', color = 'orange');
# Your code

df['churn'].value_counts(normalize = True)
df.head(6)
df['totalMinutes'] = df['total day minutes'] + df['total eve minutes'] + df['total night minutes'] + df['total intl minutes']

myTable = df.pivot_table(values = ['total day minutes', 'total eve minutes', 'total night minutes', 'total intl minutes', 'totalMinutes'], index = ['state'], aggfunc = 'sum')

myTable
myTable.plot.bar();

# plt.rcParams['figure.figsize'] = (50, 30) # раскомментировать для того, чтобы увеличить гравик снизу.
myTable[myTable['totalMinutes'] == myTable['totalMinutes'].max()]
myTable[myTable['totalMinutes'] == myTable['totalMinutes'].min()]
df.head(6)
# Your code

internationPlanCalls =  df.pivot_table(values = ['total intl calls'], index = ['international plan'], aggfunc = 'sum')

internationPlanCalls
internationPlanCalls.plot.bar(color = 'orange');
middleCalls = df['total intl calls'].mean()

print("Среднее количество звонков за границу: " + str(middleCalls))
df[df['total intl calls'] > df['total intl calls'].mean()].head(6)
plt.figure(figsize = (50, 10))

sns.countplot(y = 'total intl calls', hue = 'international plan', data = df[df['total intl calls'] > df['total intl calls'].mean()]);
internationalPlanMinutes = df.pivot_table(values = ['total intl minutes'], index = ['international plan'], aggfunc = 'sum')

internationalPlanMinutes
internationalPlanMinutes.plot.bar(color = "orange");
print("Средняя длина разговоров: " + str(df['total intl minutes'].mean()))
plt.figure(figsize = (50, 30))

sns.countplot(y = 'total intl minutes', hue = 'international plan', data = df[df['total intl minutes'] > df['total intl minutes'].mean()]);
df.head(6)
pd.crosstab(df['voice mail plan'], df['churn'])
from scipy.stats import chi2_contingency, fisher_exact

chi2_contingency(pd.crosstab(df['voice mail plan'], df['churn']))
fisher_exact(pd.crosstab(df['voice mail plan'], df['churn']))
pd.crosstab(df['international plan'], df['churn'])
chi2_contingency(pd.crosstab(df['international plan'], df['churn']))
fisher_exact(pd.crosstab(df['international plan'], df['churn']))
# Your code

callCentre = df.pivot_table(values = ['customer service calls'], index = ['churn'], aggfunc = 'sum')

callCentre
plt.figure(figsize=(40, 15))

sns.countplot(y='customer service calls', hue='churn', data=df);
from scipy.stats import pointbiserialr

pointbiserialr(df['customer service calls'], df['churn'])
# Your code

numeric = ['account length', 'number vmail messages', 'total day minutes', 'total day calls', 'total day charge', 'total eve minutes', 'total eve calls', 'total eve charge', 'total night minutes', 'total night calls', 'total night charge', 'total intl minutes', 'total intl calls', 'total intl charge', 'customer service calls']

sns.pairplot(df[numeric]);
sns.heatmap(df[numeric].corr(method='spearman'));
sns.heatmap(df[numeric].corr(method='pearson'));
sns.heatmap(df[numeric].corr(method='kendall'));
# Your code

from scipy.stats import pearsonr, spearmanr, kendalltau

pearsonr(df['account length'], df['customer service calls'])
kendalltau(df['account length'], df['customer service calls'])
totalMessages = df.pivot_table(values = ['number vmail messages'], index = ['state'], aggfunc = 'sum')

totalMessages.T
totalMessages[totalMessages['number vmail messages'] == totalMessages['number vmail messages'].max()]
totalMessages[totalMessages['number vmail messages'] == totalMessages['number vmail messages'].min()]
df.groupby('churn')['number vmail messages'].count()
df.groupby('churn')['number vmail messages'].hist();
pointbiserialr(df['number vmail messages'], df['churn'])
stateChurn = df.pivot_table(values = ['churn'], index = ['state'], aggfunc = 'sum')

stateChurn.T
stateChurn.plot.bar(color = 'orange');
stateChurn[stateChurn['churn'] == stateChurn['churn'].max()].T
stateChurn[stateChurn['churn'] == stateChurn['churn'].min()].T
df.groupby('churn')['account length'].hist()

plt.ylabel('users');

plt.xlabel('account length');
pearsonr(df['account length'], df['churn'])