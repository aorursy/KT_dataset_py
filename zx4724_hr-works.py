# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')
df_HR = pd.read_csv('../input/HR_comma_sep.csv')

print(df_HR.columns)

print(df_HR['satisfaction_level'].describe())



sns.distplot(df_HR['satisfaction_level'])

print("Skewness: %f" % df_HR['satisfaction_level'].skew())

print("Kurtosis: %f" % df_HR['satisfaction_level'].kurt())

var = 'last_evaluation'

data = pd.concat([df_HR['satisfaction_level'], df_HR[var]], axis=1)

data.plot.scatter(x=var, y='satisfaction_level', ylim=(0,1))
cnt_top = df_HR.groupby("number_project")["satisfaction_level"].aggregate(np.mean).reset_index()

cnt_top = pd.DataFrame(cnt_top)

cnt_top=cnt_top.sort_values(by="satisfaction_level",ascending=False)

print(cnt_top)

plt.figure(figsize=(12,8))

sns.barplot(cnt_top.number_project, cnt_top.satisfaction_level, alpha=0.8)

plt.ylabel('satisfaction_level', fontsize=12)

plt.xlabel('number_project', fontsize=12)



plt.xticks(rotation='vertical')

plt.show()
var = 'average_montly_hours'

data = pd.concat([df_HR['satisfaction_level'], df_HR[var]], axis=1)

data.plot.scatter(x=var, y='satisfaction_level', ylim=(0,1))

cnt_top = df_HR.groupby("time_spend_company")["satisfaction_level"].aggregate(np.mean).reset_index()

cnt_top = pd.DataFrame(cnt_top)

cnt_top=cnt_top.sort_values(by="satisfaction_level",ascending=False)



plt.figure(figsize=(12,8))

sns.barplot(cnt_top.time_spend_company, cnt_top.satisfaction_level, alpha=0.8)

plt.ylabel('satisfaction_level', fontsize=12)

plt.xlabel('time_spend_company', fontsize=12)



plt.xticks(rotation='vertical')

plt.show()
cnt_top = df_HR.groupby("time_spend_company")["satisfaction_level"].aggregate(np.mean).reset_index()

cnt_top = pd.DataFrame(cnt_top)

cnt_top=cnt_top.sort_values(by="satisfaction_level",ascending=False)



plt.figure(figsize=(12,8))

sns.barplot(cnt_top.time_spend_company, cnt_top.satisfaction_level, alpha=0.8)

plt.ylabel('satisfaction_level', fontsize=12)

plt.xlabel('time_spend_company', fontsize=12)



plt.xticks(rotation='vertical')

plt.show()
cnt_top = df_HR.groupby("Work_accident")["satisfaction_level"].aggregate(np.mean).reset_index()

cnt_top = pd.DataFrame(cnt_top)

cnt_top=cnt_top.sort_values(by="satisfaction_level",ascending=False).head(10)

print(cnt_top.head(10))

plt.figure(figsize=(12,8))

sns.barplot(cnt_top.Work_accident, cnt_top.satisfaction_level, alpha=0.8)

plt.ylabel('satisfaction_level', fontsize=12)

plt.xlabel('Work_accident', fontsize=12)



plt.xticks(rotation='vertical')

plt.show()


cnt_top = df_HR.groupby("left")["satisfaction_level"].aggregate(np.mean).reset_index()

cnt_top = pd.DataFrame(cnt_top)

cnt_top=cnt_top.sort_values(by="satisfaction_level",ascending=False).head(10)

print(cnt_top.head(10))

plt.figure(figsize=(12,8))

sns.barplot(cnt_top.left, cnt_top.satisfaction_level, alpha=0.8)

plt.ylabel('satisfaction_level', fontsize=12)

plt.xlabel('left', fontsize=12)

cnt_top = df_HR.groupby("promotion_last_5years")["satisfaction_level"].aggregate(np.mean).reset_index()

cnt_top = pd.DataFrame(cnt_top)

cnt_top=cnt_top.sort_values(by="promotion_last_5years",ascending=False)

print(cnt_top)

plt.figure(figsize=(12,8))

sns.barplot(cnt_top.promotion_last_5years, cnt_top.satisfaction_level, alpha=0.8)

plt.ylabel('satisfaction_level', fontsize=12)

plt.xlabel('promotion_last_5years', fontsize=12)



plt.xticks(rotation='vertical')

plt.show()

cnt_top = df_HR.groupby("sales")["satisfaction_level"].aggregate(np.mean).reset_index()

cnt_top = pd.DataFrame(cnt_top)

cnt_top=cnt_top.sort_values(by="satisfaction_level",ascending=False).head(10)

print(cnt_top.head(10))

plt.figure(figsize=(12,8))

sns.barplot(cnt_top.sales, cnt_top.satisfaction_level, alpha=0.8)

plt.ylabel('satisfaction_level', fontsize=12)

plt.xlabel('sales', fontsize=12)



plt.xticks(rotation='vertical')

plt.show()

cnt_top = df_HR.groupby("salary")["satisfaction_level"].aggregate(np.mean).reset_index()

cnt_top = pd.DataFrame(cnt_top)

cnt_top=cnt_top.sort_values(by="satisfaction_level",ascending=False).head(10)

print(cnt_top.head(10))

plt.figure(figsize=(12,8))

sns.barplot(cnt_top.salary, cnt_top.satisfaction_level, alpha=0.8)

plt.ylabel('satisfaction_level', fontsize=12)

plt.xlabel('sales', fontsize=12)



plt.xticks(rotation='vertical')

plt.show()

corrmat = df_HR.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);

plt.show()
k = 5 #number of variables for heatmap

cols = corrmat.nlargest(k, 'satisfaction_level')['satisfaction_level'].index

cm = np.corrcoef(df_HR[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()