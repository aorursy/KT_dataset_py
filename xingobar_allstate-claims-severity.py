# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import statsmodels.stats.api as sms

import seaborn as sns

from scipy.stats import skew

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
numeric_feat = []

for key,val in df_train.iloc[1,:].iteritems():

    if not str(val).isalpha():

        numeric_feat.append(key)

print(numeric_feat)
ids = numeric_feat.pop(0) ## delete id 

loss = numeric_feat.pop(len(numeric_feat) - 1) # delete loss

print(numeric_feat)
fig,ax = plt.subplots(figsize=(8,6))

sns.boxplot(df_train[numeric_feat])

ticks = plt.setp(ax.get_xticklabels(),rotation=45)
numeric_feat.append('loss')

correlation = df_train[numeric_feat].corr(method='pearson')

fig,ax = plt.subplots(figsize=(8,6))

sns.heatmap(correlation,annot=True,ax=ax,fmt='2.2f')

plt.ylabel('Column')

plt.xlabel('Column')

plt.title('Numeric feature correlation matrix')
fig,ax = plt.subplots(figsize=(8,6))

sns.distplot(df_train['loss'])

sns.boxplot(df_train['loss'])

plt.title('Loss')
plt.title('Log Loss')

sns.distplot(np.log1p(df_train['loss']))
cat_feat = []

for key,val in df_train.iloc[1,:].iteritems():

    if str(val).isalpha():

        cat_feat.append(key)

print(cat_feat)
numeric_feat.remove('loss')

skewed_list = []

for cn in df_train[numeric_feat].columns:

    skewed_list.append(skew(df_train[cn]))

    

plt.figure(figsize=(8,6))

plt.plot(skewed_list,'bo-')

plt.xlabel('continuous feature')

plt.ylabel('skewed')

plt.xticks(range(15),range(15,1))

plt.plot([0.25 for i in range(0,14)],'r--')

plt.text(6,.1,'threshold 0.25')

plt.show()

numeric_feat.append('loss')
skewed = df_train[numeric_feat].apply(lambda x:skew(x.dropna()))

skewed_less = skewed[skewed < 0.25].index

skewed_greater= skewed[skewed >= 0.25].index

df_train[skewed_greater] = np.log1p(df_train[skewed_greater])

df_train[skewed_less] = np.exp(df_train[skewed_less])



numeric_feat.remove('loss')

fig,ax = plt.subplots(figsize=(8,6))

sns.boxplot(df_train[numeric_feat],ax=ax)

ticks = plt.setp(ax.get_xticklabels(),rotation=45)

numeric_feat.append('loss')



del fig,ax,ticks
print('cont1 : ',sms.DescrStatsW(df_train['cont1']).tconfint_mean())

print('cont6 : ',sms.DescrStatsW(df_train['cont6']).tconfint_mean())

print('cont9 : ',sms.DescrStatsW(df_train['cont9']).tconfint_mean())

print('cont10 : ',sms.DescrStatsW(df_train['cont10']).tconfint_mean())

print('cont13 : ',sms.DescrStatsW(df_train['cont13']).tconfint_mean())

print('min : ' ,df_train[numeric_feat].min())

print

print('max : ', df_train[numeric_feat].max())
#numeric_feat.append('loss')

df_train_copy = df_train.copy()

df_train_copy['cont1'] = df_train_copy[df_train_copy['cont1'] >= 0.3930894470200948]['cont1']

df_train_copy['cont9'] = df_train_copy[df_train_copy['cont9'] >= 0.38823598271681442]['cont9']

df_train_copy['cont10'] = df_train_copy[df_train_copy['cont10'] >= 0.39602509335206326]['cont10']

fig,ax = plt.subplots(figsize=(8,6))

#numeric_feat.remove('loss')

sns.boxplot(df_train_copy[numeric_feat],ax=ax)

#numeric_feat.append('loss')

ticks = plt.setp(ax.get_xticklabels(),rotation=45)



del fig,ax,ticks