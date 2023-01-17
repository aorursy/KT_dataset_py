# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
with pd.HDFStore('../input/train.h5') as train:

    df = train.get('train')
print('Shape : {}'.format(df.shape))
derive_count = 0

fundamental_count = 0

technical_count = 0

for col in df.columns:

    if 'derived' in col:

        derive_count +=1

    elif 'fundamental' in col:

        fundamental_count +=1

    elif 'technical' in col:

        technical_count +=1

print('the number of derive : {} \nthe number of fundamental: {} \nthe number of technical : {}'.\

       format(derive_count,fundamental_count,technical_count))
print('Column dtypes')

print(df.dtypes.value_counts())
plt.title('Distribution of target variable')

df['y'].plot(kind='hist',bins=50)

plt.ylabel('Count')

plt.xlabel('Target Value')

print('min : {0:.3f} , max : {1:.3f} , mean : {2:.3f} ,std : {3:.3f}'.format(np.min(df['y']), np.max(df['y']) , np.mean(df['y']) , np.std(df['y']) ))
target = df['y'].values ## array

print('Target value less than -0.05  : {}'.format(len(target[target < -0.05])))

print('Target Value less than -0.05  : {} times' .format(round(len(target[target < -0.05]) / len(target),4) ))

print('Target valaue greater than 0.05 : {}'.format(len(target[target > 0.05])))

print('Target Value greater than 0.05  : {} times' .format(round(len(target[target > 0.05]) / len(target),4) ))
df.head()
df['timestamp'].plot(kind='hist',bins=50)

plt.xlabel('Timestamp')

plt.ylabel('Count')
plt.figure(figsize=(12,5))

y_bytimestamp = df.groupby('timestamp')['y'].agg('mean').reset_index()

y_bytimestamp.plot(x='timestamp',y='y')

plt.ylabel('mean of target')

plt.legend().set_visible(False) ## remove legend





y_bytimestamp.plot(kind='scatter',x='timestamp',y='y')

plt.ylabel('mean of target')

plt.xlabel('timestamp')
y_bytimestamp[:300].plot(x='timestamp',y='y',marker='^',mfc='red',markersize=3)

plt.ylabel('mean of target')

plt.legend().set_visible(False) ## remove legend





y_bytimestamp[:300].plot(kind='scatter',x='timestamp',y='y')

plt.ylabel('mean of target')
col = ['derived_0','derived_1','derived_2','derived_4','y']

corr = df[col].corr()

sns.heatmap(corr)

plt.xticks(rotation=90)
plt.figure(figsize=(12,6))

ax = plt.subplot(321)

plt.scatter(x = df['derived_0'],y=df['y'])

ax.set_title('derived_0 vs y')



ax = plt.subplot(322)

plt.scatter(x=df['derived_1'],y=df['y'])

ax.set_title('derived_1 vs y')



ax = plt.subplot(323)

plt.scatter(x=df['derived_2'],y=df['y'])

ax.set_title('derived_2 vs y')



ax = plt.subplot(324)

plt.scatter(x=df['derived_3'],y=df['y'])

ax.set_title('derived_3 vs y')





ax = plt.subplot(325)

plt.scatter(x=df['derived_4'],y=df['y'])

ax.set_title('derived_4 vs y')
col = ['fundamental_0','fundamental_1','fundamental_2','fundamental_3','fundamental_5',

 'fundamental_6', 'fundamental_7', 'fundamental_8', 'fundamental_9',

 'fundamental_10', 'fundamental_11', 'fundamental_12', 'fundamental_13',

 'fundamental_14', 'fundamental_15', 'fundamental_16', 'fundamental_17',

 'fundamental_18', 'fundamental_19', 'fundamental_20', 'fundamental_21',

 'fundamental_22', 'fundamental_23', 'fundamental_24',

 'fundamental_25', 'fundamental_26', 'fundamental_27', 'fundamental_28',

 'fundamental_29', 'fundamental_30', 'fundamental_31', 'fundamental_32',

 'fundamental_33', 'fundamental_34', 'fundamental_35', 'fundamental_36',

 'fundamental_37', 'fundamental_38', 'fundamental_39', 'fundamental_40',

 'fundamental_41', 'fundamental_42', 'fundamental_43', 'fundamental_44',

 'fundamental_45', 'fundamental_46', 'fundamental_47', 'fundamental_48',

 'fundamental_49', 'fundamental_50', 'fundamental_51', 'fundamental_52',

 'fundamental_53', 'fundamental_54', 'fundamental_55', 'fundamental_56',

 'fundamental_57', 'fundamental_58', 'fundamental_59', 'fundamental_60',

 'fundamental_61', 'fundamental_62', 'fundamental_63']
plt.figure(figsize=(12,6))

plt.subplot(221)

col_1 = col[:15]

col_1.append('y')

sns.heatmap(df[col_1].corr())



plt.subplot(222)

col_2 = col[15:31]

col_2.append('y')

sns.heatmap(df[col_2].corr())



plt.subplot(223)

col_3 = col[31:47]

col_3.append('y')

sns.heatmap(df[col_3].corr())



plt.subplot(224)

col_4 = col[47:]

col_4.append('y')

sns.heatmap(df[col_4].corr())
technical_col = [col for col in df.columns.tolist() if 'technical' in col]
col = ['fundamental_0','fundamental_1','fundamental_3','fundamental_5','technical_0',

 'technical_1','technical_2', 'technical_3', 'technical_5',

 'technical_6', 'technical_7', 'technical_9','technical_10','technical_11','technical_12']

sns.heatmap(df[col].corr())
from sklearn.preprocessing import StandardScaler
sns.distplot(df['technical_0'].dropna())
y_groupbytime = df[['timestamp','y']].groupby('timestamp').agg(['mean','std',len]).reset_index()
y_groupbytime.head()
timestamp = y_groupbytime['timestamp']

mean = y_groupbytime['y']['mean']

std = y_groupbytime['y']['std']

length = y_groupbytime['y']['len']





plt.figure(figsize=(8,6))

plt.plot(timestamp,mean,'.')

plt.xlabel('Timestamp')

plt.ylabel('mean of y')







plt.figure(figsize=(8,6))

plt.plot(timestamp,std,'.')

plt.xlabel('Timestamp')

plt.ylabel('std of y')



plt.figure(figsize=(8,6))

plt.plot(timestamp,length)

plt.xlabel('Timestamp')

plt.ylabel('length of y')
