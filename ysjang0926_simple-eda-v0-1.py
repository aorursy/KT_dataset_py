import numpy as np # Linear algebra

import pandas as pd # Data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Visualization

import seaborn as sns # Visualization

import os # Load file
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df_train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv', parse_dates=['datetime'])

df_test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv', parse_dates=['datetime'])

df_submission = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv', parse_dates=['datetime'])
df_train.head(3)
print("1) train data shape : ", df_train.shape)

print("2) train data type : ")

print(df_train.dtypes)
df_test.head(3)
print("1) test data shape : ", df_test.shape)

print("2) test data type : ")

print(df_test.dtypes)
df_submission.head(3)
print("1) submission data shape : ", df_submission.shape)

print("2) submission data type : ")

print(df_submission.dtypes)
sum(df_train.duplicated()), sum(df_test.duplicated())
df_train.isnull().sum()
df_test.isnull().sum()
df_train['year'] = df_train['datetime'].dt.year # year

df_train['month'] = df_train['datetime'].dt.month # month

df_train['day'] = df_train['datetime'].dt.day # day

df_train['hour'] = df_train['datetime'].dt.hour # hour

df_train['minute'] = df_train['datetime'].dt.minute # minute

df_train['second'] = df_train['datetime'].dt.second # second
df_train.head(3)
plt.hist(df_train['count'], bins=15)

plt.xlabel('Count', fontsize = 14)

plt.xticks(fontsize = 14)

plt.yticks(fontsize = 14)
sns.boxplot(data=df_train, y='count')
figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

figure.set_size_inches(18, 10)



sns.boxplot(data=df_train, x='season', y='count', ax=ax1)

sns.boxplot(data=df_train, x='hour', y='count', ax=ax2)

sns.boxplot(data=df_train, x='workingday', y='count', ax=ax3)
figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)

figure.set_size_inches(18, 12)



sns.barplot(data=df_train, x="year", y="count", ax=ax1)

sns.barplot(data=df_train, x="month", y="count", ax=ax2)

sns.barplot(data=df_train, x="day", y="count", ax=ax3)

sns.barplot(data=df_train, x="hour", y="count", ax=ax4)

sns.barplot(data=df_train, x="minute", y="count", ax=ax5)

sns.barplot(data=df_train, x="second", y="count", ax=ax6)



ax1.set(title="Rental amounts by year")

ax2.set(title="Rental amounts by month")

ax3.set(title="Rental amounts by day")

ax4.set(title="Rental amounts by hour")

ax5.set(title="Rental amounts by minute")

ax6.set(title="Rental amounts by second")
df_train['dayofweek'] = df_train['datetime'].dt.dayofweek

df_train.head(3)
df_train['dayofweek'].value_counts()
fig,(ax1, ax2, ax3, ax4)= plt.subplots(nrows=4)

fig.set_size_inches(18,25)



sns.pointplot(data=df_train, x='hour', y='count', hue='workingday', ax=ax1)

sns.pointplot(data=df_train, x='hour', y='count', hue='dayofweek', ax=ax2)

sns.pointplot(data=df_train, x='hour', y='count', hue='season', ax=ax3)

sns.pointplot(data=df_train, x='hour', y='count', hue='weather', ax=ax4)