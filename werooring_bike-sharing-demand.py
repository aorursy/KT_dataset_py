import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os



%matplotlib inline



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# parse_dates는 해당 column의 type을 datetime으로 만들어 줌

df_train = pd.read_csv('/kaggle/input/train.csv', parse_dates=['datetime'])

df_test = pd.read_csv('/kaggle/input/test.csv', parse_dates=['datetime'])

df_submission = pd.read_csv('/kaggle/input/sampleSubmission.csv', parse_dates=['datetime'])
df_train.head(3)
df_test.head(3)
df_submission.head(3)
print(df_train.shape, df_test.shape)

print("훈련 데이터")

print(df_train.dtypes)

print("테스트 데이터")

print(df_test.dtypes)
sum(df_train.duplicated()), sum(df_test.duplicated())
df_train.isnull().sum()
df_test.isnull().sum()
df_train['year'] = df_train['datetime'].dt.year

df_train['month'] = df_train['datetime'].dt.month

df_train['day'] = df_train['datetime'].dt.day

df_train['hour'] = df_train['datetime'].dt.hour

df_train['minute'] = df_train['datetime'].dt.minute

df_train['second'] = df_train['datetime'].dt.second
df_train.head()
figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)

figure.set_size_inches(18, 10)



sns.barplot(data=df_train, x="year", y="count", ax=ax1)

sns.barplot(data=df_train, x="month", y="count", ax=ax2)

sns.barplot(data=df_train, x="day", y="count", ax=ax3)

sns.barplot(data=df_train, x="hour", y="count", ax=ax4)

sns.barplot(data=df_train, x="minute", y="count", ax=ax5)

sns.barplot(data=df_train, x="second", y="count", ax=ax6)



ax1.set(title="Rental amounts by year")

ax2.set(title="Rental amounts by month")

ax3.set(title="Rental amounts by day")

ax4.set(title="Rental amounts by hour");
figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

figure.set_size_inches(18, 10)



sns.boxplot(data=df_train, y='count', ax=ax1)

sns.boxplot(data=df_train, x='season', y='count', ax=ax2)

sns.boxplot(data=df_train, x='hour', y='count', ax=ax3)

sns.boxplot(data=df_train, x='workingday', y='count', ax=ax4)

# The day of week with Monday=0, Sunday=6

df_train['dayofweek'] = df_train['datetime'].dt.dayofweek

df_train.head(2)
df_train['dayofweek'].value_counts()
fig,(ax1, ax2, ax3, ax4, ax5)= plt.subplots(nrows=5)

fig.set_size_inches(18,25)



sns.pointplot(data=df_train, x='hour', y='count', ax=ax1)

sns.pointplot(data=df_train, x='hour', y='count', hue='workingday', ax=ax2)

sns.pointplot(data=df_train, x='hour', y='count', hue='dayofweek', ax=ax3)

sns.pointplot(data=df_train, x='hour', y='count', hue='season', ax=ax4)

sns.pointplot(data=df_train, x='hour', y='count', hue='weather', ax=ax5);
g = sns.PairGrid(data=df_train, vars=['temp', 'atemp', 'casual', 'registered', 'humidity', 'windspeed', 'count'])

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter)
corr_matrix = df_train[['temp', 'atemp', 'casual', 'registered', 'humidity', 'windspeed', 'count']]

corr_matrix = corr_matrix.corr()

corr_matrix
mask = np.array(corr_matrix)

print(mask)

print(np.tril_indices_from(mask))

mask[np.tril_indices_from(mask)] = False

mask
sns.set(rc={'figure.figsize':(7,6)})

sns.heatmap(corr_matrix, mask=mask, annot=True);
fig,(ax1,ax2,ax3) = plt.subplots(ncols=3)

fig.set_size_inches(12, 5)

sns.regplot(x="temp", y="count", data=df_train,ax=ax1)

sns.regplot(x="windspeed", y="count", data=df_train,ax=ax2)

sns.regplot(x="humidity", y="count", data=df_train,ax=ax3);
def concat_year_month(datetime):

    return '{0}-{1}'.format(datetime.year, datetime.month)



df_train['year_month'] = df_train['datetime'].apply(concat_year_month)



df_train[['datetime', 'year_month']].head()
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

fig.set_size_inches(18,4)

sns.barplot(data=df_train, x='year', y='count', ax=ax1)

sns.barplot(data=df_train, x='month', y='count', ax=ax2)



fig, ax3 = plt.subplots(nrows=1, ncols=1)

fig.set_size_inches(18,4)

sns.barplot(data=df_train, x='year_month', y='count', ax=ax3);
df_train_without_outliers = df_train[df_train['count'] - df_train['count'].mean() < 3*df_train['count'].std()]



print(df_train.shape)

print(df_train_without_outliers.shape)
figure, axes = plt.subplots(nrows=1, ncols=2)

figure.set_size_inches(18,4)



sns.distplot(df_train_without_outliers['count'], ax=axes[0]);

sns.distplot(np.log(df_train_without_outliers['count']), ax=axes[1]);