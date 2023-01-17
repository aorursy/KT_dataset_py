import numpy as np 

import pandas as pd

import calendar

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline

plt.style.use('bmh')



from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

import statsmodels.api as sm

from sklearn.preprocessing import MinMaxScaler

from sklearn import model_selection

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV
sampleSubmission = pd.read_csv("../input/bike-sharing-demand/sampleSubmission.csv")

test_df = pd.read_csv("../input/bike-sharing-demand/test.csv")

df = pd.read_csv("../input/bike-sharing-demand/train.csv")
df.head()
def addfeatures(data):

    data["hour"] = [t.hour for t in pd.DatetimeIndex(data.datetime)]

    data["day"] = [t.dayofweek for t in pd.DatetimeIndex(data.datetime)]

    data["month"] = [t.month for t in pd.DatetimeIndex(data.datetime)]

    data['year'] = [t.year for t in pd.DatetimeIndex(data.datetime)]

    data['date'] = pd.to_datetime(data['datetime']).apply(lambda x: x.date())

    data["weekday"] = pd.to_datetime(data['datetime']).dt.dayofweek

    data['year'] = data['year'].map({2011:0, 2012:1})

    data.drop('datetime',axis=1,inplace=True)
addfeatures(df)
addfeatures(test_df)
df.info()
fig, (ax1) = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=True, figsize = (14, 10))

df.groupby('date').mean()['count'].plot(ax =ax1, title='Bike Rent Count per Date')

plt.xlabel('Date')

plt.ylabel('Count');
fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=True, figsize = (14, 10))

# rent per day per hour 

df.groupby(['weekday','hour']).mean()['count'].unstack('weekday').plot( ax=ax1, title='Bike Rent Count per day per hour' )



# rent per season per hour

df.groupby(['season', 'hour']).mean()['count'].unstack('season').rename(columns={1:'springer', 2:'summer', 3:'fall', 4:'winter'}).plot( ax=ax2, title = 'Bike Rent Count per season per hour')

# Set common labels

fig.text(0.5, 0.04, 'Hour of the Day', ha='center', va='center', fontsize = 14)

fig.text(0.06, 0.5, 'Rental Counts', ha='center', va='center', rotation='vertical', fontsize = 14);
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, sharey=True, figsize = (14, 6))

sns.regplot(x="atemp", y="count", data=df, ax=ax1)

sns.regplot(x="temp", y="count", data=df, ax=ax2)

sns.regplot(x="windspeed", y="count", data=df, ax=ax3)

sns.regplot(x="humidity", y="count", data=df, ax=ax4);
# Windspeed 

print('Number of rows with missing Windspeed: ', df[df.windspeed ==0].shape[0])
#Replace nan windspeed with last non-zero digit.

df.windspeed = df.windspeed.replace(to_replace=0, method='ffill')
# Humidity

print('Number of rows with missing Humidity: ', df[df.humidity ==0].shape[0])
# all of 0 humidity in the data comes from the month of march 2011

df[df.humidity ==0]
march_mean = df[(df.year == 1) & (df.month ==3)]['humidity'].mean()
# replace 0 hum with march mean 2012

df.humidity = df.humidity.map( lambda x : march_mean if x == 0 else x)
plt.figure(figsize=(10,6))



# plot count per working day season wise 

labels=['springer', 'summer', 'fall', 'winter']



ax = sns.barplot(data=df, x='workingday',y='count', hue='season' )



h, l = ax.get_legend_handles_labels()

ax.legend(h, labels, title="Season", loc='upper left');
plt.figure(figsize=(8,6))



year = [2011, 2012]

ax = sns.boxplot(x="workingday", y="count", hue="year", data=df, palette="Set1", );

h, l = ax.get_legend_handles_labels()

ax.legend(h, year, title="Year", loc='upper left');
# Scale features 

scaler = MinMaxScaler()

col2scale = ['humidity', 'temp', 'windspeed']

for i in col2scale:

    df[i] = scaler.fit_transform(df[i].values.reshape(-1,1))
noncat = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
# Correlation plot 

cor = df.corr()

plt.figure(figsize=(14,10))



mask = np.zeros_like(cor, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



ax = sns.heatmap(cor, mask = mask, annot=True, cmap="YlGnBu")
# Drop atemp

df.drop(['atemp'], axis=1, inplace = True)
test_df.drop(['atemp'], axis=1, inplace = True)
df['checkh'] = df.casual + df.registered
print('Number of rows where sum of causal and registered is equal to rental count:', sum(df['count'] == df.checkh))

print('Total Rows:', df.shape[0])
df = df.drop(['casual', 'registered'], axis=1)
df['count'].plot(kind = 'kde');
df['cnt_log'] = np.log(df['count'])
# check the skewness after log transformation 

df.cnt_log.plot(kind = 'kde');
# remove outlier on count column 

#df = df[df.cnt.between(df.cnt.quantile(.05), df.cnt.quantile(.95))]
fig, (ax1, ax2) = plt.subplots(1, 2)

fig.suptitle('Count vs Log Transformed Count')

ax1.set_xlabel('Count')

ax1.hist(df['count'])



ax2.set_xlabel('Log of Count')

ax2.hist(df.cnt_log);
# Droping date and checkh variable (created to verify the resistered and casual) 

df.drop(['date','checkh'], axis=1, inplace=True)

df.head()
df.columns
cat = ['season', 'year', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather']

for i in cat:

    df[i] = df[i].astype("category")
dfdummy = pd.get_dummies(df, columns=cat, drop_first=True)
features = dfdummy[[i for i in list(dfdummy.columns) if i not in ['count', 'cnt_log']]].columns
#Creating Test and train dataset

X = dfdummy.drop(['count', 'cnt_log'], axis=1).values

y = dfdummy['count'].values

yl = dfdummy.cnt_log.values
metric = pd.DataFrame(columns = ['r2', 'rmse'])

r2 = []

rms = []

def split_train_test(x,y):

    # get train test split

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)

    # linear regression 

    lModel = LinearRegression()

    print('\nTraining Linear Reggresor on Train data....')

    result = lModel.fit(X_train, y_train)

    r2.append(round((result.score(X_test,y_test)),2))

    rms.append(round((sqrt(mean_squared_error(y_test, result.predict(X_test)))),2))

    print('Done!!!')

    print('\nTraining Random Forrest Reggresor on Train data....')

    # Random forrest 

    regr = RandomForestRegressor(n_estimators=300)

    regr.fit(X_train, y_train)

    r2.append(round((regr.score(X_test, y_test)),2))

    rms.append(round((sqrt(mean_squared_error(y_test, regr.predict(X_test)))),2))

    print('Done!!!')
seed = 123

target = [y, yl]





for i in target:

    if i is y:

        print('\nTarget is Count')

    else:

        print('\nTarget is Log of Count')

    split_train_test(X,i)
metric.r2 = r2

metric.rmse= rms

metric['Target-Model'] = ['LM_count', 'RF_Count', 'LM_Count_log', 'RF_Count_log']
g = sns.barplot(x = 'Target-Model', y = 'r2', data = metric)

ax=g

#annotate axis = seaborn axis

for p in ax.patches:

             ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),

                 ha='center', va='center', fontsize=11, color='Blue', xytext=(0, 8),

                 textcoords='offset points')

_ = g.set_ylim(0,1) 
g = sns.barplot(x = 'Target-Model', y = 'rmse', data = metric)

ax=g

#annotate axis = seaborn axis

for p in ax.patches:

             ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),

                 ha='center', va='center', fontsize=11, color='Blue', xytext=(0, 8),

                 textcoords='offset points')

_ = g.set_ylim(0,109) 
# Cross validation



kfold = model_selection.KFold(n_splits=10, random_state=100)

model_kfold = LinearRegression()

results_kfold = model_selection.cross_val_score(model_kfold, X, yl, cv=kfold, scoring = 'r2')

print("R2 score: ",round(results_kfold.mean(),2))
from sklearn.metrics import mean_squared_log_error



kfold = model_selection.KFold(n_splits=5, random_state=100)

model_kfold = RandomForestRegressor(n_estimators=50)

results_kfold = model_selection.cross_val_score(model_kfold, X, yl, cv=kfold, scoring = 'r2')

print("R2 score: ",round(results_kfold.mean(),2))
X_train, X_test, y_train, y_test = train_test_split(X, yl, test_size=0.3, random_state=123)

# Random forrest 

regr = RandomForestRegressor(n_estimators=300)

regr.fit(X_train, y_train)

print("Root Mean Squared Logarithmic Error: ",sqrt(mean_squared_log_error(y_test, regr.predict(X_test))))
importances = regr.feature_importances_

std = np.std([tree.feature_importances_ for tree in regr.estimators_], axis=0)

indices = np.argsort(importances)[::-1]
# Plot the feature importances of the forest

plt.figure(figsize=(25,5))

plt.title("Feature importances")

plt.bar(range(X_test.shape[1]), importances[indices], yerr=std[indices], align="center")

plt.xticks(range(X_test.shape[1]), [features[i] for i in indices], rotation=90)

plt.xlim([-1, X_test.shape[1]])

plt.show()
sns.scatterplot(x=regr.predict(X_test), y=(y_test-regr.predict(X_test)))
# visualize subset of Test count and predicted test coutn 

plt.figure(figsize=(16,5))

plt.plot(regr.predict(X_test)[200:400],'r')

plt.plot(y_test[200:400])