# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model as lm

import kagglegym



%matplotlib inline
# Create environment

env = kagglegym.make()



# Get first observation

observation = env.reset()



# Get the train dataframe

train = observation.train
train.shape
mean_values = train.mean(axis=0)

train.fillna(mean_values, inplace=True)

train.head()
# Now let us look at the correlation coefficient of each of these variables #

x_cols = [col for col in train.columns if col not in ['id','timestamp','y']]



labels = []

values = []

for col in x_cols:

    labels.append(col)

    values.append(np.corrcoef(train[col].values, train.y.values)[0,1])

    

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots(figsize=(12,40))

rects = ax.barh(ind, np.array(values), color='y')

ax.set_yticks(ind+((width)/2.))

ax.set_yticklabels(labels, rotation='horizontal')

ax.set_xlabel("Correlation coefficient")

ax.set_title("Correlation coefficient")

#autolabel(rects)

plt.show()
a = [abs(x) for x in values]

#np.sort(a)[::-1]
cols_to_use = []

for  i in np.argsort(a)[::-1][:20]:

    cols_to_use.append(labels[i])
cols_to_use
#cols_to_use = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']



temp_df = train[cols_to_use]

corrmat = temp_df.corr()

f, ax = plt.subplots(figsize=(8, 8))



# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=.8, square=True)

plt.show()
models_dict = {}

for col in cols_to_use:

    model = lm.LinearRegression()

    model.fit(np.array(train[col].values).reshape(-1,1), train.y.values)

    models_dict[col] = model
col = 'technical_30'

model = models_dict[col]

while True:

    observation.features.fillna(mean_values, inplace=True)

    test_x = np.array(observation.features[col].values).reshape(-1,1)

    observation.target.y = model.predict(test_x)

    #observation.target.fillna(0, inplace=True)

    target = observation.target

    timestamp = observation.features["timestamp"][0]

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

        

    observation, reward, done, info = env.step(target)

    if done:

        break

info
# Get first observation

observation = env.reset()



col = 'technical_20'

model = models_dict[col]

while True:

    observation.features.fillna(mean_values, inplace=True)

    test_x = np.array(observation.features[col].values).reshape(-1,1)

    observation.target.y = model.predict(test_x)

    #observation.target.fillna(0, inplace=True)

    target = observation.target

    timestamp = observation.features["timestamp"][0]

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

        

    observation, reward, done, info = env.step(target)

    if done:

        break

info
# Get first observation

observation = env.reset()



col = 'fundamental_11'

model = models_dict[col]

while True:

    observation.features.fillna(mean_values, inplace=True)

    test_x = np.array(observation.features[col].values).reshape(-1,1)

    observation.target.y = model.predict(test_x)

    #observation.target.fillna(0, inplace=True)

    target = observation.target

    timestamp = observation.features["timestamp"][0]

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

        

    observation, reward, done, info = env.step(target)

    if done:

        break

info
# Get first observation

observation = env.reset()



col = 'technical_19'

model = models_dict[col]

while True:

    observation.features.fillna(mean_values, inplace=True)

    test_x = np.array(observation.features[col].values).reshape(-1,1)

    observation.target.y = model.predict(test_x)

    #observation.target.fillna(0, inplace=True)

    target = observation.target

    timestamp = observation.features["timestamp"][0]

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

        

    observation, reward, done, info = env.step(target)

    if done:

        break

info
cols_to_use = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']



# Get first observation

env = kagglegym.make()

observation = env.reset()

train = observation.train

train.fillna(mean_values, inplace=True)



model = lm.LinearRegression()

model.fit(np.array(train[cols_to_use]), train.y.values)



while True:

    observation.features.fillna(mean_values, inplace=True)

    test_x = np.array(observation.features[cols_to_use])

    observation.target.y = model.predict(test_x)

    target = observation.target

    timestamp = observation.features["timestamp"][0]

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

        

    observation, reward, done, info = env.step(target)

    if done:

        break

info