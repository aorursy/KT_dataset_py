import kagglegym

import numpy as np

import pandas as pd

import xgboost as xgb

import matplotlib.pyplot as plt

import seaborn as sns

# The "environment" is our interface for code competitions

env = kagglegym.make()



# We get our initial observation by calling "reset"

observation = env.reset()



# Note that the first observation we get has a "train" dataframe

print("Train has {} rows".format(len(observation.train)))



# The "target" dataframe is a template for what we need to predict:

print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))
train_data = observation.train

train_data = train_data.set_index(['id','timestamp']).sort_index()

train_data
train_data.dtypes
f_cols=[f for f in train_data.columns if f.startswith('fundamental')]

t_cols=[t for t in train_data.columns if t.startswith('technical')]

ID=train_data.index.levels[0]



print('The number of fundamental columns is {}.'.format(len(f_cols)))

print('The number of technical columns is {}.'.format(len(t_cols)))

print('The number of unique ids is {}.'.format(len(ID)))
fig, axs = plt.subplots(5,2)

font = {'weight' : 'normal',

        'size'   : 10}

plt.rc('font', **font)



rn=ID[np.random.randint(0,len(ID)-1,10)]



for i in range(0,len(rn)):

    ax = plt.subplot(5,2,i+1)

    ax.plot(train_data.loc[rn[i]].index, 

            train_data.y.loc[rn[i]],

            label='ID={}'.format(rn[i]))

    plt.legend()

    if i in [8,9]:

        ax.set_xlabel('Time Stamp')

    if i in range(0,9,2):

        ax.set_ylabel('y')    
from statsmodels.tsa.stattools import adfuller

print ('Results of Dickey-Fuller Test:')

df=pd.DataFrame([])

for i in rn:

    t = adfuller(train_data.y.loc[i],

                 autolag='AIC')

    d={'teststat': t[0],

       'pval': t[1],

       'nlags': t[2],

       'nobs': t[3],

       '1%crit': t[4]['1%'],

       '5%crit': t[4]['5%'],

       '10%crit': t[4]['10%'],

       'Nonstationary at 1% level': t[0]>t[4]['1%'],

       'Nonstationary at 5% level': t[0]>t[4]['5%'],

       'Nonstationary at 10% level': t[0]>t[4]['10%']}

    df_t=pd.DataFrame(d,index=[i])

    df=df.append(df_t)

df
fig, axs = plt.subplots(5,2)

font = {'weight' : 'normal',

        'size'   : 8}

plt.rc('font', **font)



for i in range(0,len(rn)):

    ax = plt.subplot(5,2,i+1)

    sns.distplot(train_data.y.loc[rn[i]],

                 norm_hist=True,kde=True,

                 ax=ax,

                 label='ID={}'.format(rn[i]))

    plt.legend()

    if i in [8,9]:

        ax.set_xlabel('y value')

    if i in range(0,9,2):

        ax.set_ylabel('Prob. Density')    
from scipy.signal import welch



fig, axs = plt.subplots(5,2)

font = {'weight' : 'normal',

        'size'   : 8}

plt.rc('font', **font)



for i in range(0,len(rn)):

    ax = plt.subplot(5,2,i+1)

    f,pxx=welch(train_data.y.loc[rn[i]],

                return_onesided=True,

                scaling='density')

    ax.plot(f,pxx,label='ID={}'.format(rn[i]))

    plt.legend()

    if i in [8,9]:

        ax.set_xlabel('Frequency')

    if i in range(0,9,2):

        ax.set_ylabel('Pxx')  
train_data = train_data.sample(frac=0.1)

train_data.fillna(train_data.mean(axis=0), inplace=True)





train_X = train_data.drop('y',axis=1)

train_Y = train_data.y

print("Data for model: X={}, y={}".format(train_X.shape, train_Y.shape))
model = xgb.XGBRegressor()

print("Fitting...")

model.fit(train_X, train_Y)

print("Fitting done")
fig, ax = plt.subplots(figsize=(7, 30))

xgb.plot_importance(model, ax=ax)
from sklearn import linear_model as lm

cols_to_use = ['technical_30']

models_dict = {}

for col in cols_to_use:

    model = lm.LinearRegression()

    model.fit(np.array(train_data[col]).reshape(-1,1), np.array(train.y))

    models_dict[col] = model

    

col = 'technical_30'

model = models_dict[col]

while True:

    test_x = np.array(train_data.features[col].values).reshape(-1,1)

    observation.target.y = model.predict(test_x)

    #observation.target.fillna(0, inplace=True)

    target = observation.target

    timestamp = observation.features["timestamp"][0]

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))



    observation, reward, done, info = env.step(target)

    if done:

        break