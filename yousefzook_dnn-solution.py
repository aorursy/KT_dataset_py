# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/train_V2.csv')

data.head()
print(data.info())

data.describe()
match_types = data['matchType'].unique()

print(match_types, "\nTotal:",len(match_types))
fig, ax = plt.subplots()

fig.set_size_inches(8, 6)

plt.xticks(rotation=90)

sns.countplot(data['matchType'])
data.boxplot(rot=90, figsize=(12,8))
non_numerical_cols = ['Id', 'groupId', 'matchId', 'matchType']
# features = [col for col in data.columns if col not in non_numerical_cols]

# fig, axes = plt.subplots(25, 2, figsize=(15,60))

# i = 0

# for feature in features:

#         data[feature].plot.box(ax=axes[i,0])

#         sns.distplot(data[feature], hist=True, kde=True, ax=axes[i,1], bins=len(data[feature])//1000)

# #         data[feature][0:200].plot.hist(density=True, ax=axes[i,1])

#         i += 1
solo_matches_df = data[data['matchType'] == 'solo']

solo_groups = solo_matches_df['groupId']

print("unique values len:", len(solo_groups.unique()), "len:" 

      ,len(solo_groups), "rate:", len(solo_groups.unique())/len(solo_groups))
solo_samples = data[data['matchType'].isin([ 'solo-fpp',  'solo',  'normal-solo-fpp', 'normal-solo'])]

duo_samples = data[data['matchType'].isin([ 'duo','duo-fpp','normal-duo-fpp', 'normal-duo'])]

squad_samples = data[data['matchType'].isin([ 'squad-fpp', 'squad', 'normal-squad-fpp',

                     'crashfpp', 'flaretpp','flarefpp','normal-squad', 'crashtpp'])]

print("There are {} ({:.2f}%) solo games, {} ({:.2f}%) duo games and {} ({:.2f}%) squad games.".format(len(solo_samples), 

    100*len(solo_samples)/len(data), len(duo_samples), 

    100*len(duo_samples)/len(data), len(squad_samples), 100*len(squad_samples)/len(data),))
f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='kills',y='winPlacePerc',data=solo_samples,color='black',alpha=0.8)

sns.pointplot(x='kills',y='winPlacePerc',data=duo_samples,color='#CC0000',alpha=0.8)

sns.pointplot(x='kills',y='winPlacePerc',data=squad_samples,color='#3399FF',alpha=0.8)

plt.text(37,0.6,'Solos',color='black',fontsize = 17,style = 'italic')

plt.text(37,0.55,'Duos',color='#CC0000',fontsize = 17,style = 'italic')

plt.text(37,0.5,'Squads',color='#3399FF',fontsize = 17,style = 'italic')

plt.xlabel('Number of kills',fontsize = 15,color='blue')

plt.ylabel('Win Percentage',fontsize = 15,color='blue')

plt.title('Solo vs Duo vs Squad Kills',fontsize = 20,color='blue')

plt.grid()

plt.show()
f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='kills',y='winPlacePerc',data=solo_samples,color='black',alpha=0.8)

sns.pointplot(x='kills',y='winPlacePerc',data=duo_samples,color='#CC0000',alpha=0.8)

sns.pointplot(x='kills',y='winPlacePerc',data=squad_samples,color='#3399FF',alpha=0.8)

plt.text(37,0.6,'Solos',color='black',fontsize = 17,style = 'italic')

plt.text(37,0.55,'Duos',color='#CC0000',fontsize = 17,style = 'italic')

plt.text(37,0.5,'Squads',color='#3399FF',fontsize = 17,style = 'italic')

plt.xlabel('Number of kills',fontsize = 15,color='blue')

plt.ylabel('Win Percentage',fontsize = 15,color='blue')

plt.title('Solo vs Duo vs Squad Kills',fontsize = 20,color='blue')

plt.grid()

plt.show()
solo_samples[solo_samples['DBNOs']!=0]
f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
import random

def train_test_split(df, test_size=0.2):



    # remove 'Nan' match IDs

    invalid_match_ids = df[df['winPlacePerc'].isna()]['matchId'].values

    df = df[-df['matchId'].isin(invalid_match_ids)]



    match_ids = df['matchId'].unique().tolist()

    train_size = int(len(match_ids) * (1 - test_size))

    train_match_ids = random.sample(match_ids, train_size)

    

    train = df[df['matchId'].isin(train_match_ids)]

    test = df[-df['matchId'].isin(train_match_ids)]

    X_train, y_train = train.drop(columns=['winPlacePerc']), train['winPlacePerc']

    X_test, y_test = test.drop(columns=['winPlacePerc']), test['winPlacePerc']

    return X_train, y_train, X_test, y_test
X_train, y_train, X_test, y_test = train_test_split(data)

print("X_train shape:",X_train.shape)

print("y_train shape:",y_train.shape)

print("X_test shape:",X_test.shape)

print("y_test shape:",y_test.shape)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

import time

import gc



def LinearRegression_test(X_train, y_train, X_test, y_test, loss_function):

    '''

    data: data with all numerical varibles to being tested

    return: Mean squared error

    '''    

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return loss_function(y_test, y_pred)



def drop_categorical(df):

    return df._get_numeric_data()



def test(preprocesses, loss_function=mean_squared_error):

    '''

    preprocesses: list of preprocessing functions to be tested

    return: list of each test with it's score sorted

    '''

    results = []

    for preprocess in preprocesses:

        print(" ============= Testing:", preprocess.__name__ , "=============")

        start = time.time()

        data_copy = data

        X_train, y_train, X_test, y_test = train_test_split(preprocess(data_copy))

        score = LinearRegression_test(drop_categorical(X_train), y_train, drop_categorical(X_test), y_test, loss_function)

        execution_time = time.time() - start

        results.append({

            'name': preprocess.__name__,

            'score': score,

            'execution time': f'{round(execution_time, 2)}s'

        })

        print("score:", score, " - ", "time:", f'{round(execution_time, 2)}s')

        print()

        gc.collect()

        

    return pd.DataFrame(results, columns=['name', 'score', 'execution time']).sort_values(by='score')
def original(df):

    return df



def heals_boosts_merge(df):

    df['items'] = df['heals'] + df['boosts']

    return df



def team_size(df):

    agg = df.groupby(['groupId']).size().to_frame('players_in_team')

    return df.merge(agg, how='left', on=['groupId'])



def total_distance(df):

    df['total_distance'] = df['rideDistance'] + df['swimDistance'] + df['walkDistance']

    return df



def headshotKills_over_kills(df):

    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']

    df['headshotKills_over_kills'].fillna(0, inplace=True)

    return df



def killPlace_over_maxPlace(df):

    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']

    df['killPlace_over_maxPlace'].fillna(0, inplace=True)

    df['killPlace_over_maxPlace'].replace(np.inf, 0, inplace=True)

    return df



def walkDistance_over_heals(df):

    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']

    df['walkDistance_over_heals'].fillna(0, inplace=True)

    df['walkDistance_over_heals'].replace(np.inf, 0, inplace=True)

    return df



def walkDistance_over_kills(df):

    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']

    df['walkDistance_over_kills'].fillna(0, inplace=True)

    df['walkDistance_over_kills'].replace(np.inf, 0, inplace=True)

    return df



def assists_revives_merge(df):

    df['teamwork'] = df['assists'] + df['revives']

    return df



def DBNOs_kills_merge(df):

    df['DBNOsAndKills'] = df['DBNOs'] + df['kills']

    return df



def estimated_elo_ranking(df):

    df['estimatedEloRanking'] = (df['killPoints'] + df['winPoints'] - df['rankPoints']) / 3

    df['estimatedEloRanking'].fillna(0, inplace=True)

    df['estimatedEloRanking'].replace(np.inf, 0, inplace=True)

    return df

# mean squared error loss function

test([

    original,

    heals_boosts_merge,

    team_size,

    total_distance,

    headshotKills_over_kills,

    killPlace_over_maxPlace,

    walkDistance_over_heals,

    walkDistance_over_kills,

    assists_revives_merge,

    DBNOs_kills_merge,

    estimated_elo_ranking

])
# mean absolute error loss function

test([

    original,

    heals_boosts_merge,

    team_size,

    total_distance,

    headshotKills_over_kills,

    killPlace_over_maxPlace,

    walkDistance_over_heals,

    walkDistance_over_kills,

    assists_revives_merge,

    DBNOs_kills_merge,

    estimated_elo_ranking], 

    loss_function=mean_absolute_error)
data = heals_boosts_merge(data)

data = team_size(data)

data = total_distance(data)

data = headshotKills_over_kills(data)

data = killPlace_over_maxPlace(data)

data = walkDistance_over_heals(data)

data = walkDistance_over_kills(data)

data = assists_revives_merge(data)

data = DBNOs_kills_merge(data)

data = estimated_elo_ranking(data)







X_train, y_train, X_test, y_test = train_test_split(data)

print("X_train shape:",X_train.shape)

print("y_train shape:",y_train.shape)

print("X_test shape:",X_test.shape)

print("y_test shape:",y_test.shape)
from keras.optimizers import Adam

import time



def train_test_pipeline(model, lr, beta_1, beta_2, epochs=10):

  

  # compile and fit model

  start = time.time()

  adam = Adam(lr=lr, decay=1e-6, beta_1=beta_1, beta_2=beta_2)

  model.compile(loss='mean_absolute_error',

                optimizer=adam,

                metrics=['mse'])

  model.fit(drop_categorical(X_train), y_train, validation_split=0.2, epochs=epochs, verbose=0)

  end = time.time()



  # plot results

  print("================== Total time to fit_train is: ", (end-start)/60,"minutes ==================")

  print()

  results = model.history.history



  plt.figure(figsize=[20,9])

  plt.subplot(2,2,1)

  plt.plot(results['mean_squared_error'], label='Training mse')

  plt.legend()



  plt.figure(figsize=[20,9])

  plt.subplot(2,2,2)

  plt.plot(results['loss'], label='Training loss')

  plt.legend()



  plt.figure(figsize=[20,9])

  plt.subplot(2,2,3)

  plt.plot(results['val_mean_squared_error'], label='validation mse')

  plt.legend()



  plt.figure(figsize=[20,9])

  plt.subplot(2,2,4)

  plt.plot(results['val_loss'], label='validation loss')

  plt.legend()



  # test the model

  print("================== Model Evaluation is running ==================")

  loss_mae_test, mse_test = model.evaluate(x=drop_categorical(X_test), y=y_test)

  print("================== mae on test data:", loss_mae_test, "--- mse on test data:", mse_test, "==================")
from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU

from keras.initializers import VarianceScaling



model = Sequential()



# 1st group

model.add(Dense(256, input_dim=drop_categorical(X_train).shape[-1], name='Group_1'))

model.add(BatchNormalization(momentum=0.9))

model.add(LeakyReLU())

model.add(Dropout(0.1))



# 2nd group

model.add(Dense(128, name='Group_2'))

model.add(BatchNormalization(momentum=0.9))

model.add(LeakyReLU())

model.add(Dropout(0.2))



# 3rd group

model.add(Dense(64, name='Group_3'))

model.add(BatchNormalization(momentum=0.9))

model.add(LeakyReLU())

model.add(Dropout(0.3))



# 4th group

model.add(Dense(64, name='Group_4'))

model.add(BatchNormalization(momentum=0.9))

model.add(LeakyReLU())

model.add(Dropout(0.2))



# 5th group

model.add(Dense(32, name='Group_5'))

model.add(BatchNormalization(momentum=0.9))

model.add(LeakyReLU())

model.add(Dropout(0.3))



# 6th group

model.add(Dense(32, name='Group_6'))

model.add(BatchNormalization(momentum=0.9))

model.add(LeakyReLU())

model.add(Dropout(0.2))



# 7th group

model.add(Dense(16, name='Group_7'))

model.add(BatchNormalization(momentum=0.9))

model.add(LeakyReLU())

model.add(Dropout(0.3))



# 8th group

model.add(Dense(32, name='Group_8'))

model.add(BatchNormalization(momentum=0.9))

model.add(LeakyReLU())

model.add(Dropout(0.2))



# 9th group

model.add(Dense(16, name='Group_9'))

model.add(BatchNormalization(momentum=0.9))

model.add(LeakyReLU())

model.add(Dropout(0.3))



#output

model.add(Dense(1, activation='sigmoid', name='output'))

model.summary()
train_test_pipeline(model, 0.01, 0.99, 0.99, epochs=2)
test_data = pd.read_csv('../input/test_V2.csv')

test_data.head()


test_data = heals_boosts_merge(test_data)

test_data = team_size(test_data)

test_data = total_distance(test_data)

test_data = headshotKills_over_kills(test_data)

test_data = killPlace_over_maxPlace(test_data)

test_data = walkDistance_over_heals(test_data)

test_data = walkDistance_over_kills(test_data)

test_data = assists_revives_merge(test_data)

test_data = DBNOs_kills_merge(test_data)

test_data = estimated_elo_ranking(test_data)

output = model.predict(drop_categorical(test_data))
output.shape
result = pd.concat([test_data['Id'], pd.DataFrame(output)],ignore_index=True, axis=1)

result.columns = ['Id', 'winPlacePerc']

result.head()
result.to_csv('submission.csv', index=False)
result = pd.read_csv('submission.csv')

result.head()