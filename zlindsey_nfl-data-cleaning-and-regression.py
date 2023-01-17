# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from math import sqrt

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from math import sin

from math import cos

from math import radians
raw_data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv')

raw_data.head()
numplays = len(raw_data.index)

print(numplays)

raw_data.isna().sum()/numplays*100
rowdrops = ['FieldPosition', 'DefendersInTheBox', 'Orientation', 'Dir', 'OffenseFormation']



badplays = raw_data[raw_data[rowdrops].isna().sum(axis=1) > 0]['PlayId']

badrows = raw_data[raw_data['PlayId'].isin(badplays)]

raw_data.drop(badrows.index, inplace=True)

raw_data.shape[0]/22

raw_data.drop(['DisplayName', 'PlayerCollegeName'], axis=1, inplace=True)



raw_data.drop(['TimeHandoff', 'TimeSnap'], axis=1, inplace=True)



raw_data.drop(['Temperature', 'Humidity'], axis=1, inplace=True)



parse_later = ['WindSpeed',

               'WindDirection',

               'GameWeather',

               'Turf',

               'StadiumType',

               'Location',

               'Stadium',

              'OffensePersonnel']

raw_data.drop(parse_later, axis=1, inplace=True)
raw_data.isna().sum()
dirSign = raw_data['PlayDirection'] == 'right'



raw_data['X'] = raw_data['X']*dirSign+(120-raw_data['X'])*~dirSign

raw_data['Dir'] = raw_data['Dir']*dirSign - raw_data['Dir']*~dirSign

raw_data['Orientation'] = raw_data['Orientation']*dirSign - raw_data['Orientation']*~dirSign



raw_data.drop(['PlayDirection'], axis=1, inplace=True)

# standardize the abbreviations



def abbrConv(abbr):

    '''

    convert from the XTeamAbbr and fieldPosition to PossesionTeam

    '''

    if abbr == 'ARI':

        return 'ARZ'

    elif abbr == 'BAL':

        return 'BLT'

    elif abbr == 'CLE':

        return 'CLV'

    elif abbr == 'HOU':

        return 'HST'

    else:

        return abbr

    

raw_data['HomeTeamAbbr'] = raw_data['HomeTeamAbbr'].apply(abbrConv)

raw_data['VisitorTeamAbbr'] = raw_data['VisitorTeamAbbr'].apply(abbrConv)

raw_data['FieldPosition'] = raw_data['FieldPosition'].apply(abbrConv)
possmask = raw_data['FieldPosition'] == raw_data['PossessionTeam']

raw_data['YardLineX'] = possmask*(raw_data['YardLine']+10) + (1-possmask)*(110-raw_data['YardLine'])
raw_data['VX'] = raw_data['X'].apply(lambda a: sin(radians(a)))

raw_data['VY'] = raw_data['Y'].apply(lambda a: cos(radians(a)))
def getAge(birthday):

    # epxress birthday in years old

    return (pd.Timestamp.now() - pd.Timestamp(birthday)).days/365



raw_data['Age'] = raw_data['PlayerBirthDate'].apply(getAge)

raw_data.drop('PlayerBirthDate', axis=1, inplace=True)
def parseHeight(height):

    # convert from ft-in to just inches

    feet, inches = map(int, height.split('-'))

    return 12*feet + inches

raw_data['PlayerHeight'] = raw_data['PlayerHeight'].apply(parseHeight)
runner_table = raw_data[raw_data['NflId'] == raw_data['NflIdRusher']]

raw_data = raw_data.merge(runner_table[['PlayId','X','Y']], on='PlayId', suffixes=('','Runner'))

raw_data['SqDistToRunner'] = (raw_data['X']-raw_data['XRunner'])**2 + (raw_data['Y']-raw_data['YRunner'])**2
raw_data['X'] = raw_data['X'] - raw_data['XRunner']

raw_data['Y'] = raw_data['Y'] - raw_data['YRunner']

raw_data.drop(['XRunner','YRunner'], axis=1, inplace=True)
def gameClock_to_seconds(clock):

    # convert mm:ss:xx to seconds

    minutes, seconds, _ = map(int, clock.split(':'))

    return 60*minutes+seconds



raw_data['GameClock'] = raw_data['GameClock'].apply(gameClock_to_seconds)
# first, add a column to indicate the team abbr

homemask = (raw_data['Team'] == 'home')

raw_data.loc[homemask,'TeamAbbr'] = raw_data[homemask]['HomeTeamAbbr']

raw_data.loc[~homemask,'TeamAbbr'] = raw_data[~homemask]['VisitorTeamAbbr']



raw_data['Offense'] = raw_data['TeamAbbr'] == raw_data['PossessionTeam']

raw_data.drop('PossessionTeam', axis=1, inplace=True)



raw_data['foomyscore'] = homemask*raw_data['HomeScoreBeforePlay'] + (1-homemask)*raw_data['VisitorScoreBeforePlay']

raw_data['footheirscore'] = ~homemask*raw_data['HomeScoreBeforePlay'] + (1-~homemask)*raw_data['VisitorScoreBeforePlay']

raw_data['OffenseScore'] = raw_data['Offense']*raw_data['foomyscore'] + ~raw_data['Offense']*raw_data['footheirscore']

raw_data['DefenseScore'] = ~raw_data['Offense']*raw_data['foomyscore'] + raw_data['Offense']*raw_data['footheirscore']

raw_data.drop(['foomyscore','footheirscore'], axis=1, inplace=True)

raw_data.columns
play_data = raw_data[['GameId',

                      'PlayId',

                      'Season',

                      'YardLine',

                      'Quarter',

                      'GameClock',

                      'Down',

                      'Distance',

                      'HomeScoreBeforePlay',

                      'VisitorScoreBeforePlay',

                      'NflIdRusher',

                      'OffenseFormation',

                      'DefendersInTheBox',

                      'Yards',

                      'HomeTeamAbbr',

                      'VisitorTeamAbbr',

                      'Week',

                      'YardLineX',

                      'OffenseScore',

                      'DefenseScore']]

play_data = play_data.iloc[::22,:]
# now get the player data

player_data = raw_data[['PlayId',

                        'Team',

                        'TeamAbbr',

                        'X',

                        'Y',

                        'S',

                        'A',

                        'VX',

                        'VY',

                        'SqDistToRunner',

                        'Dis',

                        'Orientation',

                        'Dir',

                        'NflId',

                        'JerseyNumber',

                        'PlayerHeight',

                        'PlayerWeight',

                        'Position',

                        'Offense'

                       ]]



player_data.sort_values(['PlayId','Offense','SqDistToRunner'], inplace=True)

player_data
fat_table = play_data

for i in range(22):

    fat_table = fat_table.merge(player_data.iloc[i::22,], on='PlayId', suffixes=['',str(i)])
fat_table
fat_table.dtypes.value_counts()



objectcols = []

boolcols = []

for col in fat_table.columns:

    if fat_table.dtypes[col] == 'object':

        objectcols.append(col)

    elif fat_table.dtypes[col] == 'bool':

        boolcols.append(col)

        

from sklearn.preprocessing import LabelEncoder

objEncoders = []

for col in objectcols:

    encoder = LabelEncoder()

    fat_table[col] = encoder.fit_transform(fat_table[col])

    objEncoders.append( (col,encoder))



from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet



from sklearn.metrics import mean_squared_error as MSE

from sklearn.metrics import r2_score as R2



from sklearn.model_selection import train_test_split



from sklearn.preprocessing import StandardScaler



features = [col for col in fat_table.columns if col != 'Yards']

X = fat_table[features]

y = fat_table['Yards']



# linreg - no hyperparams

# ridge - alpha

# lasso - alpha

# elastic net - alpha, l1_ratio


X_train, X_test, y_train, y_test = train_test_split(fat_table[features], fat_table['Yards'])

X_train = StandardScaler().fit_transform(X_train)

linreg = LinearRegression().fit(X_train, y_train)



X_test = StandardScaler().fit_transform(X_test)

yhat = linreg.predict(X_test)





print('RMSE: ', sqrt(MSE(yhat, y_test)))

print('R2: ', R2(yhat, y_test))
from sklearn.model_selection import GridSearchCV





alphas = [0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000,50000,100000]



ridge_search = GridSearchCV(Ridge(), param_grid={'alpha':alphas}, cv=5)



Xscaled = StandardScaler().fit_transform(X)

ridge_search.fit(Xscaled, y)
ridge_search.cv_results_
ax = plt.bar(range(1,1+len(alphas)), ridge_search.cv_results_['mean_test_score'], yerr=ridge_search.cv_results_['std_test_score'])

plt.gca().set_xticklabels(alphas, rotation=50)

best_ridge = ridge_search.best_estimator_

alphas = [0.01,0.05,0.1,0.5,1,2,4]



lasso_search = GridSearchCV(Lasso(), param_grid={'alpha':alphas}, cv=5)



Xscaled = StandardScaler().fit_transform(X)

lasso_search.fit(Xscaled, y)
lasso_search.cv_results_
ax = plt.bar(range(1,1+len(alphas)), lasso_search.cv_results_['mean_test_score'], yerr=lasso_search.cv_results_['std_test_score'])

plt.gca().set_xticklabels(alphas, rotation=50)

best_lasso = lasso_search.best_estimator_
lasso_coeffs = pd.Series(best_lasso.coef_, index=features)

lasso_coeffs.sort_values()
good_coeffs = lasso_coeffs[abs(lasso_coeffs) > 0.1].index

good_coeffs
d_dist_cols = ['SqDistToRunner'] + ['SqDistToRunner' + str(i) for i in range(1,11)]

Ddists = fat_table[d_dist_cols]

o_dist_cols = ['SqDistToRunner' + str(i) for i in range(11,22)]

Odists = fat_table[o_dist_cols]

for i in range(1,10):

    fat_table['Owithin' + str(i)] = Odists[Odists < i**2].count(axis=1)

    fat_table['Dwithin' + str(i)] = Ddists[Ddists < i**2].count(axis=1)

    fat_table['diffwithin' + str(i)] = fat_table['Owithin'+ str(i)] - fat_table['Dwithin'+ str(i)]

j = 5

print(fat_table['diffwithin'+str(j)].unique())



for i in fat_table['diffwithin'+str(j)].unique():

    plt.hist(fat_table[fat_table['diffwithin'+str(j)] == i]['Yards'], alpha=.4, bins=20)

plt.legend()

plt.show()
features = [col for col in fat_table.columns if col != 'Yards']

X = fat_table[features]

y = fat_table['Yards']


X_train, X_test, y_train, y_test = train_test_split(fat_table[features], fat_table['Yards'])

X_train = StandardScaler().fit_transform(X_train)

linreg = LinearRegression().fit(X_train, y_train)



X_test = StandardScaler().fit_transform(X_test)

yhat = linreg.predict(X_test)





print('SMSE: ', sqrt(MSE(yhat, y_test)))

print('R2: ', R2(yhat, y_test))
alphas = [0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000,50000,100000]



ridge_search = GridSearchCV(Ridge(), param_grid={'alpha':alphas}, cv=5)



Xscaled = StandardScaler().fit_transform(X)

ridge_search.fit(Xscaled, y)
ridge_search.cv_results_
ax = plt.bar(range(1,1+len(alphas)), ridge_search.cv_results_['mean_test_score'], yerr=ridge_search.cv_results_['std_test_score'])

plt.gca().set_xticklabels(alphas, rotation=50)

best_ridge = ridge_search.best_estimator_
alphas = [0.0001,0.005,0.01,0.05,0.1,0.5,1,2,4]



lasso_search = GridSearchCV(Lasso(), param_grid={'alpha':alphas}, cv=5)



Xscaled = StandardScaler().fit_transform(X)

lasso_search.fit(Xscaled, y)
lasso_search.cv_results_
ax = plt.bar(range(0,len(alphas)), lasso_search.cv_results_['mean_test_score'], yerr=lasso_search.cv_results_['std_test_score'])

plt.gca().set_xticks(range(len(alphas)))

plt.gca().set_xticklabels(alphas, rotation=50, ha='right')

best_lasso = lasso_search.best_estimator_
lasso_coeffs = pd.Series(best_lasso.coef_, index=features)

lasso_coeffs.sort_values()
good_coeffs = lasso_coeffs[abs(lasso_coeffs) > 0.1].index

good_coeffs
F = good_coeffs

X2 = fat_table[F]

y2 = fat_table['Yards']

LinearRegression().fit(X2,y2).score(X2,y2)