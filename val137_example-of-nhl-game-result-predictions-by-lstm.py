# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

from datetime import datetime

from datetime import timedelta



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
game = pd.read_csv('/kaggle/input/nhl-game-data/game.csv', sep=',')

teaminfo = pd.read_csv('/kaggle/input/nhl-game-data/team_info.csv', sep=',')

gameteamstats = pd.read_csv('/kaggle/input/nhl-game-data/game_teams_stats.csv', sep=',')
def ResultGeneration(date, VisitorId, HomeId):

    VisitorGoals = game.loc[(game['away_team_id'] == VisitorId) & (game['date_time'] == date)]['away_goals'].tolist()[0]

    HomeGoals = game.loc[(game['home_team_id'] == HomeId) & (game['date_time'] == date)]['home_goals'].tolist()[0]

    if VisitorGoals > HomeGoals:

        result = 1

    else:

        result = 0

    return result



def LSTMMatrixGenerationV(date, TeamId, samples, tmsteps, ftrs):

    time_steps = tmsteps

    features = ftrs



    LSTMMartix = np.zeros((samples,time_steps, features),float)



    for sc in range(samples):

        for tc in range(time_steps):

            counter = samples+time_steps-sc-tc-1

            gamesort = game[(game['date_time'] < date) & (game['away_team_id'] == TeamId)][['game_id']].sort_values('game_id', ascending=False)

            gameid = gamesort.game_id.tolist()[counter]

            LSTMMartix[sc][tc][0] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == TeamId) & (gameteamstats['game_id'] > gameid-1000000)][['goals']].mean().tolist()[0]

            LSTMMartix[sc][tc][1] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == TeamId) & (gameteamstats['game_id'] > gameid - 1000000)][['shots']].mean().tolist()[0]

            LSTMMartix[sc][tc][2] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == TeamId) & (gameteamstats['game_id'] > gameid - 1000000)][['hits']].mean().tolist()[0]

            LSTMMartix[sc][tc][3] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == TeamId) & (gameteamstats['game_id'] > gameid - 1000000)][['pim']].mean().tolist()[0]

            LSTMMartix[sc][tc][4] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == TeamId) & (gameteamstats['game_id'] > gameid - 1000000)][['powerPlayOpportunities']].mean().tolist()[0]

            LSTMMartix[sc][tc][5] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == TeamId) & (gameteamstats['game_id'] > gameid - 1000000)][['powerPlayGoals']].mean().tolist()[0]

            LSTMMartix[sc][tc][6] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == TeamId) & (gameteamstats['game_id'] > gameid - 1000000)][['faceOffWinPercentage']].mean().tolist()[0]

            LSTMMartix[sc][tc][7] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == TeamId) & (gameteamstats['game_id'] > gameid - 1000000)][['giveaways']].mean().tolist()[0]

            LSTMMartix[sc][tc][8] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == TeamId) & (gameteamstats['game_id'] > gameid - 1000000)][['takeaways']].mean().tolist()[0]



            HomeIdv = game.loc[game['game_id'] == gameid]['home_team_id'].tolist()[0]



            gamesort = game[(game['date_time'] < date) & (game['home_team_id'] == HomeIdv)][['game_id']].sort_values('game_id', ascending=False)

            gameid = gamesort.game_id.tolist()[counter]

            LSTMMartix[sc][tc][9] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == HomeIdv) & (gameteamstats['game_id'] > gameid - 1000000)][['goals']].mean().tolist()[0]

            LSTMMartix[sc][tc][10] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == HomeIdv) & (gameteamstats['game_id'] > gameid - 1000000)][['shots']].mean().tolist()[0]

            LSTMMartix[sc][tc][11] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == HomeIdv) & (gameteamstats['game_id'] > gameid - 1000000)][['hits']].mean().tolist()[0]

            LSTMMartix[sc][tc][12] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == HomeIdv) & (gameteamstats['game_id'] > gameid - 1000000)][['pim']].mean().tolist()[0]

            LSTMMartix[sc][tc][13] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == HomeIdv) & (gameteamstats['game_id'] > gameid - 1000000)][['powerPlayOpportunities']].mean().tolist()[0]

            LSTMMartix[sc][tc][14] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == HomeIdv) & (gameteamstats['game_id'] > gameid - 1000000)][['powerPlayGoals']].mean().tolist()[0]

            LSTMMartix[sc][tc][15] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == HomeIdv) & (gameteamstats['game_id'] > gameid - 1000000)][['faceOffWinPercentage']].mean().tolist()[0]

            LSTMMartix[sc][tc][16] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == HomeIdv) & (gameteamstats['game_id'] > gameid - 1000000)][['giveaways']].mean().tolist()[0]

            LSTMMartix[sc][tc][17] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == HomeIdv) & (gameteamstats['game_id'] > gameid - 1000000)][['takeaways']].mean().tolist()[0]



    return LSTMMartix



def LSTMMatrixGenerationH(date, TeamId, samples, tmsteps, ftrs):

    time_steps = tmsteps

    features = ftrs



    LSTMMartix = np.zeros((samples, time_steps, features),float)



    for sc in range(samples):

        for tc in range(time_steps):

            counter = samples+time_steps-sc-tc-1

            gamesort = game[(game['date_time'] < date) & (game['home_team_id'] == TeamId)][['game_id']].sort_values('game_id', ascending=False)

            gameid = gamesort.game_id.tolist()[counter]

            LSTMMartix[sc][tc][9] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == TeamId) & (gameteamstats['game_id'] > gameid-1000000)][['goals']].mean().tolist()[0]

            LSTMMartix[sc][tc][10] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == TeamId) & (gameteamstats['game_id'] > gameid - 1000000)][['shots']].mean().tolist()[0]

            LSTMMartix[sc][tc][11] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == TeamId) & (gameteamstats['game_id'] > gameid - 1000000)][['hits']].mean().tolist()[0]

            LSTMMartix[sc][tc][12] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == TeamId) & (gameteamstats['game_id'] > gameid - 1000000)][['pim']].mean().tolist()[0]

            LSTMMartix[sc][tc][13] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == TeamId) & (gameteamstats['game_id'] > gameid - 1000000)][['powerPlayOpportunities']].mean().tolist()[0]

            LSTMMartix[sc][tc][14] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == TeamId) & (gameteamstats['game_id'] > gameid - 1000000)][['powerPlayGoals']].mean().tolist()[0]

            LSTMMartix[sc][tc][15] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == TeamId) & (gameteamstats['game_id'] > gameid - 1000000)][['faceOffWinPercentage']].mean().tolist()[0]

            LSTMMartix[sc][tc][16] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == TeamId) & (gameteamstats['game_id'] > gameid - 1000000)][['giveaways']].mean().tolist()[0]

            LSTMMartix[sc][tc][17] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == TeamId) & (gameteamstats['game_id'] > gameid - 1000000)][['takeaways']].mean().tolist()[0]



            VisitorIdh = game.loc[game['game_id'] == gameid]['away_team_id'].tolist()[0]



            gamesort = game[(game['date_time'] < date) & (game['home_team_id'] == VisitorIdh)][['game_id']].sort_values('game_id', ascending=False)

            gameid = gamesort.game_id.tolist()[counter]

            LSTMMartix[sc][tc][0] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == VisitorIdh) & (gameteamstats['game_id'] > gameid - 1000000)][['goals']].mean().tolist()[0]

            LSTMMartix[sc][tc][1] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == VisitorIdh) & (gameteamstats['game_id'] > gameid - 1000000)][['shots']].mean().tolist()[0]

            LSTMMartix[sc][tc][2] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == VisitorIdh) & (gameteamstats['game_id'] > gameid - 1000000)][['hits']].mean().tolist()[0]

            LSTMMartix[sc][tc][3] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == VisitorIdh) & (gameteamstats['game_id'] > gameid - 1000000)][['pim']].mean().tolist()[0]

            LSTMMartix[sc][tc][4] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == VisitorIdh) & (gameteamstats['game_id'] > gameid - 1000000)][['powerPlayOpportunities']].mean().tolist()[0]

            LSTMMartix[sc][tc][5] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == VisitorIdh) & (gameteamstats['game_id'] > gameid - 1000000)][['powerPlayGoals']].mean().tolist()[0]

            LSTMMartix[sc][tc][6] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == VisitorIdh) & (gameteamstats['game_id'] > gameid - 1000000)][['faceOffWinPercentage']].mean().tolist()[0]

            LSTMMartix[sc][tc][7] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == VisitorIdh) & (gameteamstats['game_id'] > gameid - 1000000)][['giveaways']].mean().tolist()[0]

            LSTMMartix[sc][tc][8] = gameteamstats.loc[(gameteamstats['game_id'] < gameid) & (gameteamstats['team_id'] == VisitorIdh) & (gameteamstats['game_id'] > gameid - 1000000)][['takeaways']].mean().tolist()[0]



    return LSTMMartix
matches = 0                                # Counter for games, get 20 games

VisitorKoef = 0.                           # Average Deviation for Visitor predictions

HomeKoef = 0.                              # Average Deviation for Visitor predictions

RightV = 0                                 # Right predictions of Visitor games         

RightH = 0                                 # Right predictions of Home games 

ADVoutput = np.zeros((50), float)          # Array for outputs for average difference beetwen result and  Visitor predicton 

ADHoutput = np.zeros((50), float)          # Array for outputs for average difference beetwen result and Home predicton

RVoutput = np.zeros((50), float)           # Array for outputs 

RHoutput = np.zeros((50), float)           # Array for outputs
for gameid in range(2018020001,2018020051):



    matches = matches+1



    VisitorId = game.loc[game['game_id'] == gameid]['away_team_id'].tolist()[0]

    HomeId = game.loc[game['game_id'] == gameid]['home_team_id'].tolist()[0]

    date = game.loc[game['game_id'] == gameid]['date_time'].tolist()[0]

    

    time_steps = 5        # Number of previous games takes in to account 

    features = 18         # Number of features



    if VisitorId == 54 or HomeId == 54:

        samples = 40         # Because Vegas(TeamID=54) start in 2017

    else:

        samples = 150







    LSTMv = LSTMMatrixGenerationV(date, VisitorId, samples, time_steps, features)

    LSTMh = LSTMMatrixGenerationH(date, HomeId, samples, time_steps, features)



    ResultV = np.zeros((samples), int)     # Array for result generation(visitor)

    gamesort = game[(game['date_time'] < date) & (game['away_team_id'] == VisitorId)][['game_id', 'away_team_id', 'home_team_id', 'date_time']].sort_values('game_id', ascending=False)

    for sc in range(samples):

        cr=samples-sc-1

        gameidforresult = gamesort.game_id.tolist()[cr]

        visitoridforresult = gamesort.away_team_id.tolist()[cr]

        homeidforresult = gamesort.home_team_id.tolist()[cr]

        dateforresult = gamesort.date_time.tolist()[cr]

        ResultV[sc] = ResultGeneration(dateforresult, visitoridforresult, homeidforresult)



    ResultH = np.zeros((samples), int)     # Array for result generation(home)

    gamesort = game[(game['date_time'] < date) & (game['home_team_id'] == HomeId)][['game_id', 'away_team_id', 'home_team_id', 'date_time']].sort_values('game_id', ascending=False)

    for sc in range(samples):

        cr=samples-sc-1

        gameidforresult = gamesort.game_id.tolist()[cr]

        visitoridforresult = gamesort.away_team_id.tolist()[cr]

        homeidforresult = gamesort.home_team_id.tolist()[cr]

        dateforresult = gamesort.date_time.tolist()[cr]

        ResultH[sc] = ResultGeneration(dateforresult, visitoridforresult, homeidforresult)



    nextday = datetime.strptime(date, "%Y-%m-%d").date()+timedelta(days=1)

    nextday = str(nextday)



    PredictDataV = LSTMMatrixGenerationV(nextday, VisitorId, 1, time_steps, features)      # Get LSTM matrix for prediction(Visitor)

    PredictDataH = LSTMMatrixGenerationH(nextday, HomeId, 1, time_steps, features)         # Get LSTM matrix for prediction(Home)

    result = ResultGeneration(date, VisitorId, HomeId)



    batch_size = 4

    epochs = 30



    model = Sequential()

    model.add(LSTM(50, activation='relu', input_shape=(time_steps, features)))

    model.add(Dropout(0.3))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')



    model.fit(LSTMv, ResultV, epochs = epochs, batch_size = batch_size, verbose=0)

    predict = model.predict(PredictDataV)



    print('Visitor:', predict, result)

    VisitorKoef = VisitorKoef + abs(predict-result)



    if abs(predict-result) < 0.5:

        RightV = RightV+1



    model.fit(LSTMh, ResultH, epochs = epochs, batch_size = batch_size, verbose=0)

    predict = model.predict(PredictDataH)



    print('Home:', predict, result)

    HomeKoef = HomeKoef + abs(predict - result)



    if abs(predict-result) < 0.5:

        RightH = RightH+1



    print(gameid)

    AvgDiffV = VisitorKoef/matches

    print('Difference Visitor:',AvgDiffV)

    AvgDiffH= HomeKoef/matches

    print('Difference Home:',AvgDiffH)

    

    index = matches-1

    print('index=',index)

    

    print('Right Visitor:',RightV/matches*100,'%')

    print('Right Home:', RightH/matches*100,'%')

    ADVoutput[index] = AvgDiffV

    ADHoutput[index] = AvgDiffH

    RVoutput[index] = RightV/matches

    RHoutput[index] = RightH/matches



gamescounter = []



for x in range(1,51):

    gamescounter.append(x)

    

plt.figure(figsize=(16,12))



plt.plot(gamescounter, ADVoutput,label ='AvgDiffV')

plt.plot(gamescounter, ADHoutput,label ='AvgDiffH')

plt.plot(gamescounter, RVoutput,label ='RightKoefV')

plt.plot(gamescounter, RHoutput,label ='RightKoefH') 



plt.legend()



plt.show()