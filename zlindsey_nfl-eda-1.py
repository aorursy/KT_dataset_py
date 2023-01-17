# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from math import cos

from math import sin

from math import radians



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
raw_data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv')
raw_data.head()
raw_data.dtypes
plt.figure(figsize=(20,10))

plt.title("Plays per Game")

raw_data['GameId'].value_counts().plot(kind='bar')

raw_data.groupby(['PlayId']).count().describe()
len(raw_data['GameId'].unique())
def randomPlay():

    '''

    Returns a random valid (GameId,PlayId) pair

    '''

    N = raw_data.shape[0]

    i = np.random.randint(0,N)

    return (raw_data.iloc[i,:].GameId, raw_data.iloc[i,:].PlayId)



randomPlay()
raw_data['Team'].value_counts()
def graphPlay1(GameId, PlayId):

    play_data = raw_data[(raw_data['GameId'] == GameId) & (raw_data['PlayId'] == PlayId)]

    colors = {'away':'red', 'home':'blue'}

    for i in play_data.index:

        plt.plot(play_data.loc[i,'X'], play_data.loc[i,'Y'], 'ro', markersize=10, color=colors[play_data.loc[i,'Team']])

    plt.show()

    return None
graphPlay1(*randomPlay())
ax_x = raw_data['X'].plot(kind='hist', bins = int(np.max(raw_data['X'])-np.min(raw_data['X'])))

plt.title('Distribution of X')

ax_x.axvline(10, color='r')

ax_x.axvline(110, color='r')

ax_x.axvline(35, color='y')

plt.show()
ax_y = raw_data['Y'].plot(kind='hist', bins = int(np.max(raw_data['Y'])-np.min(raw_data['Y'])))

plt.title('Distribution of Y')

hashdist = 23 + 1/3 + 9/36 # how far the hashes are from the sideline

ax_y.axvline(hashdist, color='y')

ax_y.axvline(53.3-hashdist, color='y')

def graphPlay2(GameId, PlayId):

    '''

    Makes a visual depiction of the play.

    The ball carrier is marked yellow.

    The arrows indicate the direction of movement of the player. 

    The length of the black arrow is proportional to the speed.

    The length of the orange arrow is proportional to the accel. 

    The direction of accel is not provided, and so it is plotted in the direction of motion.

    '''

    play_data = raw_data[(raw_data['GameId'] == GameId) & (raw_data['PlayId'] == PlayId)]

    colors = {'away':'red', 'home':'blue'}

    print(play_data.shape)

    play_data.index = range(22)

    print(play_data.loc[0,'OffenseFormation'])

    print(play_data.loc[0,'PlayDirection'])

    print(play_data.loc[0,'TimeHandoff'])

    print(play_data.loc[0,'Yards'])

    for i in play_data.index:

        playerInfo = play_data.loc[i,:]

        if playerInfo['NflId'] != playerInfo['NflIdRusher']:

            plt.plot('X', 'Y', 'ro', markersize=10, color=colors[play_data.loc[i,'Team']], data = playerInfo)

        else:

            plt.plot('X', 'Y', 'ro', markersize=10, color='yellow', data = playerInfo)

        plt.arrow(playerInfo['X'], playerInfo['Y'], playerInfo['S']*sin(radians(playerInfo['Dir'])), playerInfo['S']*cos(radians(playerInfo['Dir'])), head_width=0.5)

        plt.arrow(playerInfo['X'], playerInfo['Y'], playerInfo['A']*sin(radians(playerInfo['Dir'])), playerInfo['A']*cos(radians(playerInfo['Dir'])),color='orange',head_width=0.5)

    plt.show()

    return None
graphPlay2(*randomPlay())
raw_data.groupby(['GameId','PlayId']).count()
raw_data['S'].plot(kind='hist', bins=50)

plt.title('Speeds of players')

plt.show()



raw_data['A'].plot(kind='hist', bins=50)

plt.title('Accels of players')

plt.show()



raw_data['Dir'].plot(kind='hist', bins=50)

plt.title('Angle of motion of players')
raw_data['Orientation'].plot(kind='hist', bins= 50, alpha=.9)

raw_data['Dir'].plot(kind='hist', bins= 50, alpha=.4)

plt.legend(bbox_to_anchor = (1.1,0.5), bbox_transform = plt.gcf().transFigure)

plt.xlabel('Angle')

plt.title('Player orientation and movement direction')
def abbrConv(abbr):

    '''

    convert from the XTeamAbbr to PossesionTeam

    see code cell below for why we need this

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



def isOffense(row):

    offense = row['PossessionTeam']

    side = row['Team']

    if side == 'away':

        key = 'VisitorTeamAbbr'

    else:

        key = 'HomeTeamAbbr'

    if offense == abbrConv(row[key]):

        return True

    else:

        return False

    

offenseMask = raw_data.apply(isOffense, axis=1)

        
print(2*offenseMask.sum())

len(raw_data.index)


# annoying inconsistency: There are different sets of abbreviations for the two teams! GRR.

team_data = pd.DataFrame(raw_data['PossessionTeam'].value_counts())

team_data = pd.concat([team_data, raw_data['HomeTeamAbbr'].value_counts(), raw_data['VisitorTeamAbbr'].value_counts()], axis = 1)

team_data
# now that we have the offense mask, let's replot some of the old data with it

raw_data['S'][offenseMask].plot(kind='hist', bins=50, color='r', alpha=.5)

raw_data['S'][~offenseMask].plot(kind='hist', bins=50,alpha=.5)

plt.title('Speeds of players')

plt.legend(['Offense','Defense'])

plt.show()



raw_data['A'][offenseMask].plot(kind='hist', bins=50,color='r', alpha=.5)

raw_data['A'][~offenseMask].plot(kind='hist', bins=50, alpha=.5)

plt.title('Accels of players')

plt.legend(['Offense','Defense'])

plt.show()



raw_data['Dir'][offenseMask].plot(kind='hist', bins=50, color='r',alpha=.5)

raw_data['Dir'][~offenseMask].plot(kind='hist', bins=50, alpha=.5)

plt.title('Angle of motion of players')

plt.legend(['Offense','Defense'])

plt.show()
id_name_pairs = raw_data[['NflId', 'DisplayName']].drop_duplicates()

id_name_counts = id_name_pairs.groupby('NflId').count().reset_index().rename(columns = {'DisplayName':'Count'})

id_name_counts.describe()
double_name_IDs = id_name_counts[id_name_counts['Count'] > 1]['NflId']

id_name_pairs[id_name_pairs['NflId'].isin(double_name_IDs)].sort_values('NflId')
raw_data['JerseyNumber'].plot(kind='hist', bins=50)

# popular jersy numbers, weighted by number of plays!
raw_data['Season'].value_counts().plot(kind='bar')
raw_data['YardLine'].plot(kind='hist', bins=50)
raw_data['FieldPosition'].value_counts().plot(kind='bar')


field_pos_data = pd.DataFrame(raw_data['FieldPosition'].value_counts())

field_pos_data = pd.concat([team_data, raw_data['HomeTeamAbbr'].value_counts(), raw_data['VisitorTeamAbbr'].value_counts()], axis = 1)

field_pos_data
def yardline_to_X(row):

    poss_team = row['PossessionTeam']

    field_team = row['FieldPosition']

    yardline = row['YardLine']

    print("Poss:", poss_team)

    print("Field:", field_team)

    print("Yardline:", yardline)

    print(row['PlayDirection'])

    if row['PlayDirection'] == 'left':

        # <<<

        if poss_team == field_team:

            return 110-yardline

        else:

            return 10+yardline

    else:

        # >>>

        if poss_team == field_team:

            return 10+yardline

        else:

            return 110-yardline

        

    

#raw_data.apply(yardLine_to_X, axis=1).plot(kind='hist', bins=100)
def graphPlay3(GameId, PlayId):

    '''

    Makes a visual depiction of the play.

    The ball carrier is marked yellow.

    The arrows indicate the direction of movement of the player. 

    The length of the black arrow is proportional to the speed.

    The length of the orange arrow is proportional to the accel. 

    The direction of accel is not provided, and so it is plotted in the direction of motion.

    The dotted blue line is the line of scrimmage

    '''

    play_data = raw_data[(raw_data['GameId'] == GameId) & (raw_data['PlayId'] == PlayId)]

    colors = {'away':'red', 'home':'blue'}

    print(play_data.shape)

    play_data.index = range(22)

    print(play_data.loc[0,'OffenseFormation'])

    print(play_data.loc[0,'PlayDirection'])

    print(play_data.loc[0,'TimeHandoff'])

    print(play_data.loc[0,'Yards'])

    ax = plt.gca()

    scrimmX = yardline_to_X(play_data.loc[0,:])

    ax.axvline(scrimmX, ls='--')

    

    if play_data.loc[0,'PlayDirection'] == 'left':

        downline = scrimmX-play_data.loc[0,'Distance']

    else:

        downline = scrimmX+play_data.loc[0,'Distance']

    ax.axvline(downline, ls='--', color='orange')

    for i in play_data.index:

        playerInfo = play_data.loc[i,:]

        if playerInfo['NflId'] != playerInfo['NflIdRusher']:

            plt.plot('X', 'Y', 'ro', markersize=10, color=colors[play_data.loc[i,'Team']], data = playerInfo)

        else:

            plt.plot('X', 'Y', 'ro', markersize=10, color='yellow', data = playerInfo)

        plt.arrow(playerInfo['X'], playerInfo['Y'], playerInfo['S']*sin(radians(playerInfo['Dir'])), playerInfo['S']*cos(radians(playerInfo['Dir'])), head_width=0.5)

        plt.arrow(playerInfo['X'], playerInfo['Y'], playerInfo['A']*sin(radians(playerInfo['Dir'])), playerInfo['A']*cos(radians(playerInfo['Dir'])),color='orange',head_width=0.5)

    plt.show()

    return None
graphPlay3(*randomPlay())
raw_data['GameClock'].value_counts()
raw_data[raw_data['GameClock']=='00:00:00']
graphPlay3(2017110600, 20171106003613)
raw_data['Quarter'].value_counts()



def gameClock(PlayId):

    row = raw_data[raw_data['PlayId'] == PlayId]

    res = 'Q' + str(row.iloc[0]['Quarter']) + ' - ' + str(row.iloc[0]['GameClock'])

    return res

    

gameClock(randomPlay()[1])
raw_data['Down'].value_counts().plot(kind='bar')
for i in range(1,5):

    plays = raw_data[raw_data['Down'] == i]

    plays['Yards'].plot(kind='hist', bins=20)

    plt.title('Yard on down ' + str(i))

    print('Average', plays['Yards'].mean(), 'yards on down', i)

    plt.show()
def downAndDist(playId):

    row = raw_data[raw_data['PlayId'] == playId].iloc[0]

    return str(row['Down']) + ' and ' + str(row['Distance'])



downAndDist(randomPlay()[1])
def fieldPos(playId):

    row = raw_data[raw_data['PlayId'] == playId].iloc[0]

    return row['FieldPosition'] + ' ' + str(row['YardLine'])



fieldPos(randomPlay()[1])
raw_data['HomeScoreBeforePlay'].value_counts().plot(kind='bar')

plt.show()



raw_data['VisitorScoreBeforePlay'].value_counts().plot(kind='bar')

plt.show()
def scoreString(playId):

    row = raw_data[raw_data['PlayId'] == playId].iloc[0]

    return row['HomeTeamAbbr'] + ' ' + str(row['HomeScoreBeforePlay']) + '\n' + row['VisitorTeamAbbr'] + ' ' + str(row['VisitorScoreBeforePlay'])



print(scoreString(randomPlay()[1]))

                                     
raw_data['OffenseFormation'].value_counts()
raw_data['OffensePersonnel'].value_counts()
print(raw_data['DefendersInTheBox'].value_counts())



raw_data[['DefendersInTheBox', 'Yards']].groupby('DefendersInTheBox').mean().plot(kind='bar')
raw_data['DefensePersonnel'].value_counts()
def handoffDelay(playId):

    row = raw_data[raw_data['PlayId']==playId].iloc[0]

    snap = pd.Timestamp(row['TimeSnap'])

    handoff = pd.Timestamp(row['TimeHandoff'])

    return (handoff-snap).seconds

    

def handoffDelayFromRow(row):

    snap = pd.Timestamp(row['TimeSnap'])

    handoff = pd.Timestamp(row['TimeHandoff'])

    return (handoff-snap).seconds



handoffs = raw_data.apply(handoffDelayFromRow, axis=1)
handoffDelay(randomPlay()[1])
handoffs.value_counts().plot(kind='bar')
raw_data['PlayerWeight'].plot(kind='hist', bins=50)

plt.show()



raw_data['PlayerHeight'].value_counts().plot(kind='bar')

plt.show()



print(raw_data['PlayerCollegeName'].value_counts())



# make a graph of player's ages

raw_data['PlayerBirthDate'].apply(lambda x: int((pd.Timestamp.now() - pd.Timestamp(x)).days/365)).value_counts().plot(kind='bar')
sigma = raw_data['Yards'].std()

mu = raw_data['Yards'].mean()

errors = sigma/np.sqrt(raw_data.groupby('Week')['Yards'].count())



ax = raw_data.groupby('Week')['Yards'].mean().plot(kind='bar', yerr=2*errors)

ax.axhline(raw_data['Yards'].mean())

plt.show()

means = []

errs = []

plt.gca().axhline(raw_data['Yards'].mean())

for i in range(17):

    sample = raw_data.sample(frac=1/17)

    means.append(sample['Yards'].mean())

    errs.append(2*sigma/np.sqrt(sample['Yards'].count()))

plt.bar(range(17), means, yerr=errs)

raw_data['StadiumType'].value_counts()
raw_data['Turf'].value_counts()
pd.set_option('display.max_rows', 1000)

print(raw_data['GameWeather'].value_counts())

pd.set_option('display.max_rows', 10)
raw_data['Temperature'].plot(kind='hist',bins=20)
raw_data['Humidity'].plot(kind='hist', bins=20)
raw_data['WindSpeed'].value_counts()
raw_data['WindDirection'].value_counts()
raw_data['Dis'].plot(kind='hist', bins=50)