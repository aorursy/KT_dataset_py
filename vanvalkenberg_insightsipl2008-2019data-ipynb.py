###########################################################
#          work in Progress                               #
###########################################################
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
DFmatches = pd.read_csv('/kaggle/input/ipl-data-set/matches.csv')
DFmatches.describe()
DFmatches.loc[0:10]
def Wins(DataFrame):
    teamNames = set(DataFrame['team1'])
    teamList = []
    winList = []
    
    for mem in teamNames:
        teamList.append(mem)
        winList.append(len (DataFrame[DataFrame['winner'] == mem]) )
    
    plt.figure(figsize=(10,5))
    plt.bar(teamList, winList)
    plt.xlabel('Team Names')
    plt.ylabel('No. of Wins')
    plt.title('IPL 2008-2019: Bar graphs of Matches won by teams ')
    plt.grid('True')
    plt.xticks(rotation = 'vertical')
    plt.show()
Wins(DFmatches)
## modifing the above Function to get the player with most man of the matches
def ManOfTheMatch(DataFrame):
    playerNames = set(DataFrame['player_of_match'])
    playerNameList =[]
    playerWinList = []
    
    for mem in playerNames:
        wons = len(DataFrame[DataFrame['player_of_match'] == mem])
        if wons < 5:
            continue
        
        playerNameList.append(mem)
        playerWinList.append(wons)
    
    playerNameList = np.array(playerNameList)
    playerWinList = np.array(playerWinList)
    
    plt.figure(figsize=(20,10))
    plt.bar(playerNameList, playerWinList)
    plt.xlabel('Player Names')
    plt.ylabel('No. of times title ManOmatch')
    plt.title('IPL 2008-2019: Bar graph of Man of match titles (more than 5 titles won) ')
    plt.grid('True')
    plt.xticks(rotation = 'vertical')
    plt.show()
    
    
 
ManOfTheMatch(DFmatches)
##lets write a function to do a team wise analysis of the data by writting a function
def Analyzer(DataFrame):
    nameOfTeams = set(DataFrame['team1'])
    
    for mem in nameOfTeams:
        total_matchesPlayed = len(DataFrame[  (DataFrame['team1'] == mem) | (DataFrame['team2'] == mem)      ])
        total_matchesWon = len(DataFrame[  ((DataFrame['team1'] == mem) | (DataFrame['team2'] == mem)) & (DataFrame['winner'] == mem)])
        winRatio = total_matchesWon /  total_matchesPlayed
        winsWhenBat = len(DataFrame[  ((DataFrame['team1'] == mem) | (DataFrame['team2'] == mem)) & (DataFrame['winner'] == mem) & (DataFrame['toss_decision'] == 'bat')] )
        
        print('=====================================================================\n')
        print ('Team Name:{}'.format(mem))
        print('Total matches Played:{}'.format(total_matchesPlayed))
        print('Total matches Won:{}'.format(total_matchesWon))
        print('Total matches Lost:{}'.format(total_matchesPlayed - total_matchesWon ))
        print('Wining Percentage:{}'.format(winRatio * 100))
        print('Winining ratio when Choosed to bat:{}'.format((winsWhenBat/total_matchesWon) * 100))
        print('Winning ratio when Choosed to field:{}'.format((1 - (winsWhenBat/total_matchesWon))*100))
        print('=====================================================================\n')
        
        
        
Analyzer(DFmatches)
## let's See another dataFrame 
DFrunsAvgStrikeRate = pd.read_csv('/kaggle/input/ipl-data-set/most_runs_average_strikerate.csv')
DFrunsAvgStrikeRate.describe()
DFrunsAvgStrikeRate.loc[0 : 9]
## let Write a simple function to proccess this DataFrame and show the top players according to runs, average, and strike rate
def Analaze2ndDF(DataFrame):
    categories = ['total_runs', 'average', 'strikerate']
    
    for mem in categories:
        DataFrame.sort_values(by = mem, ascending = False, inplace = True)
        print('=====================================================================\n')
        print ('Top 10 player accoring to '+mem+':')
        print(DataFrame.head(10))
        print('=====================================================================\n')
Analaze2ndDF(DFrunsAvgStrikeRate)
## lets look at this data set /kaggle/input/ipl-data-set/Players.xlsx
DFplayer = pd.read_excel('/kaggle/input/ipl-data-set/Players.xlsx')
DFplayer.loc[0:9]
## 