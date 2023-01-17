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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
mydf = pd.read_csv('../input/ipl/matches.csv')

mydf
matches_in_2017 = ((mydf['city']=='Mumbai') & (mydf['season']==2017)).value_counts()[1]

matches_in_2016 = ((mydf['city']=='Mumbai') & (mydf['season']==2016)).value_counts()[1]

matches_in_2015 = ((mydf['city']=='Mumbai') & (mydf['season']==2015)).value_counts()[1]

matches_in_2014 = ((mydf['city']=='Mumbai') & (mydf['season']==2014)).value_counts()[1]

matches_in_2013 = ((mydf['city']=='Mumbai') & (mydf['season']==2013)).value_counts()[1]

matches_in_total = matches_in_2013 + matches_in_2014 + matches_in_2015 + matches_in_2016 + matches_in_2017

print('Number of matches played in Mumbai in the period of 2013 and 2017 : ',matches_in_total)

margin_of_victory = (mydf['win_by_runs']>30).value_counts()[1]

print('Number of matches where the margin of victory was greater than 30 : ',margin_of_victory)
total_matches_played = (mydf['team1']=='Kolkata Knight Riders').value_counts()[1] + (mydf['team2']=='Kolkata Knight Riders').value_counts()[1]

batted_by_toss = ((mydf['toss_winner']=='Kolkata Knight Riders') & (mydf['toss_decision']=='bat')).value_counts()[1]

percentage_of_batted_by_toss = (batted_by_toss/total_matches_played)*100

print('Percent times KKR decides to bat after winning the toss : ',percentage_of_batted_by_toss)
mask_frame=(mydf['season']>=2010) & (mydf['season']<=2015) & (mydf['city']=="Mumbai")

player=mydf[mask_frame]['player_of_match'].value_counts().head(1).index[0]



print("Name of the player who won max number of man of the matches award in the period of 2010 and 2015 in Mumbai :",player )
mask_frame2=mydf['win_by_runs']>50

team_name=mydf[mask_frame2]['winner'].value_counts().head(1).index[0]



print("the team who has won most number of matches with victory margin > 50 : ", team_name)
def winning_stat(team1,team2):

    mask_frame3=((mydf['team1']==team1)& (mydf['team2']==team2)) | ((mydf['team1']==team2)& (mydf['team2']==team1))

    win_result=mydf[mask_frame3]['winner'].value_counts()

    #return win_result

    print("{} {}(matches won by {}) and {} {}(matches won by {})".format(win_result.index[0],win_result.values[0],win_result.index[0],win_result.index[1],win_result.values[1],win_result.index[1]))

    

winning_stat("Kolkata Knight Riders","Chennai Super Kings")
def winning_percentage(team):

    mask_frame_4=((mydf['team1']==team) | (mydf['team2']==team)) & (mydf['toss_winner']==team)

    mask_frame_5=((mydf['team1']==team) | (mydf['team2']==team)) & (mydf['toss_winner']==team) & (mydf['winner']==team)

    total_matches_won=mydf[mask_frame_4]['winner'].value_counts().values.sum()

    matches_won_when_toss_won=mydf[mask_frame_5]['winner'].value_counts().values[0]

    win_percentage=(matches_won_when_toss_won/total_matches_won)*100

    print("Predicted win percentage of", team ,"after winning  the toss is : " , win_percentage , "%")

    

winning_percentage("Mumbai Indians")