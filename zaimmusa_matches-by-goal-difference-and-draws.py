# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



con = sqlite3.connect('../input/database v2.sqlite')

countries = pd.read_sql_query("select * from country", con)

matches = pd.read_sql_query("select * from match", con)

leagues = pd.read_sql_query("select * from league", con)

teams = pd.read_sql_query("select * from team", con)

player_stats = pd.read_sql_query("select * from player_attributes", con)

tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con)

players = pd.read_sql_query("select * from player", con)
matches['goal_diff'] = pd.Series(matches['home_team_goal'] - matches['away_team_goal'], index = matches.index)



matches_by_goal_diff = matches.groupby(['goal_diff']).count()['id']

matches_by_goal_diff.plot(kind='bar')



diff_by_one = matches_by_goal_diff[0]+matches_by_goal_diff[-1]+matches_by_goal_diff[1];

print (round(100*(diff_by_one)/matches.shape[0], 2), " % of matches is decided by one goal. ")



print (round(100*matches_by_goal_diff[0]/matches.shape[0], 2), " % matches were draws. " )
betting_houses = ('B365', 'BW', 'IW', 'LB', 'PS', 'WH', 'SJ', 'VC', 'GB', 'BS')

length = len(betting_houses)

draw_odds_mean = [np.mean(matches[house + 'D']) for house in betting_houses]

plt.bar(np.arange(length), draw_odds_mean, align='center', alpha=0.5)

plt.xticks(np.arange(length), betting_houses)

plt.ylabel('Odds')

plt.xlabel('Betting houses')

plt.title('Draw odds')

plt.axis([0, length, 3.5, 4.5])

plt.grid(True)

plt.show()

print (np.max(draw_odds_mean))
def best_draw_odd(match):

    draw_odds = [match[house+'D'] for house in betting_houses if str(match[house+'D']) != 'nan']

    if (len(draw_odds) == 0):

        return 0.0;

    return np.max(draw_odds)



def calculateProfit(matches):

    profit = 0.0

    count = 0

    for intex, row in matches.iterrows():

        best_draw_odds = best_draw_odd(row)

        if (best_draw_odds == 0.0):

            continue;

        if row['home_team_goal'] == row['away_team_goal']:

            profit += best_draw_odds 

        profit -= 1.0 # money invested

        count += 1

    if (count == 0):

        return profit, profit

    return profit, profit/count



profit, avg_profit = calculateProfit(matches)

print ('Profit : ', profit)

print ('Profit per match', avg_profit)
def fcl(df, date):

    return df.ix[np.argmin(np.abs(to_timestamp(df['date_stat']) - date))]['overall_rating']



def avg_overall_rating(side, match):

    result = 0.0

    count = 0

    for i in range(1,12):

        if not pd.isnull(match[side + str(i)]):

            result += get_player_overall_rating(int(match[side + str(i)]) , match['date'])

            count += 1

    if count == 0:

        return None;

    return result/count 



def avg_home_overall_rating(match):

    return avg_overall_rating('home_player_', match)



def avg_away_overall_rating(match):

    return avg_overall_rating('away_player_', match)



def get_player_overall_rating(player_api_id, on_date):

    return fcl(player_stats.loc[player_stats['player_api_id'] == player_api_id], to_timestamp(on_date))



def to_timestamp(string):

    return pd.to_datetime(string, infer_datetime_format=True)
import warnings

warnings.filterwarnings("ignore")



this_year_matches = matches[matches['date'] > '2016-01-01']

this_year_matches[['home_player_1', 'home_player_2', 'home_player_3', 'home_player_4']].head()



#ser = this_year_matches.apply(lambda x: avg_home_overall_rating(x), axis=1)

#this_year_matches['home_avg_overall_rating'] = ser



#ser2 = this_year_matches.apply(lambda x: avg_away_overall_rating(x), axis=1)

#this_year_matches['away_avg_overall_rating'] = ser2



#this_year_matches = this_year_matches[!pd.isnull(this_year_matches['home_avg_overall_rating'] and !pd.isnull(this_year_matches['away_avg_overall_rating'])]



#this_year_matches['draw'] = this_year_matches['home_team_goal'] == this_year_matches['away_team_goal']



#import matplotlib

#matplotlib.style.use('ggplot')



#ax = this_year_matches[this_year_matches['draw']].plot.scatter(x='home_avg_overall_rating', y='away_avg_overall_rating', color='DarkRed', label='Draws')

#this_year_matches[this_year_matches['draw'] == False].plot.scatter(x='home_avg_overall_rating', y='away_avg_overall_rating', color='DarkBlue', label='Not draws', ax=ax)

#plt.show()